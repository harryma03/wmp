import numpy as np
import torch
import warp as wp


def depth_image_preprocessing(depth_image, near_plane=0.0, far_plane=2.0, depth_scale=1.0):
    near_mask = torch.abs(depth_image) < near_plane / depth_scale
    far_mask = torch.abs(depth_image) > far_plane / depth_scale
    depth_image[near_mask] = 0
    depth_image[far_mask] = far_plane / depth_scale
    if torch.isnan(depth_image).any() or torch.isinf(depth_image).any():
        raise RuntimeError("nan or inf detected in Warp depth image")
    return depth_image


@wp.kernel
def draw_depth(
    mesh: wp.uint64,
    cam_pos: wp.array(dtype=wp.float32),
    cam_rot: wp.array(dtype=wp.float32),
    width: int,
    height: int,
    depth: wp.array(dtype=wp.float32),
    fov_offset: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    batch = tid // (height * width)
    pixel = tid % (height * width)
    column = pixel % width
    row = pixel // width

    sy = -2.0 * float(column) / float(height) + float(width / height)
    sz = -2.0 * float(row) / float(height) + 1.0
    ray_origin = wp.vec3(
        cam_pos[batch * 3], cam_pos[batch * 3 + 1], cam_pos[batch * 3 + 2]
    )
    rotation = wp.mat33(
        cam_rot[batch * 9], cam_rot[batch * 9 + 1], cam_rot[batch * 9 + 2],
        cam_rot[batch * 9 + 3], cam_rot[batch * 9 + 4], cam_rot[batch * 9 + 5],
        cam_rot[batch * 9 + 6], cam_rot[batch * 9 + 7], cam_rot[batch * 9 + 8],
    )
    ray_direction = wp.normalize(wp.vec3(float(width / height) + fov_offset[batch], sy, sz))
    forward_component = ray_direction[0]
    ray_direction = rotation @ ray_direction
    query = wp.mesh_query_ray(mesh, ray_origin, ray_direction, 5.0)
    if query.result:
        depth[tid] = query.t * forward_component
    else:
        depth[tid] = 5.0


def euler_pos_to_matrix(position, euler):
    position = position.detach().cpu().numpy()
    euler = euler.detach().cpu().numpy()
    matrices = torch.zeros(position.shape[0], 4, 4)
    for i in range(position.shape[0]):
        roll, pitch, yaw = euler[i]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        matrices[i, :3, :3] = torch.tensor([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        matrices[i, :3, 3] = torch.from_numpy(position[i])
        matrices[i, 3, 3] = 1
    return matrices


def quat_pos_to_matrix(position, quaternion):
    batch = position.shape[0]
    x, y, z, w = quaternion.unbind(dim=1)
    rotation = torch.zeros(batch, 3, 3, dtype=position.dtype, device=position.device)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    matrix = torch.zeros(batch, 4, 4, dtype=position.dtype, device=position.device)
    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = position
    matrix[:, 3, 3] = 1
    return matrix


class DepthRendererWarp:
    def __init__(self, image_size, camera_position, camera_euler, horizontal_fov, device):
        self.height, self.width = image_size
        self.device = device
        self.camera_to_base = euler_pos_to_matrix(camera_position, camera_euler).to(device)
        self.warp_to_gym = torch.tensor(
            [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )
        self.fov_offset = 1.0 / torch.tan(torch.deg2rad(horizontal_fov) / 2) - 1.0
        self.mesh = None

    def render_mesh(self, vertices, triangles):
        with wp.ScopedDevice(self.device):
            vertices = (self.warp_to_gym[:3, :3].cpu().numpy() @ vertices.T).T
            self.mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                velocities=None,
                indices=wp.array(triangles.flatten(), dtype=int, device=self.device),
            )

    def render_depth(self, base_position, base_quaternion):
        batch = base_position.shape[0]
        depth = wp.zeros(batch * self.width * self.height, dtype=wp.float32, device=self.device)
        base_to_world = quat_pos_to_matrix(base_position, base_quaternion)
        camera_to_warp = self.warp_to_gym.unsqueeze(0) @ base_to_world @ self.camera_to_base
        camera_position = camera_to_warp[:, :3, 3].reshape(batch * 3)
        camera_rotation = camera_to_warp[:, :3, :3].reshape(batch * 9)
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=draw_depth,
                dim=batch * self.width * self.height,
                inputs=[
                    self.mesh.id,
                    wp.array(camera_position, dtype=wp.float32),
                    wp.array(camera_rotation, dtype=wp.float32),
                    self.width,
                    self.height,
                    depth,
                    wp.array(self.fov_offset, dtype=wp.float32),
                ],
                device=self.device,
            )
        return wp.to_torch(depth.reshape((batch, self.height, self.width)))
