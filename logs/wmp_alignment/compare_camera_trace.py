"""Compare recorded raster depth against training's pixel ray convention."""
import argparse
import json
import mujoco
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('scene')
parser.add_argument('trace')
args = parser.parse_args()
m = mujoco.MjModel.from_xml_path(args.scene)
d = mujoco.MjData(m)
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'depth_camera')
assert cam >= 0
with open(args.trace) as f:
    records = [json.loads(line) for line in f]
print('time,root_x,root_y,root_z,depth_mae_m,depth_p99_m,max_error_m')
for rec in records:
    h, w = rec['height'], rec['width']
    d.qpos[:] = rec['qpos']
    d.qvel[:] = rec['qvel']
    mujoco.mj_forward(m, d)
    y, x = np.mgrid[:h, :w]
    # Warp uses integer pixel coordinates, rather than raster pixel centers.
    rays = np.stack([(2*x-w)/h, 1-2*y/h,
                     np.full_like(x, -(w/h)/np.tan(np.deg2rad(58/2)), dtype=float)], -1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    forward = -rays[:, 2]
    dirs = rays @ d.cam_xmat[cam].reshape(3, 3).T
    ids = np.zeros(h*w, np.int32)
    dist = np.zeros(h*w)
    mujoco.mj_multiRay(m, d, d.cam_xpos[cam], dirs.ravel(),
                      np.array([1,1,0,0,1,1], np.uint8), 1, -1, ids, dist, h*w, 5)
    expected = np.clip(np.where(dist < 0, 5, dist*forward), 0, 2)
    actual = np.clip(np.array(rec['depth'], dtype=float), 0, 2)
    err = np.abs(expected-actual)
    print(','.join(f'{v:.6f}' for v in [rec['time'], *d.qpos[:3],
          np.nanmean(err), np.nanquantile(err, .99), np.nanmax(err)]))
