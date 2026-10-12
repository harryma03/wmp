import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class ZGM1(LeggedRobot):
    """ZG-M1 with position-controlled legs and velocity-controlled wheels."""

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        scales = cfg.noise.noise_scales
        level = cfg.noise.noise_level
        start = self.privileged_dim - 3
        dofs = cfg.env.num_actions

        noise_vec[start:start + 3] = scales.lin_vel * level * self.obs_scales.lin_vel
        noise_vec[start + 3:start + 6] = scales.ang_vel * level * self.obs_scales.ang_vel
        noise_vec[start + 6:start + 9] = scales.gravity * level
        noise_vec[start + 12:start + 12 + dofs] = scales.dof_pos * level * self.obs_scales.dof_pos
        noise_vec[start + 12 + dofs:start + 12 + 2 * dofs] = scales.dof_vel * level * self.obs_scales.dof_vel
        if cfg.terrain.measure_heights:
            noise_vec[start + 12 + 3 * dofs:] = scales.height_measurements * level * self.obs_scales.height_measurements
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.wheel_dof_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "FOOT_JOINT" in name],
            device=self.device,
            dtype=torch.long,
        )
        self.leg_dof_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "FOOT_JOINT" not in name],
            device=self.device,
            dtype=torch.long,
        )
        if len(self.wheel_dof_indices) != 4 or len(self.leg_dof_indices) != 12:
            raise ValueError(
                f"ZG-M1 expects 12 leg joints and 4 FOOT_JOINT wheels, got "
                f"{len(self.leg_dof_indices)} and {len(self.wheel_dof_indices)}"
            )

    def _compute_torques(self, actions):
        leg_targets = (
            actions[:, self.leg_dof_indices] * self.cfg.control.leg_action_scale
            + self.default_dof_pos[:, self.leg_dof_indices]
        )
        torques = torch.zeros_like(actions)
        torques[:, self.leg_dof_indices] = (
            self.p_gains[self.leg_dof_indices]
            * (leg_targets - self.dof_pos[:, self.leg_dof_indices])
            - self.d_gains[self.leg_dof_indices] * self.dof_vel[:, self.leg_dof_indices]
        )

        wheel_targets = actions[:, self.wheel_dof_indices] * self.cfg.control.wheel_action_scale
        torques[:, self.wheel_dof_indices] = self.d_gains[self.wheel_dof_indices] * (
            wheel_targets - self.dof_vel[:, self.wheel_dof_indices]
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def get_amp_observations(self):
        # A1 demonstrations contain 12 joints; wheels are task-controlled, not AMP-controlled.
        return torch.cat(
            (
                self.dof_pos[:, self.leg_dof_indices],
                self.base_lin_vel,
                self.base_ang_vel,
                self.dof_vel[:, self.leg_dof_indices],
            ),
            dim=-1,
        )

    def _reward_dof_acc(self):
        # Wheel angular acceleration is expected during rolling; penalize leg chatter only.
        leg_acc = (
            self.last_dof_vel[:, self.leg_dof_indices]
            - self.dof_vel[:, self.leg_dof_indices]
        ) / self.dt
        return torch.sum(torch.square(leg_acc), dim=1)

    def _reward_action_rate(self):
        action_delta = self.last_actions - self.actions
        leg_rate = torch.sum(torch.square(action_delta[:, self.leg_dof_indices]), dim=1)
        wheel_rate = torch.sum(torch.square(action_delta[:, self.wheel_dof_indices]), dim=1)
        return leg_rate + 0.1 * wheel_rate

    def _reward_dof_error(self):
        """Keep the 12 leg joints near the nominal pose without penalizing wheel rotation."""
        return torch.sum(
            torch.square(
                self.dof_pos[:, self.leg_dof_indices]
                - self.default_dof_pos[:, self.leg_dof_indices]
            ),
            dim=1,
        )

