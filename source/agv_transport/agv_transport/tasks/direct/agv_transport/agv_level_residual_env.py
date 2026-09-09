from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from .agv_level_carry_lift_env import AgvLevelCarryLiftVisualEnv
from .agv_level_residual_env_cfg import AgvLevelResidualEnvCfg


class AgvLevelResidualEnv(AgvLevelCarryLiftVisualEnv):
    """Residual Lift task with scripted AGV translation.

    The V7.5 geometric + Board-feedback controller remains the baseline. The
    policy action contains only three normalized residual Lift commands.
    """

    cfg: AgvLevelResidualEnvCfg

    def __init__(self, cfg: AgvLevelResidualEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.base_leveling_target_height = torch.full_like(
            self.lift_height, float(self.cfg.lift_neutral_height)
        )
        self.last_residual_height = torch.zeros_like(self.lift_height)
        self.last_residual_saturated = torch.zeros_like(self.lift_height, dtype=torch.bool)

        self.randomized_target_speed = torch.full(
            (self.num_envs,), float(self.cfg.residual_scripted_speed), device=self.device
        )
        self.randomized_terrain_amplitude = torch.full(
            (self.num_envs,), float(self.cfg.bump_amplitude), device=self.device
        )
        self.randomized_terrain_phase_x = torch.full(
            (self.num_envs,), float(self.cfg.bump_phase_x), device=self.device
        )
        self.randomized_terrain_phase_y = torch.full(
            (self.num_envs,), float(self.cfg.bump_phase_y), device=self.device
        )
        self.randomized_cargo_mass = torch.full(
            (self.num_envs,), float(self.cfg.cargo_mass), device=self.device
        )
        self.randomized_cargo_offset_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.cargo_initial_relative_xy = torch.zeros((self.num_envs, 2), device=self.device)

        if self.cargo is None:
            raise RuntimeError("AgvLevelResidualEnv requires enable_cargo=True")
        self._cargo_reference_mass = float(self.cfg.cargo_mass)
        self._cargo_reference_inertias = self.cargo.root_physx_view.get_inertias().clone()
        self._validate_residual_cfg()

    def _validate_residual_cfg(self) -> None:
        if str(self.cfg.leveling_controller_mode) != "geometric_feedback":
            raise ValueError("Residual task requires leveling_controller_mode='geometric_feedback'")
        if float(self.cfg.residual_height_limit) <= 0.0:
            raise ValueError("residual_height_limit must be positive")
        max_speed = float(self.cfg.max_agv_linear_speed)
        if not 0.0 <= float(self.cfg.residual_scripted_speed) <= max_speed:
            raise ValueError("residual_scripted_speed is outside the AGV speed range")
        for name in (
            "residual_speed_range",
            "residual_terrain_amplitude_range",
            "residual_terrain_phase_x_range",
            "residual_terrain_phase_y_range",
            "residual_cargo_mass_range",
            "residual_cargo_offset_x_range",
            "residual_cargo_offset_y_range",
        ):
            low, high = (float(value) for value in getattr(self.cfg, name))
            if low > high:
                raise ValueError(f"{name} must be ordered, got {(low, high)}")
        speed_low, speed_high = self.cfg.residual_speed_range
        if float(speed_low) < 0.0 or float(speed_high) > max_speed:
            raise ValueError("residual_speed_range is outside the AGV speed range")
        mass_low, _ = self.cfg.residual_cargo_mass_range
        if float(mass_low) <= 0.0:
            raise ValueError("residual_cargo_mass_range must remain positive")

    # ------------------------------------------------------------------
    # Residual action and scripted AGV motion
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = torch.clamp(actions, -1.0, 1.0)

        # Compute V7.5 E first, then add only the bounded RL residual.
        self._update_leveling_controller()
        self.base_leveling_target_height[:] = self.lift_target_height
        self.last_residual_height[:] = self.actions * float(self.cfg.residual_height_limit)
        raw_target = self.base_leveling_target_height + self.last_residual_height
        lift_min = float(self.cfg.lift_min_height)
        lift_max = float(self.cfg.lift_max_height)
        self.last_residual_saturated[:] = (raw_target < lift_min) | (raw_target > lift_max)
        self.lift_target_height[:] = torch.clamp(raw_target, min=lift_min, max=lift_max)

    def _compute_linear_cmds(self) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_speed = torch.clamp(
            self.randomized_target_speed / float(self.cfg.max_agv_linear_speed),
            min=0.0,
            max=1.0,
        )
        commands = normalized_speed.unsqueeze(1).expand(-1, 3)
        return commands, commands

    def _compute_angular_speeds(self) -> torch.Tensor:
        applied = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.actions.dtype)
        self.last_rear_lateral_error.zero_()
        self.last_rear_lateral_correction.zero_()
        self.last_applied_angular_speed.zero_()
        return applied

    # ------------------------------------------------------------------
    # Per-environment analytical terrain
    # ------------------------------------------------------------------
    def _terrain_height_for_envs(
        self, local_xy: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        if not bool(self.cfg.enable_bumpy_support):
            return torch.zeros(local_xy.shape[0], device=self.device, dtype=local_xy.dtype)
        kx = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_x), 1.0e-6)
        ky = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_y), 1.0e-6)
        amplitude = self.randomized_terrain_amplitude[env_ids]
        phase_x = self.randomized_terrain_phase_x[env_ids]
        phase_y = self.randomized_terrain_phase_y[env_ids]
        return (
            amplitude
            * torch.sin(kx * local_xy[:, 0] + phase_x)
            * torch.sin(ky * local_xy[:, 1] + phase_y)
        )

    def _terrain_height(self, local_xy: torch.Tensor) -> torch.Tensor:
        # Base diagnostics use an env-major flattened sample layout.
        if not hasattr(self, "randomized_terrain_amplitude"):
            return super()._terrain_height(local_xy)
        if local_xy.shape[0] % self.num_envs != 0:
            raise ValueError(
                f"Terrain batch {local_xy.shape[0]} is not divisible by num_envs={self.num_envs}"
            )
        repeats = local_xy.shape[0] // self.num_envs
        env_ids = torch.arange(self.num_envs, device=self.device).repeat_interleave(repeats)
        return self._terrain_height_for_envs(local_xy, env_ids)

    def _sample_agv_terrain_plane(
        self, world_xy: torch.Tensor, yaw: torch.Tensor, env_xy: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        half_wheelbase = 0.5 * float(self.cfg.terrain_contact_wheelbase)
        half_track = 0.5 * float(self.cfg.terrain_contact_track)
        local_contacts = torch.tensor(
            (
                (half_wheelbase, half_track),
                (half_wheelbase, -half_track),
                (-half_wheelbase, half_track),
                (-half_wheelbase, -half_track),
            ),
            device=self.device,
            dtype=world_xy.dtype,
        )
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        offset_x = (
            cos_yaw * local_contacts[None, :, 0]
            - sin_yaw * local_contacts[None, :, 1]
        )
        offset_y = (
            sin_yaw * local_contacts[None, :, 0]
            + cos_yaw * local_contacts[None, :, 1]
        )
        sample_xy_w = world_xy[:, None, :] + torch.stack((offset_x, offset_y), dim=2)

        if env_xy is None:
            if world_xy.shape[0] != self.num_envs:
                raise ValueError("Full terrain updates must contain exactly one row per environment")
            env_xy = self.scene.env_origins[:, :2]
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            # Partial resets pass origins but not their indices. Resolve each
            # origin back to the vectorized environment index.
            origin_error = env_xy[:, None, :] - self.scene.env_origins[None, :, :2]
            env_ids = torch.argmin(torch.sum(torch.square(origin_error), dim=2), dim=1)

        local_samples = (sample_xy_w - env_xy[:, None, :]).reshape(-1, 2)
        sample_env_ids = env_ids.repeat_interleave(4)
        samples = self._terrain_height_for_envs(local_samples, sample_env_ids).reshape(-1, 4)

        front_mean = 0.5 * (samples[:, 0] + samples[:, 1])
        rear_mean = 0.5 * (samples[:, 2] + samples[:, 3])
        left_mean = 0.5 * (samples[:, 0] + samples[:, 2])
        right_mean = 0.5 * (samples[:, 1] + samples[:, 3])
        slope_x = (front_mean - rear_mean) / max(2.0 * half_wheelbase, 1.0e-6)
        slope_y = (left_mean - right_mean) / max(2.0 * half_track, 1.0e-6)
        pitch = -torch.atan(slope_x)
        roll = torch.atan(slope_y * torch.cos(pitch))
        return samples.mean(dim=1), roll, pitch, samples

    # ------------------------------------------------------------------
    # Residual policy observation
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lw, lx, ly, lz = left.unbind(dim=1)
        rw, rx, ry, rz = right.unbind(dim=1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=1,
        )

    @staticmethod
    def _quat_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qw, qx, qy, qz = quat.unbind(dim=1)
        roll = torch.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        return roll, pitch

    def _cargo_relative_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        board_quat_inverse = self.payload.data.root_quat_w.clone()
        board_quat_inverse[:, 1:4] *= -1.0
        relative_position = self._quat_rotate_vector(
            board_quat_inverse,
            self.cargo.data.root_pos_w - self.payload.data.root_pos_w,
        )
        relative_velocity = self._quat_rotate_vector(
            board_quat_inverse,
            self.cargo.data.root_lin_vel_w - self.payload.data.root_lin_vel_w,
        )
        relative_quat = self._quat_multiply(
            board_quat_inverse, self.cargo.data.root_quat_w
        )
        relative_roll, relative_pitch = self._quat_roll_pitch(relative_quat)
        relative_angular_speed = torch.linalg.norm(
            self.cargo.data.root_ang_vel_w - self.payload.data.root_ang_vel_w,
            dim=1,
        )
        return (
            relative_position,
            relative_velocity,
            relative_roll,
            relative_pitch,
            relative_angular_speed,
        )

    def _support_top_relative_heights(self) -> torch.Tensor:
        half_thickness = 0.5 * float(self.cfg.lift_plate_size[2])
        top_z = []
        for lift in self.lifts:
            local_z = self._quat_rotate_z(lift.data.root_quat_w)
            top_z.append(lift.data.root_pos_w[:, 2] + half_thickness * local_z[:, 2])
        heights = torch.stack(top_z, dim=1)
        return heights - heights.mean(dim=1, keepdim=True)

    def _get_observations(self) -> dict:
        roll, pitch, _ = self._get_payload_rpy()
        board_quat_inverse = self.payload.data.root_quat_w.clone()
        board_quat_inverse[:, 1:4] *= -1.0
        board_ang_vel_local = self._quat_rotate_vector(
            board_quat_inverse, self.payload.data.root_ang_vel_w
        )
        (
            cargo_relative_position,
            cargo_relative_velocity,
            cargo_relative_roll,
            cargo_relative_pitch,
            cargo_relative_angular_speed,
        ) = self._cargo_relative_state()
        neutral = float(self.cfg.lift_neutral_height)

        obs = torch.cat(
            (
                roll.unsqueeze(1),
                pitch.unsqueeze(1),
                board_ang_vel_local[:, 0:2],
                self.payload.data.root_lin_vel_w[:, 2:3],
                self.lift_height - neutral,
                self.lift_velocity,
                self.base_leveling_target_height - neutral,
                self.leveling_feedback_height,
                self._support_top_relative_heights(),
                self.agv_terrain_roll,
                self.agv_terrain_pitch,
                cargo_relative_position[:, 0:2],
                cargo_relative_velocity[:, 0:2],
                cargo_relative_roll.unsqueeze(1),
                cargo_relative_pitch.unsqueeze(1),
                cargo_relative_angular_speed.unsqueeze(1),
            ),
            dim=1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Stability reward and termination
    # ------------------------------------------------------------------
    def _failure_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()
        move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)
        support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
        contact_count = self._compute_support_contact_flags(support_targets).float().sum(dim=1)
        roll, pitch, _ = self._get_payload_rpy()
        board_dropped = self.payload.data.root_pos_w[:, 2] < float(self.cfg.payload_min_z)
        board_tipped = torch.maximum(torch.abs(roll), torch.abs(pitch)) > float(
            self.cfg.tip_roll_pitch_threshold
        )
        support_lost = (
            contact_count < float(self.cfg.critical_support_contacts)
        ) & (self.episode_length_buf > int(self.cfg.support_loss_grace_steps))
        out_of_bounds = self._compute_out_of_bounds(payload_xy)

        cargo_relative_position, _, cargo_roll, cargo_pitch, _ = self._cargo_relative_state()
        board_half_xy = 0.5 * torch.tensor(
            self.cfg.payload_size[:2], device=self.device, dtype=cargo_relative_position.dtype
        )
        cargo_center_over_board = torch.all(
            torch.abs(cargo_relative_position[:, :2]) <= board_half_xy, dim=1
        )
        cargo_dropped = (~cargo_center_over_board) | (cargo_relative_position[:, 2] < 0.0)
        cargo_tipped = torch.maximum(torch.abs(cargo_roll), torch.abs(cargo_pitch)) > float(
            self.cfg.cargo_tip_threshold
        )
        return (
            board_dropped,
            board_tipped,
            support_lost,
            out_of_bounds,
            cargo_dropped,
            cargo_tipped,
            contact_count,
        )

    def _get_rewards(self) -> torch.Tensor:
        roll, pitch, _ = self._get_payload_rpy()
        board_quat_inverse = self.payload.data.root_quat_w.clone()
        board_quat_inverse[:, 1:4] *= -1.0
        board_ang_vel_local = self._quat_rotate_vector(
            board_quat_inverse, self.payload.data.root_ang_vel_w
        )
        (
            cargo_relative_position,
            cargo_relative_velocity,
            cargo_relative_roll,
            cargo_relative_pitch,
            cargo_relative_angular_speed,
        ) = self._cargo_relative_state()

        board_angle_cost = torch.square(roll) + torch.square(pitch)
        board_angular_velocity_cost = torch.sum(
            torch.square(board_ang_vel_local[:, 0:2]), dim=1
        )
        board_vertical_velocity_cost = torch.square(
            self.payload.data.root_lin_vel_w[:, 2]
        )
        cargo_slip = cargo_relative_position[:, 0:2] - self.cargo_initial_relative_xy
        cargo_slip_cost = torch.sum(torch.square(cargo_slip), dim=1)
        cargo_velocity_cost = torch.sum(
            torch.square(cargo_relative_velocity[:, 0:2]), dim=1
        )
        cargo_tilt_cost = torch.square(cargo_relative_roll) + torch.square(
            cargo_relative_pitch
        )
        cargo_angular_velocity_cost = torch.square(cargo_relative_angular_speed)
        action_cost = torch.mean(torch.square(self.actions), dim=1)
        action_rate_cost = torch.mean(torch.square(self.actions - self.prev_actions), dim=1)
        lift_velocity_cost = torch.mean(torch.square(self.lift_velocity), dim=1)

        (
            board_dropped,
            board_tipped,
            support_lost,
            out_of_bounds,
            cargo_dropped,
            cargo_tipped,
            contact_count,
        ) = self._failure_state()
        failure = (
            board_dropped
            | board_tipped
            | support_lost
            | out_of_bounds
            | cargo_dropped
            | cargo_tipped
        )

        reward = (
            float(self.cfg.residual_alive_reward)
            - float(self.cfg.residual_board_angle_penalty_scale) * board_angle_cost
            - float(self.cfg.residual_board_angular_velocity_penalty_scale)
            * board_angular_velocity_cost
            - float(self.cfg.residual_board_vertical_velocity_penalty_scale)
            * board_vertical_velocity_cost
            - float(self.cfg.residual_cargo_slip_penalty_scale) * cargo_slip_cost
            - float(self.cfg.residual_cargo_velocity_penalty_scale) * cargo_velocity_cost
            - float(self.cfg.residual_cargo_tilt_penalty_scale) * cargo_tilt_cost
            - float(self.cfg.residual_cargo_angular_velocity_penalty_scale)
            * cargo_angular_velocity_cost
            - float(self.cfg.residual_action_penalty_scale) * action_cost
            - float(self.cfg.residual_action_rate_penalty_scale) * action_rate_cost
            - float(self.cfg.residual_lift_velocity_penalty_scale) * lift_velocity_cost
            - float(self.cfg.residual_failure_penalty) * failure.float()
        )

        self.extras["log"] = {
            "Residual/reward_mean": reward.mean().detach(),
            "Residual/action_rms": torch.sqrt(action_cost.mean()).detach(),
            "Residual/height_rms_mm": (
                1000.0 * torch.sqrt(torch.mean(torch.square(self.last_residual_height)))
            ).detach(),
            "Residual/saturation_rate": self.last_residual_saturated.float().mean().detach(),
            "Board/roll_abs_mean": torch.abs(roll).mean().detach(),
            "Board/pitch_abs_mean": torch.abs(pitch).mean().detach(),
            "Board/rp_angular_speed_mean": torch.sqrt(
                board_angular_velocity_cost
            ).mean().detach(),
            "Board/vertical_speed_abs_mean": torch.abs(
                self.payload.data.root_lin_vel_w[:, 2]
            ).mean().detach(),
            "Cargo/slip_from_reset_mean": torch.sqrt(cargo_slip_cost).mean().detach(),
            "Cargo/relative_speed_mean": torch.sqrt(cargo_velocity_cost).mean().detach(),
            "Cargo/relative_angular_speed_mean": cargo_relative_angular_speed.mean().detach(),
            "Support/contact_count_mean": contact_count.mean().detach(),
            "Done/board_drop_rate": board_dropped.float().mean().detach(),
            "Done/board_tip_rate": board_tipped.float().mean().detach(),
            "Done/support_lost_rate": support_lost.float().mean().detach(),
            "Done/cargo_drop_rate": cargo_dropped.float().mean().detach(),
            "Done/cargo_tip_rate": cargo_tipped.float().mean().detach(),
            "Randomization/speed_mean": self.randomized_target_speed.mean().detach(),
            "Randomization/terrain_amplitude_mean": (
                self.randomized_terrain_amplitude.mean().detach()
            ),
            "Randomization/cargo_mass_mean": self.randomized_cargo_mass.mean().detach(),
        }

        self.last_payload_dropped[:] = board_dropped.detach()
        self.last_payload_tipped[:] = board_tipped.detach()
        self.last_support_lost[:] = support_lost.detach()
        self.last_out_of_bounds[:] = out_of_bounds.detach()
        self.last_contact_count[:] = contact_count.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        failure_parts = self._failure_state()[:6]
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for failure in failure_parts:
            terminated |= failure
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    def _uniform(self, bounds: tuple[float, float], count: int) -> torch.Tensor:
        low, high = (float(value) for value in bounds)
        return low + (high - low) * torch.rand(count, device=self.device)

    def _randomize(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        if bool(self.cfg.residual_domain_randomization):
            self.randomized_target_speed[env_ids] = self._uniform(
                self.cfg.residual_speed_range, count
            )
            self.randomized_terrain_amplitude[env_ids] = self._uniform(
                self.cfg.residual_terrain_amplitude_range, count
            )
            self.randomized_terrain_phase_x[env_ids] = self._uniform(
                self.cfg.residual_terrain_phase_x_range, count
            )
            self.randomized_terrain_phase_y[env_ids] = self._uniform(
                self.cfg.residual_terrain_phase_y_range, count
            )
            self.randomized_cargo_mass[env_ids] = self._uniform(
                self.cfg.residual_cargo_mass_range, count
            )
            self.randomized_cargo_offset_xy[env_ids, 0] = self._uniform(
                self.cfg.residual_cargo_offset_x_range, count
            )
            self.randomized_cargo_offset_xy[env_ids, 1] = self._uniform(
                self.cfg.residual_cargo_offset_y_range, count
            )
        else:
            self.randomized_target_speed[env_ids] = float(self.cfg.residual_scripted_speed)
            self.randomized_terrain_amplitude[env_ids] = float(self.cfg.bump_amplitude)
            self.randomized_terrain_phase_x[env_ids] = float(self.cfg.bump_phase_x)
            self.randomized_terrain_phase_y[env_ids] = float(self.cfg.bump_phase_y)
            self.randomized_cargo_mass[env_ids] = float(self.cfg.cargo_mass)
            self.randomized_cargo_offset_xy[env_ids] = 0.0

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.payload._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._randomize(env_ids)
        super()._reset_idx(env_ids)

        self.base_leveling_target_height[env_ids] = float(self.cfg.lift_neutral_height)
        self.last_residual_height[env_ids] = 0.0
        self.last_residual_saturated[env_ids] = False

        self.cargo_initial_relative_xy[env_ids] = self.randomized_cargo_offset_xy[env_ids]
        if self.cfg.residual_domain_randomization:
            # Reposition the free Cargo without attaching it to the Board. The
            # disabled path intentionally performs no extra PhysX writes so a
            # zero residual remains trajectory-equivalent to the V7.5 task.
            count = len(env_ids)
            cargo_init = torch.tensor(
                self.cfg.cargo_init_pos, device=self.device, dtype=torch.float32
            )
            cargo_pose = torch.zeros((count, 7), device=self.device)
            cargo_pose[:, :3] = self.scene.env_origins[env_ids] + cargo_init
            cargo_pose[:, :2] += self.randomized_cargo_offset_xy[env_ids]
            cargo_pose[:, 3] = 1.0
            cargo_velocity = torch.zeros((count, 6), device=self.device)
            self.cargo.write_root_pose_to_sim(cargo_pose, env_ids=env_ids)
            self.cargo.write_root_velocity_to_sim(cargo_velocity, env_ids=env_ids)

            # Scale inertia with mass so randomization changes the physical
            # load rather than only replacing the scalar mass value. PhysX
            # mass/inertia buffers and their indices are CPU tensors.
            physx_indices = env_ids.to(device="cpu", dtype=torch.int32)
            reference_indices = physx_indices.to(dtype=torch.long)
            masses = (
                self.randomized_cargo_mass[env_ids]
                .detach()
                .to(device="cpu")
                .unsqueeze(1)
            )
            mass_scale = masses / self._cargo_reference_mass
            inertias = self._cargo_reference_inertias[reference_indices] * mass_scale
            self.cargo.root_physx_view.set_masses(masses, physx_indices)
            self.cargo.root_physx_view.set_inertias(inertias, physx_indices)
