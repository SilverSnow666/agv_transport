from __future__ import annotations

from collections.abc import Sequence
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from pxr import Usd, UsdGeom

from .agv_carry_env_cfg import AgvCarryEnvCfg


class AgvCarryEnv(DirectRLEnv):
    """V6.0 三 AGV 无刚性连接协同驮运环境。

    与 V5.x 推送环境不同，本环境把 payload 放在三台 AGV 顶部，依靠接触、重力和摩擦
    进行承载运输。第一版只做平地最小闭环：
    - 三台 AGV 保持三角支撑队形；
    - payload 到达目标；
    - payload roll/pitch、垂向速度、滑移和支撑丢失受到惩罚。

    该环境不依赖真实接触传感器，训练 reward 中的 contact_flags 是解析近似量：
    AGV 位于对应支撑目标附近，且 AGV 顶面接近 payload 底面。
    """

    cfg: AgvCarryEnvCfg

    def __init__(self, cfg: AgvCarryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.agv_yaw = torch.zeros((self.num_envs, 3), device=self.device)
        self.agv_planar_vel = torch.zeros((self.num_envs, 3, 2), device=self.device)

        target_xy = self._get_target_xy()
        payload_xy = self.payload.data.root_pos_w[:, :2]
        self.prev_goal_dist = torch.linalg.norm(target_xy - payload_xy, dim=1)

        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_payload_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_payload_tipped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_support_lost = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.last_roll_abs = torch.zeros(self.num_envs, device=self.device)
        self.last_pitch_abs = torch.zeros(self.num_envs, device=self.device)
        self.last_slip_error = torch.zeros(self.num_envs, device=self.device)
        self.last_contact_count = torch.zeros(self.num_envs, device=self.device)
        self.last_support_margin = torch.zeros(self.num_envs, device=self.device)

    # ---------------------------------------------------------------------
    # Scene
    # ---------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.agv1 = RigidObject(self.cfg.agv1_cfg)
        self.agv2 = RigidObject(self.cfg.agv2_cfg)
        self.agv3 = RigidObject(self.cfg.agv3_cfg)
        self.payload = RigidObject(self.cfg.payload_cfg)
        self.agvs = [self.agv1, self.agv2, self.agv3]

        self.scene.rigid_objects["agv1"] = self.agv1
        self.scene.rigid_objects["agv2"] = self.agv2
        self.scene.rigid_objects["agv3"] = self.agv3
        self.scene.rigid_objects["payload"] = self.payload

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 目标点 marker。
        target_marker_cfg = sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.025),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.0),
        )
        target_marker_cfg.func(
            "/World/envs/env_0/CarryTargetMarker",
            target_marker_cfg,
            translation=(self.cfg.target_pos[0], self.cfg.target_pos[1], 0.02),
        )

        # AGV 外观模型只是视觉，不参与物理承载；承载由 Cuboid 刚体完成。
        for agv_name in ["AGV1", "AGV2", "AGV3"]:
            self.cfg.agv_visual_cfg.func(
                f"/World/envs/env_0/{agv_name}/Visual",
                self.cfg.agv_visual_cfg,
                translation=(0.05, 0.0, -0.02),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        self.scene.clone_environments(copy_from_source=False)
        self._hide_agv_collision_visual()

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _hide_agv_collision_visual(self) -> None:
        """隐藏 AGV 简化支撑方块视觉，只保留 USD 小车外观。

        如果你想检查承载接触面，可以临时注释掉这个函数调用。
        """
        stage = sim_utils.get_current_stage()
        for env_id in range(self.num_envs):
            for agv_name in ["AGV1", "AGV2", "AGV3"]:
                agv_root_path = f"/World/envs/env_{env_id}/{agv_name}"
                agv_root = stage.GetPrimAtPath(agv_root_path)
                if not agv_root.IsValid():
                    continue
                for prim in Usd.PrimRange(agv_root):
                    prim_path = prim.GetPath().pathString
                    if "/Visual" in prim_path or prim_path == agv_root_path:
                        continue
                    imageable = UsdGeom.Imageable(prim)
                    if imageable:
                        imageable.MakeInvisible()

    # ---------------------------------------------------------------------
    # Action
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """三台差速 AGV 的 kinematic 平面运动。

        V6.0 默认 z 不变；若 cfg.enable_bumpy_support=True，则 AGV 支撑面高度随位置变化，
        可用于下一阶段模拟低幅坑洼激励，但这不等同于真实轮地接触。
        """
        dt = self.cfg.sim.dt * self.cfg.decimation
        env_xy = self.scene.env_origins[:, :2]

        for i, agv in enumerate(self.agvs):
            agv_state = agv.data.root_state_w.clone()
            linear_speed = self.actions[:, 2 * i] * self.cfg.max_agv_linear_speed
            angular_speed = self.actions[:, 2 * i + 1] * self.cfg.max_agv_angular_speed

            self.agv_yaw[:, i] = self._wrap_to_pi(self.agv_yaw[:, i] + angular_speed * dt)
            heading = torch.stack((torch.cos(self.agv_yaw[:, i]), torch.sin(self.agv_yaw[:, i])), dim=1)
            new_xy = agv_state[:, :2] + linear_speed.unsqueeze(-1) * heading * dt

            rel_xy = torch.clamp(new_xy - env_xy, -self.cfg.workspace_limit, self.cfg.workspace_limit)
            new_xy = env_xy + rel_xy
            local_xy = new_xy - env_xy
            z = self.cfg.agv_center_z + self._terrain_height(local_xy)

            quat = self._yaw_to_quat(self.agv_yaw[:, i])
            agv_state[:, 0:2] = new_xy
            agv_state[:, 2] = z
            agv_state[:, 3:7] = quat
            agv_state[:, 7] = linear_speed * heading[:, 0]
            agv_state[:, 8] = linear_speed * heading[:, 1]
            self.agv_planar_vel[:, i, 0] = agv_state[:, 7]
            self.agv_planar_vel[:, i, 1] = agv_state[:, 8]
            agv_state[:, 9] = 0.0
            agv_state[:, 10] = 0.0
            agv_state[:, 11] = 0.0
            agv_state[:, 12] = angular_speed

            agv.write_root_pose_to_sim(agv_state[:, :7])
            agv.write_root_velocity_to_sim(agv_state[:, 7:])

        if bool(getattr(self.cfg, "enable_virtual_friction_carry", False)):
            self._apply_virtual_friction_carry(dt)

    def _apply_virtual_friction_carry(self, dt: float) -> None:
        """用虚拟摩擦耦合修正 kinematic 支撑台无法可靠带动 payload 的问题。

        该项不是刚性连接，也不直接改 payload 位置；它只在解析判断仍有至少两台 AGV
        支撑 payload 时，把 payload 的平面速度软耦合到支撑平台平均速度，并根据
        payload 相对支撑三角形的滑移误差增加一个小的恢复速度。
        """
        payload_state = self.payload.data.root_state_w.clone()
        payload_xy = payload_state[:, :2]
        target_xy = self._get_target_xy()
        move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)
        support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
        contact_flags = self._compute_support_contact_flags(support_targets)
        contact_float = contact_flags.float()
        contact_count = contact_float.sum(dim=1)
        valid = contact_count >= float(self.cfg.virtual_friction_min_contacts)
        if not torch.any(valid):
            return

        weights = contact_float.unsqueeze(-1)
        denom = contact_count.clamp_min(1.0).unsqueeze(-1)
        support_vel = (self.agv_planar_vel * weights).sum(dim=1) / denom

        agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in self.agvs], dim=1)
        support_centroid = (agv_xy * weights).sum(dim=1) / denom
        offsets = torch.tensor(self.cfg.support_offsets_xy, device=self.device, dtype=torch.float32)
        mean_forward = (offsets[None, :, 0] * contact_float).sum(dim=1) / contact_count.clamp_min(1.0)
        mean_lateral = (offsets[None, :, 1] * contact_float).sum(dim=1) / contact_count.clamp_min(1.0)
        expected_payload_xy = support_centroid - mean_forward.unsqueeze(-1) * move_dir - mean_lateral.unsqueeze(-1) * lateral_dir
        slip_vec = expected_payload_xy - payload_xy

        desired_vxy = support_vel + float(self.cfg.slip_correction_gain) * slip_vec
        speed = torch.linalg.norm(desired_vxy, dim=1, keepdim=True)
        max_speed = float(self.cfg.max_payload_planar_speed)
        desired_vxy = desired_vxy * torch.clamp(max_speed / speed.clamp_min(1e-6), max=1.0)

        alpha = float(self.cfg.virtual_friction_coupling)
        current_vxy = payload_state[:, 7:9]
        new_vxy = (1.0 - alpha) * current_vxy + alpha * desired_vxy
        payload_state[:, 7:9] = torch.where(valid.unsqueeze(-1), new_vxy, current_vxy)

        # 在支撑有效时增加垂向、roll/pitch 角速度阻尼，避免最小闭环阶段货物持续抖动。
        vz = payload_state[:, 9]
        roll_w = payload_state[:, 10]
        pitch_w = payload_state[:, 11]
        payload_state[:, 9] = torch.where(valid, vz * (1.0 - float(self.cfg.payload_vertical_damping)), vz)
        payload_state[:, 10] = torch.where(valid, roll_w * (1.0 - float(self.cfg.payload_roll_pitch_damping)), roll_w)
        payload_state[:, 11] = torch.where(valid, pitch_w * (1.0 - float(self.cfg.payload_roll_pitch_damping)), pitch_w)

        # Fix4：增加 payload yaw 软阻尼/对齐。
        # Fix3 只耦合平动速度，三车速度差、接触摩擦和支撑点不完全对称会给 payload
        # 产生绕 z 轴的净力矩，表现为“能被驮着走，但货物一直自转”。
        # 这里不直接改 pose，只写入角速度：在支撑有效时抑制持续 yaw spinning，
        # 并让 payload 航向缓慢对齐当前 payload->target 方向。
        _, _, payload_yaw = self._get_payload_rpy()
        target_yaw = torch.atan2(move_dir[:, 1], move_dir[:, 0])
        yaw_error = self._wrap_to_pi(target_yaw - payload_yaw)
        desired_wz = float(self.cfg.payload_yaw_alignment_gain) * yaw_error
        desired_wz = torch.clamp(
            desired_wz,
            min=-float(self.cfg.max_payload_yaw_rate),
            max=float(self.cfg.max_payload_yaw_rate),
        )
        current_wz = payload_state[:, 12]
        damped_wz = current_wz * (1.0 - float(self.cfg.payload_yaw_damping))
        yaw_alpha = float(self.cfg.payload_yaw_alignment_coupling)
        new_wz = (1.0 - yaw_alpha) * damped_wz + yaw_alpha * desired_wz
        payload_state[:, 12] = torch.where(valid, new_wz, current_wz)

        self.payload.write_root_velocity_to_sim(payload_state[:, 7:])

    # ---------------------------------------------------------------------
    # Observation
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        env_xy = self.scene.env_origins[:, :2]
        payload_xy = self.payload.data.root_pos_w[:, :2]
        payload_z = self.payload.data.root_pos_w[:, 2:3]
        target_xy = self._get_target_xy()
        payload_to_target = target_xy - payload_xy
        target_dist = torch.linalg.norm(payload_to_target, dim=1, keepdim=True).clamp_min(1e-6)
        move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)

        roll, pitch, yaw = self._get_payload_rpy()
        payload_rpy = torch.stack((roll, pitch, yaw), dim=1)
        payload_lin_vel = self.payload.data.root_lin_vel_w[:, :3]
        payload_ang_vel = self.payload.data.root_ang_vel_w[:, :3]

        support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
        formation_errors = self._compute_formation_errors(support_targets)
        contact_flags = self._compute_support_contact_flags(support_targets)
        slip_error = self._compute_slip_error(payload_xy, move_dir, lateral_dir).unsqueeze(-1)
        support_margin = self._compute_support_polygon_margin(payload_xy).unsqueeze(-1)

        obs_parts = [
            payload_xy - env_xy,                 # 2
            payload_z,                           # 1
            target_xy - env_xy,                  # 2
            payload_to_target / 3.5,             # 2
            target_dist / 3.5,                   # 1
            move_dir,                            # 2
            lateral_dir,                         # 2
            payload_rpy,                         # 3
            payload_lin_vel,                     # 3
            payload_ang_vel,                     # 3
            slip_error,                          # 1
            support_margin,                      # 1
            contact_flags.float(),               # 3
        ]

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            agv_z = agv.data.root_pos_w[:, 2:3]
            agv_vel_xy = agv.data.root_lin_vel_w[:, :2]
            agv_heading = torch.stack((torch.cos(self.agv_yaw[:, i]), torch.sin(self.agv_yaw[:, i])), dim=1)
            agv_to_support = support_targets[:, i, :] - agv_xy
            obs_parts.extend([
                agv_xy - env_xy,                 # 2
                agv_z,                           # 1
                agv_heading,                     # 2
                agv_vel_xy,                      # 2
                agv_to_support,                  # 2
                formation_errors[:, i:i + 1],    # 1
            ])

        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs}

    # ---------------------------------------------------------------------
    # Reward / Done / Reset
    # ---------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()
        goal_dist = torch.linalg.norm(target_xy - payload_xy, dim=1)
        progress = self.prev_goal_dist - goal_dist
        self.prev_goal_dist[:] = goal_dist.detach()

        move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)
        support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
        formation_errors = self._compute_formation_errors(support_targets)
        contact_flags = self._compute_support_contact_flags(support_targets)
        contact_count = contact_flags.float().sum(dim=1)
        slip_error = self._compute_slip_error(payload_xy, move_dir, lateral_dir)
        support_margin = self._compute_support_polygon_margin(payload_xy)
        roll, pitch, yaw = self._get_payload_rpy()
        target_yaw = torch.atan2(move_dir[:, 1], move_dir[:, 0])
        yaw_error = self._wrap_to_pi(target_yaw - yaw)

        roll_pitch_cost = torch.square(roll) + torch.square(pitch)
        yaw_cost = torch.square(yaw_error)
        payload_ang_vel = self.payload.data.root_ang_vel_w[:, :3]
        angular_cost = torch.sum(torch.square(payload_ang_vel[:, :2]), dim=1)
        yaw_rate_cost = torch.square(payload_ang_vel[:, 2])
        vertical_vel_cost = torch.square(self.payload.data.root_lin_vel_w[:, 2])

        formation_cost = torch.mean(torch.square(formation_errors), dim=1)
        slip_violation = torch.clamp(slip_error - self.cfg.slip_penalty_threshold, min=0.0)
        support_margin_violation = torch.clamp(self.cfg.support_polygon_min_margin - support_margin, min=0.0)
        support_loss = torch.clamp(2.0 - contact_count, min=0.0)

        dropped = self.payload.data.root_pos_w[:, 2] < self.cfg.payload_min_z
        tipped = torch.maximum(torch.abs(roll), torch.abs(pitch)) > self.cfg.tip_roll_pitch_threshold
        out_of_bounds = self._compute_out_of_bounds(payload_xy)
        success = self._compute_success(goal_dist, roll, pitch, yaw_error, slip_error, contact_count, support_margin)

        reward = (
            self.cfg.progress_reward_scale * progress
            - self.cfg.distance_penalty_scale * goal_dist
            - self.cfg.roll_pitch_penalty_scale * roll_pitch_cost
            - self.cfg.yaw_alignment_penalty_scale * yaw_cost
            - self.cfg.angular_velocity_penalty_scale * angular_cost
            - self.cfg.yaw_rate_penalty_scale * yaw_rate_cost
            - self.cfg.vertical_velocity_penalty_scale * vertical_vel_cost
            + self.cfg.support_contact_reward_scale * (contact_count / 3.0)
            - self.cfg.support_loss_penalty_scale * support_loss
            - self.cfg.formation_error_penalty_scale * formation_cost
            - self.cfg.slip_penalty_scale * torch.square(slip_violation)
            - self.cfg.support_polygon_penalty_scale * torch.square(support_margin_violation)
            - self.cfg.action_penalty_scale * torch.mean(torch.square(self.actions), dim=1)
            - self.cfg.action_rate_penalty_scale * torch.mean(torch.square(self.actions - self.prev_actions), dim=1)
            + self.cfg.success_reward_scale * success.float()
            - self.cfg.drop_penalty * dropped.float()
            - self.cfg.tip_penalty * tipped.float()
            - self.cfg.out_of_bounds_penalty * out_of_bounds.float()
        )

        self.last_success[:] = success.detach()
        self.last_payload_dropped[:] = dropped.detach()
        self.last_payload_tipped[:] = tipped.detach()
        self.last_support_lost[:] = (support_loss > 0.5).detach()
        self.last_out_of_bounds[:] = out_of_bounds.detach()
        self.last_goal_dist[:] = goal_dist.detach()
        self.last_roll_abs[:] = torch.abs(roll).detach()
        self.last_pitch_abs[:] = torch.abs(pitch).detach()
        self.last_slip_error[:] = slip_error.detach()
        self.last_contact_count[:] = contact_count.detach()
        self.last_support_margin[:] = support_margin.detach()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()
        goal_dist = torch.linalg.norm(target_xy - payload_xy, dim=1)
        move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)
        support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
        contact_count = self._compute_support_contact_flags(support_targets).float().sum(dim=1)
        slip_error = self._compute_slip_error(payload_xy, move_dir, lateral_dir)
        support_margin = self._compute_support_polygon_margin(payload_xy)
        roll, pitch, yaw = self._get_payload_rpy()
        target_yaw = torch.atan2(move_dir[:, 1], move_dir[:, 0])
        yaw_error = self._wrap_to_pi(target_yaw - yaw)

        success = self._compute_success(goal_dist, roll, pitch, yaw_error, slip_error, contact_count, support_margin)
        dropped = self.payload.data.root_pos_w[:, 2] < self.cfg.payload_min_z
        tipped = torch.maximum(torch.abs(roll), torch.abs(pitch)) > self.cfg.tip_roll_pitch_threshold
        out_of_bounds = self._compute_out_of_bounds(payload_xy)
        support_lost = (contact_count < 2.0) & (self.episode_length_buf > self.cfg.support_loss_grace_steps)

        terminated = success | dropped | tipped | out_of_bounds | support_lost
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.payload._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.agv_yaw[env_ids] = 0.0
        self.agv_planar_vel[env_ids] = 0.0

        # Payload 初始状态。
        payload_init_pos = torch.tensor(self.cfg.payload_init_pos, device=self.device, dtype=torch.float32)
        payload_pose = torch.zeros((num_reset, 7), device=self.device)
        payload_pose[:, :3] = env_origins + payload_init_pos
        payload_pose[:, 3] = 1.0
        payload_vel = torch.zeros((num_reset, 6), device=self.device)
        self.payload.write_root_pose_to_sim(payload_pose, env_ids=env_ids)
        self.payload.write_root_velocity_to_sim(payload_vel, env_ids=env_ids)

        # 根据 payload->target 方向初始化三台 AGV 的支撑三角形与车头朝向。
        # 这样即使后续把 target 改到别的方向，也不会出现开局三车顶着货物原地大角度旋转。
        target_xy = self._get_target_xy()[env_ids]
        payload_xy = payload_pose[:, :2]
        direction = target_xy - payload_xy
        direction_norm = torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
        move_dir = direction / direction_norm
        lateral_dir = torch.stack((-move_dir[:, 1], move_dir[:, 0]), dim=1)
        init_yaw = torch.atan2(move_dir[:, 1], move_dir[:, 0])
        init_quat = self._yaw_to_quat(init_yaw)
        self.agv_yaw[env_ids] = init_yaw.unsqueeze(1).repeat(1, 3)

        offsets = torch.tensor(self.cfg.support_offsets_xy, device=self.device, dtype=torch.float32)
        for i, agv in enumerate(self.agvs):
            pose = torch.zeros((num_reset, 7), device=self.device)
            support_xy = payload_xy + offsets[i, 0] * move_dir + offsets[i, 1] * lateral_dir
            pose[:, 0:2] = support_xy
            pose[:, 2] = self.cfg.agv_center_z + self._terrain_height(support_xy - env_origins[:, :2])
            pose[:, 3:7] = init_quat
            vel = torch.zeros((num_reset, 6), device=self.device)
            agv.write_root_pose_to_sim(pose, env_ids=env_ids)
            agv.write_root_velocity_to_sim(vel, env_ids=env_ids)

        self.prev_goal_dist[env_ids] = torch.linalg.norm(target_xy - payload_xy, dim=1)

        self.last_success[env_ids] = False
        self.last_payload_dropped[env_ids] = False
        self.last_payload_tipped[env_ids] = False
        self.last_support_lost[env_ids] = False
        self.last_out_of_bounds[env_ids] = False
        self.last_goal_dist[env_ids] = self.prev_goal_dist[env_ids]
        self.last_roll_abs[env_ids] = 0.0
        self.last_pitch_abs[env_ids] = 0.0
        self.last_slip_error[env_ids] = 0.0
        self.last_contact_count[env_ids] = 3.0
        self.last_support_margin[env_ids] = 0.0

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    def _get_target_xy(self) -> torch.Tensor:
        target_local = torch.tensor(self.cfg.target_pos[:2], device=self.device, dtype=torch.float32)
        return self.scene.env_origins[:, :2] + target_local

    def _compute_move_frame(self, payload_xy: torch.Tensor, target_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direction = target_xy - payload_xy
        norm = torch.linalg.norm(direction, dim=1, keepdim=True)
        fallback = torch.zeros_like(direction)
        fallback[:, 0] = 1.0
        move_dir = torch.where(norm > 1e-5, direction / norm.clamp_min(1e-6), fallback)
        lateral_dir = torch.stack((-move_dir[:, 1], move_dir[:, 0]), dim=1)
        return move_dir, lateral_dir

    def _compute_support_targets(
        self,
        payload_xy: torch.Tensor,
        move_dir: torch.Tensor,
        lateral_dir: torch.Tensor,
    ) -> torch.Tensor:
        offsets = torch.tensor(self.cfg.support_offsets_xy, device=self.device, dtype=torch.float32)
        targets = []
        for i in range(3):
            forward_offset = offsets[i, 0]
            lateral_offset = offsets[i, 1]
            targets.append(payload_xy + forward_offset * move_dir + lateral_offset * lateral_dir)
        return torch.stack(targets, dim=1)

    def _compute_formation_errors(self, support_targets: torch.Tensor) -> torch.Tensor:
        agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in self.agvs], dim=1)
        return torch.linalg.norm(agv_xy - support_targets, dim=2)

    def _compute_support_contact_flags(self, support_targets: torch.Tensor) -> torch.Tensor:
        formation_errors = self._compute_formation_errors(support_targets)
        xy_ok = formation_errors < self.cfg.support_contact_xy_margin
        payload_bottom_z = self.payload.data.root_pos_w[:, 2] - 0.5 * self.cfg.payload_size[2]
        z_flags = []
        for agv in self.agvs:
            agv_top_z = agv.data.root_pos_w[:, 2] + 0.5 * self.cfg.agv_size[2]
            z_gap = torch.abs(payload_bottom_z - agv_top_z)
            z_flags.append(z_gap < self.cfg.support_contact_z_margin)
        z_ok = torch.stack(z_flags, dim=1)
        return xy_ok & z_ok

    def _compute_slip_error(
        self,
        payload_xy: torch.Tensor,
        move_dir: torch.Tensor,
        lateral_dir: torch.Tensor,
    ) -> torch.Tensor:
        """payload 相对三车支撑结构中心的滑移量。

        由于支撑三角形本身均值不一定在 payload 中心，这里用 support_offsets 的均值修正。
        """
        agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in self.agvs], dim=1)
        support_centroid = agv_xy.mean(dim=1)
        offsets = torch.tensor(self.cfg.support_offsets_xy, device=self.device, dtype=torch.float32)
        mean_forward = offsets[:, 0].mean()
        mean_lateral = offsets[:, 1].mean()
        expected_payload_xy = support_centroid - mean_forward * move_dir - mean_lateral * lateral_dir
        return torch.linalg.norm(payload_xy - expected_payload_xy, dim=1)

    def _compute_support_polygon_margin(self, payload_xy: torch.Tensor) -> torch.Tensor:
        """返回 payload CoM 投影在三角支撑多边形内的最小 barycentric margin。

        >0 表示在三角形内部；<0 表示已经越出支撑三角形。
        """
        agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in self.agvs], dim=1)
        a = agv_xy[:, 0, :]
        b = agv_xy[:, 1, :]
        c = agv_xy[:, 2, :]
        p = payload_xy
        v0 = b - a
        v1 = c - a
        v2 = p - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den_safe = torch.where(torch.abs(den) < 1e-6, torch.full_like(den, 1e-6), den)
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den_safe
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den_safe
        w = 1.0 - u - v
        return torch.minimum(torch.minimum(u, v), w)

    def _compute_success(
        self,
        goal_dist: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        yaw_error: torch.Tensor,
        slip_error: torch.Tensor,
        contact_count: torch.Tensor,
        support_margin: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (goal_dist < self.cfg.target_radius)
            & (torch.abs(roll) < self.cfg.stable_roll_pitch_radius)
            & (torch.abs(pitch) < self.cfg.stable_roll_pitch_radius)
            & (torch.abs(yaw_error) < self.cfg.stable_yaw_radius)
            & (slip_error < self.cfg.slip_success_threshold)
            & (contact_count >= 2.0)
            & (support_margin > -0.02)
        )

    def _compute_out_of_bounds(self, payload_xy: torch.Tensor) -> torch.Tensor:
        env_xy = self.scene.env_origins[:, :2]
        payload_rel = payload_xy - env_xy
        payload_oob = torch.any(torch.abs(payload_rel) > self.cfg.workspace_limit, dim=1)
        agv_oob_list = []
        for agv in self.agvs:
            agv_rel = agv.data.root_pos_w[:, :2] - env_xy
            agv_oob_list.append(torch.any(torch.abs(agv_rel) > self.cfg.workspace_limit, dim=1))
        return payload_oob | torch.stack(agv_oob_list, dim=1).any(dim=1)

    def _terrain_height(self, local_xy: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.cfg, "enable_bumpy_support", False)):
            return torch.zeros(local_xy.shape[0], device=self.device)
        amp = float(self.cfg.bump_amplitude)
        kx = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_x), 1e-6)
        ky = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_y), 1e-6)
        return amp * torch.sin(kx * local_xy[:, 0]) * torch.sin(ky * local_xy[:, 1])

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), device=yaw.device)
        quat[:, 0] = torch.cos(0.5 * yaw)
        quat[:, 3] = torch.sin(0.5 * yaw)
        return quat

    def _get_payload_rpy(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quat = getattr(self.payload.data, "root_quat_w", None)
        if quat is None:
            quat = self.payload.data.root_state_w[:, 3:7]
        qw = quat[:, 0]
        qx = quat[:, 1]
        qy = quat[:, 2]
        qz = quat[:, 3]

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
