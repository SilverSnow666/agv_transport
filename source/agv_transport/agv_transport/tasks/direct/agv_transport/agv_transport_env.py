from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .agv_transport_env_cfg import AgvTransportEnvCfg
from pxr import Usd, UsdGeom

class AgvTransportEnv(DirectRLEnv):
    """三 AGV 无连接协同推送 payload 的 DirectRLEnv 环境。

    当前版本用于 V4.3-D0A0d-stable-two-pusher：
    - 延续 D0A0c 的 top-2 有效推动 credit；
    - 允许任意两台 AGV 推动 payload 到终点，不强制三车同时接触；
    - 当已有两台 AGV 有效推动时，低贡献 AGV 会受到动作和动作变化率惩罚，减少抽搐；
    - 增加最大路径进度保持与接触保持奖励，避免 payload 被推到半途后断接触或后退。
    """

    cfg: AgvTransportEnvCfg


    def __init__(self, cfg: AgvTransportEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_payload_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_path_progress = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.prev_agv_payload_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_formation_error_mean = torch.zeros(self.num_envs, device=self.device)

        # Episode-level credit for two-pusher behavior.
        # 用于缩放 success reward，避免单车把 payload 推到终点也拿到满分。
        self.episode_positive_progress = torch.zeros(self.num_envs, device=self.device)
        self.episode_two_pusher_progress = torch.zeros(self.num_envs, device=self.device)

        # Episode 内最大路径进度，用于惩罚 payload 推到半途后明显后退。
        self.max_path_progress = torch.zeros(self.num_envs, device=self.device)
        # V4.0：每个环境当前跟踪的 waypoint 编号
        self.current_waypoint_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.prev_waypoint_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.last_success = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.last_position_success = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.last_yaw_success = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.last_out_of_bounds = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.last_bad_rear_push = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.last_payload_goal_dist = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.last_payload_yaw_abs = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        # AGV 朝向角，单位 rad
        self.agv_yaw = torch.zeros((self.num_envs, 3), device=self.device)

        self._agv_height = self.cfg.agv_init_pos[2]
        self._identity_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(self.num_envs, 1)
        # 初始化一次距离缓存
        target_xy, _, path_progress, _ = self._compute_path_tracking_quantities()

        payload_xy = self.payload.data.root_pos_w[:, :2]
        agv_xy = self.agv.data.root_pos_w[:, :2]

        self.prev_path_progress[:] = path_progress.detach()
        self.max_path_progress[:] = path_progress.detach()
        self.prev_payload_goal_dist[:] = torch.linalg.norm(
            payload_xy - target_xy,
            dim=1,
        )
        self.prev_agv_payload_dist[:] = torch.linalg.norm(
            agv_xy - payload_xy,
            dim=1,
        )

        formation_errors = self._compute_formation_errors()
        self.prev_formation_error_mean[:] = formation_errors.mean(dim=1).detach()

    def _hide_agv_collision_visual(self) -> None:
        """隐藏三台简化 AGV 碰撞体的视觉显示，只保留小车 USD 外观。"""

        stage = sim_utils.get_current_stage()

        for env_id in range(self.num_envs):
            for agv_name in ["AGV1", "AGV2", "AGV3"]:
                agv_root_path = f"/World/envs/env_{env_id}/{agv_name}"
                agv_root = stage.GetPrimAtPath(agv_root_path)

                if not agv_root.IsValid():
                    continue

                for prim in Usd.PrimRange(agv_root):
                    prim_path = prim.GetPath().pathString

                    if "/Visual" in prim_path:
                        continue

                    if prim_path == agv_root_path:
                        continue

                    imageable = UsdGeom.Imageable(prim)
                    if imageable:
                        imageable.MakeInvisible()

    def _setup_scene(self):
        # 创建刚体
        self.agv1 = RigidObject(self.cfg.agv1_cfg)
        self.agv2 = RigidObject(self.cfg.agv2_cfg)
        self.agv3 = RigidObject(self.cfg.agv3_cfg)
        self.payload = RigidObject(self.cfg.payload_cfg)

        self.agvs = [self.agv1, self.agv2, self.agv3]
        self.agv = self.agv1
        self.scene.rigid_objects["agv1"] = self.agv1
        self.scene.rigid_objects["agv2"] = self.agv2
        self.scene.rigid_objects["agv3"] = self.agv3
        self.scene.rigid_objects["payload"] = self.payload

        # 地面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
        )

        # 在 env_0 中创建目标点可视化标记，后续会被 clone 到其它环境
        # V4.0：可视化所有 waypoint
        # 中间 waypoint 用黄色，最终 waypoint 用绿色
        for i, waypoint in enumerate(self.cfg.waypoints):
            is_final = i == len(self.cfg.waypoints) - 1

            marker_color = (
                (0.0, 1.0, 0.0) if is_final else (1.0, 1.0, 0.0)
            )

            marker_size = (
                (0.22, 0.22, 0.025) if is_final else (0.16, 0.16, 0.02)
            )

            waypoint_marker_cfg = sim_utils.CuboidCfg(
                size=marker_size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=marker_color,
                    metallic=0.0,
                ),
            )

            waypoint_marker_cfg.func(
                f"/World/envs/env_0/WaypointMarker_{i}",
                waypoint_marker_cfg,
                translation=(waypoint[0], waypoint[1], 0.02),
            )

        # 添加 Isaac Sim 自带小车 / AGV 视觉模型
        # 注意：它只是视觉模型，真正的碰撞和推动仍由 /AGV 这个简化刚体负责
        for agv_name in ["AGV1", "AGV2", "AGV3"]:
            self.cfg.agv_visual_cfg.func(
                f"/World/envs/env_0/{agv_name}/Visual",
                self.cfg.agv_visual_cfg,
                translation=(0.15, 0.0, -0.03),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        # 克隆多环境
        self.scene.clone_environments(copy_from_source=False)

        # 隐藏简化碰撞方块的视觉显示，只保留小车外观
        self._hide_agv_collision_visual()

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 光照
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """三 AGV 差速运动学。

        actions:
            [v1, w1, v2, w2, v3, w3]
        """
        dt = self.cfg.sim.dt * self.cfg.decimation
        env_xy = self.scene.env_origins[:, :2]

        for i, agv in enumerate(self.agvs):
            agv_state = agv.data.root_state_w.clone()

            linear_speed = self.actions[:, 2 * i] * self.cfg.max_agv_linear_speed
            angular_speed = self.actions[:, 2 * i + 1] * self.cfg.max_agv_angular_speed

            self.agv_yaw[:, i] = self.agv_yaw[:, i] + angular_speed * dt
            self.agv_yaw[:, i] = torch.atan2(
                torch.sin(self.agv_yaw[:, i]),
                torch.cos(self.agv_yaw[:, i]),
            )

            heading_x = torch.cos(self.agv_yaw[:, i])
            heading_y = torch.sin(self.agv_yaw[:, i])

            delta_x = linear_speed * heading_x * dt
            delta_y = linear_speed * heading_y * dt

            new_xy = agv_state[:, :2] + torch.stack((delta_x, delta_y), dim=1)

            rel_xy = new_xy - env_xy
            rel_xy = torch.clamp(rel_xy, -self.cfg.workspace_limit, self.cfg.workspace_limit)
            new_xy = env_xy + rel_xy

            quat = self._yaw_to_quat(self.agv_yaw[:, i])

            agv_state[:, :2] = new_xy
            agv_state[:, 2] = self._agv_height
            agv_state[:, 3:7] = quat

            agv_state[:, 7] = linear_speed * heading_x
            agv_state[:, 8] = linear_speed * heading_y
            agv_state[:, 9] = 0.0
            agv_state[:, 10] = 0.0
            agv_state[:, 11] = 0.0
            agv_state[:, 12] = angular_speed

            agv.write_root_pose_to_sim(agv_state[:, :7])
            agv.write_root_velocity_to_sim(agv_state[:, 7:])

    def _get_observations(self) -> dict:
        env_xy = self.scene.env_origins[:, :2]

        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()
        payload_vel_xy = self.payload.data.root_lin_vel_w[:, :2]

        payload_xy_rel = payload_xy - env_xy
        target_xy_rel = target_xy - env_xy
        payload_to_target_xy = target_xy - payload_xy

        # 原 32 维观测的前 8 维
        obs_parts = [
            payload_xy_rel,
            target_xy_rel,
            payload_to_target_xy,
            payload_vel_xy,
        ]

        # 原 32 维观测中，每台 AGV 8 维
        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            agv_vel_xy = agv.data.root_lin_vel_w[:, :2]

            agv_xy_rel = agv_xy - env_xy
            agv_to_payload_xy = payload_xy - agv_xy

            agv_heading_xy = torch.stack(
                (
                    torch.cos(self.agv_yaw[:, i]),
                    torch.sin(self.agv_yaw[:, i]),
                ),
                dim=1,
            )

            obs_parts.extend(
                [
                    agv_xy_rel,
                    agv_to_payload_xy,
                    agv_heading_xy,
                    agv_vel_xy,
                ]
            )

        # 追加 6 维 PPO 辅助观测：
        # payload_heading 2 维
        # payload_yaw_rate 1 维
        # contact_flags 3 维
        payload_yaw = self._get_payload_yaw()
        payload_heading = torch.stack(
            (
                torch.cos(payload_yaw),
                torch.sin(payload_yaw),
            ),
            dim=1,
        )

        payload_yaw_rate = self._get_payload_yaw_rate().unsqueeze(-1)

        contact_flags = self._compute_contact_flags().float()

        # 三台 AGV 到各自目标队形点的向量，每台 2 维，共 6 维
        desired_positions = self._compute_formation_target_xy()

        formation_target_vecs = []

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            desired_xy = desired_positions[:, i, :]
            agv_to_formation_xy = desired_xy - agv_xy
            formation_target_vecs.append(agv_to_formation_xy)

        formation_target_vecs = torch.cat(formation_target_vecs, dim=1)

        obs_parts.extend(
            [
                payload_heading,
                payload_yaw_rate,
                contact_flags,
                formation_target_vecs,
            ]
        )

        obs = torch.cat(obs_parts, dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        payload_xy = self.payload.data.root_pos_w[:, :2]

        target_xy, path_lateral_error, path_progress, _ = (
            self._compute_path_tracking_quantities()
        )

        payload_to_target = target_xy - payload_xy

        payload_goal_dist = torch.linalg.norm(
            payload_to_target,
            dim=1,
        )

        push_dir = payload_to_target / payload_goal_dist.unsqueeze(-1).clamp_min(1e-6)

        # V4.1C：沿路径方向的连续进度奖励
        path_progress_delta = path_progress - self.prev_path_progress
        self.prev_path_progress[:] = path_progress.detach()

        self.prev_payload_goal_dist[:] = payload_goal_dist.detach()

        payload_yaw = self._get_payload_yaw()
        payload_yaw_abs = torch.abs(payload_yaw)
        payload_yaw_rate_abs = torch.abs(self._get_payload_yaw_rate())

        payload_speed = torch.linalg.norm(
            self.payload.data.root_lin_vel_w[:, :2],
            dim=1,
        )

        num_waypoints = len(self.cfg.waypoints)
        final_waypoint_idx = num_waypoints - 1
        is_final_waypoint = self.current_waypoint_idx == final_waypoint_idx

        final_target_xy = self._get_final_target_xy()

        final_goal_dist = torch.linalg.norm(
            payload_xy - final_target_xy,
            dim=1,
        )

        near_goal = (final_goal_dist < 0.45).float()

        contact_flags = self._compute_contact_flags()
        contact_flags_float = contact_flags.float()
        contact_count = contact_flags_float.sum(dim=1)

        (
            heading_to_payload,
            front_dists,
            rear_dists,
            v_actions,
        ) = self._compute_contact_geometry(payload_xy)

        # 车头是否合理朝向 payload。
        # heading_to_payload = 1 表示车头正对 payload；
        # heading_to_payload = -1 表示车尾正对 payload。
        front_facing_score = torch.clamp(
            (
                heading_to_payload - self.cfg.front_contact_heading_min
            )
            / (1.0 - self.cfg.front_contact_heading_min),
            min=0.0,
            max=1.0,
        )

        front_contact_score = contact_flags_float * front_facing_score
        front_contact_count = front_contact_score.sum(dim=1)

        # 如果 rear point 比 front point 更靠近 payload，说明很可能是车尾接触。
        rear_closer_than_front = (
            rear_dists + self.cfg.front_rear_margin < front_dists
        ).float()

        rear_contact_penalty = torch.sum(
            contact_flags_float * rear_closer_than_front,
            dim=1,
        ) / 3.0

        # 接触状态下的负线速度：允许远离 payload 时倒车换位，
        # 但不鼓励贴着 payload 倒车推。
        reverse_contact_penalty = torch.sum(
            contact_flags_float * torch.square(torch.clamp(-v_actions, min=0.0)),
            dim=1,
        ) / 3.0

        # 接触时车头明显没有朝向 payload，也给软惩罚。
        bad_contact_heading_penalty = torch.sum(
            contact_flags_float
            * torch.square(
                torch.clamp(
                    self.cfg.front_contact_heading_min - heading_to_payload,
                    min=0.0,
                )
            ),
            dim=1,
        ) / 3.0

        _, bad_rear_push = self._compute_bad_rear_push(payload_xy)
        self.last_bad_rear_push[:] = bad_rear_push.detach()

        # 沿路径真实前进时的 gate。只对正向进度开启，避免倒退时产生虚假接触奖励。
        positive_progress = torch.clamp(path_progress_delta, min=0.0)
        negative_progress = torch.clamp(-path_progress_delta, min=0.0)
        progress_gate = torch.clamp(
            positive_progress / 0.005,
            min=0.0,
            max=1.0,
        )

        # D0A0d：记录 episode 内最大路径进度，惩罚明显回退或中途丢失进度。
        self.max_path_progress[:] = torch.maximum(
            self.max_path_progress,
            path_progress.detach(),
        )
        progress_drop = torch.clamp(
            self.max_path_progress - path_progress - self.cfg.progress_drop_tolerance,
            min=0.0,
        )

        # D0A0d：有效推动贡献。不是奖励三车都贴上去，而是奖励任意两台 AGV
        # 以前端接触、朝向合理、正向动作推动 payload。
        push_utility = (
            contact_flags_float
            * front_facing_score
            * torch.clamp(v_actions, min=0.0)
        )

        sorted_push_utility, _ = torch.sort(
            push_utility,
            dim=1,
            descending=True,
        )
        top1_push_utility = sorted_push_utility[:, 0]
        second_push_utility = sorted_push_utility[:, 1]
        top2_push_utility = top1_push_utility + second_push_utility

        # 如果只有一台 AGV 在推，second_push 接近 0，two_pusher_gate 接近 0；
        # 第二台 AGV 也有效推时，主要 progress / success reward 才逐步放大。
        two_pusher_gate = torch.clamp(
            second_push_utility / self.cfg.two_pusher_gate_threshold,
            min=0.0,
            max=1.0,
        )

        progress_reward = (
            self.cfg.progress_base_reward_scale * positive_progress
            + self.cfg.progress_two_pusher_bonus_scale * positive_progress * two_pusher_gate
            - self.cfg.backward_progress_penalty_scale * negative_progress
        )
        single_pusher_progress_penalty = positive_progress * (1.0 - two_pusher_gate)

        # Episode-level two-pusher progress ratio，用于 success reward credit assignment。
        two_pusher_progress = positive_progress * two_pusher_gate
        self.episode_positive_progress += positive_progress.detach()
        self.episode_two_pusher_progress += two_pusher_progress.detach()
        two_pusher_progress_ratio = (
            self.episode_two_pusher_progress
            / self.episode_positive_progress.clamp_min(1e-6)
        )
        success_quality_gate = torch.clamp(
            self.cfg.success_base_ratio
            + self.cfg.success_two_pusher_ratio * two_pusher_progress_ratio,
            min=0.0,
            max=1.0,
        )

        # 冷启动辅助：只要已经形成前端接触且正向推，就给一点小奖励；
        # 但第二台有效推动车的 credit 更高，避免 AGV1 单车捷径。
        pre_push_reward = self.cfg.pre_push_reward_scale * top2_push_utility
        contact_reward = pre_push_reward + progress_gate * (
            self.cfg.effective_push_reward_scale * top2_push_utility
            + self.cfg.second_pusher_reward_scale * second_push_utility
            + self.cfg.front_contact_count_reward_scale
            * torch.clamp(front_contact_count, max=2.0)
        )

        # D0A0d：接触保持奖励。即使某一步 payload 进度不明显，只要两台 AGV
        # 仍以前端接触并保持正向推送，就给小的稳定奖励，减少“推一下就断开”。
        contact_persistence_reward = two_pusher_gate * top2_push_utility

        # 任意两台 AGV 靠近有效接触带的接触前引导奖励，不固定 AGV2/AGV3 身份。
        contact_zone_errors = self._compute_contact_zone_errors(payload_xy)
        sorted_zone_errors, _ = torch.sort(contact_zone_errors, dim=1)
        best_two_zone_error = sorted_zone_errors[:, 0] + sorted_zone_errors[:, 1]
        contact_zone_approach_reward = torch.clamp(
            1.0 - best_two_zone_error / self.cfg.contact_zone_error_norm,
            min=0.0,
            max=1.0,
        )

        agv_activity = torch.stack(
            (
                torch.linalg.norm(self.actions[:, 0:2], dim=1),
                torch.linalg.norm(self.actions[:, 2:4], dim=1),
                torch.linalg.norm(self.actions[:, 4:6], dim=1),
            ),
            dim=1,
        )

        agv_action_rate = torch.stack(
            (
                torch.linalg.norm(self.actions[:, 0:2] - self.prev_actions[:, 0:2], dim=1),
                torch.linalg.norm(self.actions[:, 2:4] - self.prev_actions[:, 2:4], dim=1),
                torch.linalg.norm(self.actions[:, 4:6] - self.prev_actions[:, 4:6], dim=1),
            ),
            dim=1,
        )

        agv_payload_dists = torch.stack(
            (
                torch.linalg.norm(self.agv1.data.root_pos_w[:, :2] - payload_xy, dim=1),
                torch.linalg.norm(self.agv2.data.root_pos_w[:, :2] - payload_xy, dim=1),
                torch.linalg.norm(self.agv3.data.root_pos_w[:, :2] - payload_xy, dim=1),
            ),
            dim=1,
        )

        # D0A0c：第三台低贡献 AGV 不必强行参与，但已有两台有效推动时，
        # 低贡献 AGV 应保持低动作、低动作变化率，并处于可重新加入的待命距离。
        low_utility_weight = torch.clamp(
            (self.cfg.idle_low_utility_threshold - push_utility)
            / self.cfg.idle_low_utility_threshold,
            min=0.0,
            max=1.0,
        )
        idle_gate = (
                two_pusher_gate > self.cfg.idle_two_pusher_gate_threshold
        ).float()

        idle_action_penalty = torch.sum(
            idle_gate.unsqueeze(1) * low_utility_weight * agv_activity,
            dim=1,
        )
        idle_action_rate_penalty = torch.sum(
            idle_gate.unsqueeze(1) * low_utility_weight * agv_action_rate,
            dim=1,
        )

        standby_too_far = torch.clamp(
            agv_payload_dists - self.cfg.idle_standby_max_dist,
            min=0.0,
        )
        standby_too_close = torch.clamp(
            self.cfg.idle_standby_min_dist - agv_payload_dists,
            min=0.0,
        )
        idle_standby_penalty = torch.sum(
            idle_gate.unsqueeze(1)
            * low_utility_weight
            * (standby_too_far * standby_too_far + standby_too_close * standby_too_close),
            dim=1,
        )

        # 队形误差 + 角色相关朝向奖励
        desired_positions = self._compute_formation_target_xy()

        formation_errors = []
        heading_alignments = []

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            desired_xy = desired_positions[:, i, :]

            agv_to_formation = desired_xy - agv_xy

            dist_to_formation = torch.linalg.norm(
                agv_to_formation,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-6)

            formation_error = dist_to_formation.squeeze(-1)
            formation_errors.append(formation_error)

            approach_dir = agv_to_formation / dist_to_formation

            # 没到自己的队形点时，朝自己的队形点走；
            # 到队形点附近后，再朝 payload 推送方向走。
            close_to_formation = formation_error < 0.30

            desired_heading_dir = torch.where(
                close_to_formation.unsqueeze(-1),
                push_dir,
                approach_dir,
            )

            agv_heading_xy = torch.stack(
                (
                    torch.cos(self.agv_yaw[:, i]),
                    torch.sin(self.agv_yaw[:, i]),
                ),
                dim=1,
            )

            alignment = torch.sum(
                agv_heading_xy * desired_heading_dir,
                dim=1,
            )

            heading_alignments.append(alignment)

        formation_errors = torch.stack(formation_errors, dim=1)
        heading_alignments = torch.stack(heading_alignments, dim=1)

        formation_error_mean = formation_errors.mean(dim=1)
        formation_error_max = formation_errors.max(dim=1).values

        # D0A0：接触前引导奖励。
        # 在 payload 尚未明显前进时，模型仍可通过靠近队形目标点获得正反馈，
        # 避免从零训练时长期停在起点或随机后退。
        formation_progress = self.prev_formation_error_mean - formation_error_mean
        self.prev_formation_error_mean[:] = formation_error_mean.detach()
        approach_reward = torch.clamp(
            formation_progress / 0.005,
            min=-1.0,
            max=1.0,
        )

        heading_alignment_mean = heading_alignments.mean(dim=1)
        heading_alignment_min = heading_alignments.min(dim=1).values

        position_success = final_goal_dist < self.cfg.target_radius

        yaw_success = torch.abs(payload_yaw) < self.cfg.target_yaw_radius

        success = position_success & yaw_success

        out_of_bounds = self._compute_out_of_bounds()

        action_penalty = torch.sum(
            torch.square(self.actions),
            dim=1,
        )

        action_rate_penalty = torch.sum(
            torch.square(self.actions - self.prev_actions),
            dim=1,
        )

        angular_action_penalty = torch.sum(
            torch.square(self.actions[:, [1, 3, 5]]),
            dim=1,
        )

        angular_rate_penalty = torch.sum(
            torch.square(
                self.actions[:, [1, 3, 5]]
                - self.prev_actions[:, [1, 3, 5]]
            ),
            dim=1,
        )

        pairwise_distances = self._compute_agv_pairwise_distances()

        agv_overlap_penalty = torch.sum(
            torch.square(
                torch.clamp(
                    self.cfg.agv_safe_distance - pairwise_distances,
                    min=0.0,
                )
            ),
            dim=1,
        )

        agv_collision = self._compute_agv_collision()

        reward = (
            # 核心目标：payload 向目标移动。
            # 单车推可以获得少量基础进度奖励；两台 AGV 有效推时才获得主要进度奖励。
                progress_reward

                # 目标距离
                - 0.8 * payload_goal_dist

                # 距离最终终点
                - 0.6 * final_goal_dist

                - self.cfg.path_lateral_error_scale * path_lateral_error

                # payload 姿态稳定
                - 2.0 * payload_yaw_abs

                # 接近目标后加强位姿稳定
                - 2.0 * near_goal * payload_yaw_abs
                - 0.8 * near_goal * payload_yaw_rate_abs
                - 0.4 * near_goal * payload_speed
                # D0A0：奖励 top-2 有效推送；接触前先奖励靠近推送队形点。
                + contact_reward
                + self.cfg.approach_reward_scale * approach_reward
                + self.cfg.contact_zone_approach_reward_scale * contact_zone_approach_reward
                + self.cfg.contact_persistence_reward_scale * contact_persistence_reward
                - self.cfg.single_pusher_progress_penalty_scale * single_pusher_progress_penalty
                - self.cfg.progress_drop_penalty_scale * progress_drop
                - self.cfg.idle_action_penalty_scale * idle_action_penalty
                - self.cfg.idle_action_rate_penalty_scale * idle_action_rate_penalty
                - self.cfg.idle_standby_penalty_scale * idle_standby_penalty

                # 队形约束（弱化固定队形，允许两车/三车策略性切换）
                - self.cfg.formation_error_mean_scale * formation_error_mean
                - self.cfg.formation_error_max_scale * formation_error_max

                # 角色相关朝向奖励
                + self.cfg.heading_alignment_mean_scale * heading_alignment_mean
                + self.cfg.heading_alignment_min_scale * heading_alignment_min

                # 动作平滑
                - 0.005 * action_penalty
                - 0.015 * action_rate_penalty

                # 抑制角速度抖动
                - 0.010 * angular_action_penalty
                - 0.020 * angular_rate_penalty

                # AGV 间软分离
                - self.cfg.agv_overlap_penalty_scale * agv_overlap_penalty
                - self.cfg.agv_collision_penalty_scale * agv_collision.float()

                # V4.2-C0：软前端接触约束。
                # 不终止，只轻微压制车尾接触、接触倒车和明显车尾倒推。
                - self.cfg.rear_contact_penalty_scale * rear_contact_penalty
                - self.cfg.reverse_contact_penalty_scale * reverse_contact_penalty
                - self.cfg.contact_heading_penalty_scale * bad_contact_heading_penalty
                - self.cfg.bad_rear_push_penalty_scale * bad_rear_push.float()

                # 成功奖励按两车有效推动进度占比缩放，避免 AGV1 单车推到终点拿满分。
                + self.cfg.success_reward_scale * success.float() * success_quality_gate
                - 30.0 * out_of_bounds.float()

        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判断 episode 是否结束。

        V4.1C:
            - 不再进行离散 waypoint 切换；
            - target 由路径 lookahead 连续生成；
            - 只有到达最终路径终点并满足 yaw，才认为成功。
        """
        payload_xy = self.payload.data.root_pos_w[:, :2]

        final_target_xy = self._get_final_target_xy()

        final_goal_dist = torch.linalg.norm(
            payload_xy - final_target_xy,
            dim=1,
        )

        payload_yaw = self._get_payload_yaw()

        position_success = final_goal_dist < self.cfg.target_radius

        yaw_success = torch.abs(payload_yaw) < self.cfg.target_yaw_radius

        success = position_success & yaw_success

        out_of_bounds = self._compute_out_of_bounds()

        _, bad_rear_push = self._compute_bad_rear_push(payload_xy)
        self.last_bad_rear_push[:] = bad_rear_push.detach()

        # 保存 terminal 判断时刻的真实状态，供评估脚本读取
        self.last_success[:] = success.detach()
        self.last_position_success[:] = position_success.detach()
        self.last_yaw_success[:] = yaw_success.detach()
        self.last_out_of_bounds[:] = out_of_bounds.detach()
        self.last_payload_goal_dist[:] = final_goal_dist.detach()
        self.last_payload_yaw_abs[:] = torch.abs(payload_yaw).detach()

        if self.cfg.terminate_on_bad_rear_push:
            terminated = success | out_of_bounds | bad_rear_push
        else:
            terminated = success | out_of_bounds

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids_tensor = torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(
                list(env_ids),
                device=self.device,
                dtype=torch.long,
            )

        super()._reset_idx(env_ids_tensor)

        num_reset = env_ids_tensor.shape[0]
        env_origins = self.scene.env_origins[env_ids_tensor]

        identity_quat = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            device=self.device,
        ).repeat(num_reset, 1)

        # 重置三台 AGV
        for i, agv in enumerate(self.agvs):
            agv_state = agv.data.default_root_state[env_ids_tensor].clone()

            agv_offset = torch.tensor(
                self.cfg.agv_init_positions[i],
                device=self.device,
            ).repeat(num_reset, 1)

            agv_state[:, :3] = env_origins + agv_offset
            agv_state[:, 3:7] = identity_quat
            agv_state[:, 7:] = 0.0

            agv.write_root_pose_to_sim(agv_state[:, :7], env_ids_tensor)
            agv.write_root_velocity_to_sim(agv_state[:, 7:], env_ids_tensor)
            agv.reset(env_ids_tensor)

        # 重置 payload
        payload_state = self.payload.data.default_root_state[env_ids_tensor].clone()
        payload_offset = torch.tensor(
            self.cfg.payload_init_pos,
            device=self.device,
        ).repeat(num_reset, 1)

        payload_state[:, :3] = env_origins + payload_offset
        payload_state[:, 3:7] = identity_quat
        payload_state[:, 7:] = 0.0

        self.payload.write_root_pose_to_sim(payload_state[:, :7], env_ids_tensor)
        self.payload.write_root_velocity_to_sim(payload_state[:, 7:], env_ids_tensor)
        self.payload.reset(env_ids_tensor)

        self.actions[env_ids_tensor] = 0.0
        self.prev_actions[env_ids_tensor] = 0.0
        self.agv_yaw[env_ids_tensor, :] = 0.0
        # V4.0：每个 episode 从第 0 个 waypoint 开始
        self.current_waypoint_idx[env_ids_tensor] = 0
        self.prev_waypoint_idx[env_ids_tensor] = 0

        target_xy, _, path_progress, _ = self._compute_path_tracking_quantities()

        target_xy = target_xy[env_ids_tensor]
        path_progress = path_progress[env_ids_tensor]

        payload_xy = payload_state[:, :2]

        self.prev_path_progress[env_ids_tensor] = path_progress.detach()
        self.max_path_progress[env_ids_tensor] = path_progress.detach()
        self.episode_positive_progress[env_ids_tensor] = 0.0
        self.episode_two_pusher_progress[env_ids_tensor] = 0.0

        self.prev_payload_goal_dist[env_ids_tensor] = torch.linalg.norm(
            payload_xy - target_xy,
            dim=1,
        )

        formation_errors = self._compute_formation_errors()
        self.prev_formation_error_mean[env_ids_tensor] = formation_errors[
            env_ids_tensor
        ].mean(dim=1).detach()

        # 兼容旧变量，如果你代码里还保留 prev_agv_payload_dist
        if hasattr(self, "prev_agv_payload_dist"):
            agv1_xy = self.agv1.data.root_pos_w[env_ids_tensor, :2]
            self.prev_agv_payload_dist[env_ids_tensor] = torch.linalg.norm(
                agv1_xy - payload_xy,
                dim=1,
            )

    def _get_waypoints_xy(self) -> torch.Tensor:
        """返回所有 waypoint 的世界坐标。

        Returns:
            Tensor shape = [num_envs, num_waypoints, 2]
        """
        waypoint_offsets = torch.tensor(
            self.cfg.waypoints,
            device=self.device,
            dtype=torch.float32,
        )

        return self.scene.env_origins[:, None, :2] + waypoint_offsets[None, :, :]

    def _get_path_points_xy(self) -> torch.Tensor:
        """返回路径点序列。

        路径点包括：
            payload 初始点 + cfg.waypoints

        Returns:
            Tensor shape = [num_envs, num_points, 2]
        """
        start_offset = torch.tensor(
            self.cfg.payload_init_pos[:2],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 2)

        start_xy = self.scene.env_origins[:, None, :2] + start_offset

        waypoints_xy = self._get_waypoints_xy()

        return torch.cat(
            (start_xy, waypoints_xy),
            dim=1,
        )

    def _get_final_target_xy(self) -> torch.Tensor:
        """返回最终路径终点。"""
        waypoints_xy = self._get_waypoints_xy()
        return waypoints_xy[:, -1, :]

    def _compute_path_tracking_quantities(self):
        """计算连续路径跟踪所需的前视目标点、横向误差和路径进度。

        Returns:
            target_xy: 前视目标点，shape = [num_envs, 2]
            lateral_error: payload 到路径段的横向误差，shape = [num_envs]
            progress_s: payload 沿路径的投影进度，shape = [num_envs]
            segment_idx: payload 当前所在路径段编号，shape = [num_envs]
        """
        payload_xy = self.payload.data.root_pos_w[:, :2]

        path_points = self._get_path_points_xy()

        segment_start = path_points[:, :-1, :]
        segment_end = path_points[:, 1:, :]

        segment_vec = segment_end - segment_start

        segment_len = torch.linalg.norm(
            segment_vec,
            dim=2,
        ).clamp_min(1e-6)

        segment_len_sq = segment_len * segment_len

        payload_vec = payload_xy[:, None, :] - segment_start

        t = torch.sum(
            payload_vec * segment_vec,
            dim=2,
        ) / segment_len_sq

        t = torch.clamp(t, 0.0, 1.0)

        closest_points = segment_start + t.unsqueeze(-1) * segment_vec

        distances = torch.linalg.norm(
            payload_xy[:, None, :] - closest_points,
            dim=2,
        )

        segment_idx = torch.argmin(
            distances,
            dim=1,
        )

        env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        best_t = t[env_ids, segment_idx]

        lateral_error = distances[env_ids, segment_idx]

        cumulative_start = torch.cat(
            (
                torch.zeros(
                    (self.num_envs, 1),
                    device=self.device,
                ),
                torch.cumsum(segment_len[:, :-1], dim=1),
            ),
            dim=1,
        )

        progress_s = (
                cumulative_start[env_ids, segment_idx]
                + best_t * segment_len[env_ids, segment_idx]
        )

        total_length = torch.sum(
            segment_len,
            dim=1,
        )

        lookahead_dist = getattr(
            self.cfg,
            "path_lookahead_dist",
            0.25,
        )

        target_s = torch.clamp(
            progress_s + lookahead_dist,
            max=total_length,
        )

        cumulative_end = cumulative_start + segment_len

        target_segment_mask = target_s.unsqueeze(1) <= cumulative_end + 1e-6

        target_segment_idx = torch.argmax(
            target_segment_mask.to(torch.int64),
            dim=1,
        )

        target_local_s = (
                target_s
                - cumulative_start[env_ids, target_segment_idx]
        )

        target_t = torch.clamp(
            target_local_s / segment_len[env_ids, target_segment_idx],
            0.0,
            1.0,
        )

        target_xy = (
                segment_start[env_ids, target_segment_idx]
                + target_t.unsqueeze(-1) * segment_vec[env_ids, target_segment_idx]
        )

        # 这里保留 current_waypoint_idx，用于评估 CSV 中观察路径进度。
        # 在 V4.1C 中它表示“当前最近路径段编号”，不再表示离散 waypoint 是否通过。
        self.current_waypoint_idx[:] = segment_idx.detach()

        return target_xy, lateral_error, progress_s, segment_idx

    def _get_target_xy(self) -> torch.Tensor:
        """返回连续路径跟踪的 lookahead target。"""
        target_xy, _, _, _ = self._compute_path_tracking_quantities()
        return target_xy

    def _compute_path_lateral_error(self) -> torch.Tensor:
        """返回 payload 到当前路径的横向误差。"""
        _, lateral_error, _, _ = self._compute_path_tracking_quantities()
        return lateral_error

    def _compute_contact_flags(self) -> torch.Tensor:
        """返回三台 AGV 的前端接触状态，shape = [num_envs, 3]。

        D0A0c 使用 AGV 前边缘线段与 payload 后缘接触带的几何重叠判定。
        相比只检查 front center，线段判定对侧车 AGV2/AGV3 更公平：
        只要 AGV 前边缘有一部分与 payload 后缘横向区间重叠，就可判为有效前端接触。
        """
        payload_xy = self.payload.data.root_pos_w[:, :2]
        payload_yaw = self._get_payload_yaw()

        cos_yaw = torch.cos(payload_yaw)
        sin_yaw = torch.sin(payload_yaw)

        payload_half_x = 0.5 * self.cfg.payload_size[0]
        payload_half_y = 0.5 * self.cfg.payload_size[1]
        agv_half_length = 0.5 * self.cfg.agv_size[0]
        agv_half_width = 0.5 * self.cfg.agv_size[1]

        rear_face_x = -payload_half_x
        x_margin = getattr(self.cfg, "front_contact_x_margin", 0.12)
        y_margin = getattr(self.cfg, "front_contact_y_margin", 0.08)

        contact_flags = []

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            agv_heading_xy = torch.stack(
                (
                    torch.cos(self.agv_yaw[:, i]),
                    torch.sin(self.agv_yaw[:, i]),
                ),
                dim=1,
            )
            agv_lateral_xy = torch.stack(
                (-agv_heading_xy[:, 1], agv_heading_xy[:, 0]),
                dim=1,
            )

            front_center_xy = agv_xy + agv_heading_xy * agv_half_length
            front_left_xy = front_center_xy + agv_lateral_xy * agv_half_width
            front_right_xy = front_center_xy - agv_lateral_xy * agv_half_width

            def to_payload_local(points_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                rel = points_xy - payload_xy
                local_x = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 1]
                local_y = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]
                return local_x, local_y

            front_center_x, _ = to_payload_local(front_center_xy)
            left_x, left_y = to_payload_local(front_left_xy)
            right_x, right_y = to_payload_local(front_right_xy)

            front_x_min = torch.minimum(left_x, right_x)
            front_x_max = torch.maximum(left_x, right_x)
            front_y_min = torch.minimum(left_y, right_y)
            front_y_max = torch.maximum(left_y, right_y)

            rear_band_overlap = (
                (front_x_max > rear_face_x - x_margin)
                & (front_x_min < rear_face_x + x_margin)
            )
            # 如果前边缘与 payload 后缘近似平行，front_x_min/max 可能很窄；
            # 再加一个 front center 的后缘带判断，避免数值抖动漏判。
            rear_center_near = torch.abs(front_center_x - rear_face_x) < x_margin
            rear_band_ok = rear_band_overlap | rear_center_near

            lateral_overlap_ok = (
                (front_y_max > -payload_half_y - y_margin)
                & (front_y_min < payload_half_y + y_margin)
            )

            to_payload = payload_xy - agv_xy
            to_payload_dist = torch.linalg.norm(to_payload, dim=1, keepdim=True).clamp_min(1e-6)
            dir_to_payload = to_payload / to_payload_dist
            heading_to_payload = torch.sum(agv_heading_xy * dir_to_payload, dim=1)
            front_facing = heading_to_payload > self.cfg.front_contact_heading_min

            contact_flags.append(rear_band_ok & lateral_overlap_ok & front_facing)

        return torch.stack(contact_flags, dim=1)

    def _compute_contact_zone_errors(self, payload_xy: torch.Tensor | None = None) -> torch.Tensor:
        """计算每台 AGV 到 payload 后缘有效接触带的几何误差，shape = [num_envs, 3]。

        误差越小，说明该 AGV 的前边缘越接近可推动的后缘接触区。
        该量只用于接触前 shaping，不指定必须是哪两台 AGV 参与。
        """
        if payload_xy is None:
            payload_xy = self.payload.data.root_pos_w[:, :2]

        payload_yaw = self._get_payload_yaw()
        cos_yaw = torch.cos(payload_yaw)
        sin_yaw = torch.sin(payload_yaw)

        payload_half_x = 0.5 * self.cfg.payload_size[0]
        payload_half_y = 0.5 * self.cfg.payload_size[1]
        agv_half_length = 0.5 * self.cfg.agv_size[0]
        agv_half_width = 0.5 * self.cfg.agv_size[1]

        rear_face_x = -payload_half_x
        x_margin = getattr(self.cfg, "front_contact_x_margin", 0.12)
        y_margin = getattr(self.cfg, "front_contact_y_margin", 0.08)

        errors = []

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            agv_heading_xy = torch.stack(
                (
                    torch.cos(self.agv_yaw[:, i]),
                    torch.sin(self.agv_yaw[:, i]),
                ),
                dim=1,
            )
            agv_lateral_xy = torch.stack(
                (-agv_heading_xy[:, 1], agv_heading_xy[:, 0]),
                dim=1,
            )

            front_center_xy = agv_xy + agv_heading_xy * agv_half_length
            front_left_xy = front_center_xy + agv_lateral_xy * agv_half_width
            front_right_xy = front_center_xy - agv_lateral_xy * agv_half_width

            def to_payload_local(points_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                rel = points_xy - payload_xy
                local_x = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 1]
                local_y = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]
                return local_x, local_y

            front_center_x, _ = to_payload_local(front_center_xy)
            _, left_y = to_payload_local(front_left_xy)
            _, right_y = to_payload_local(front_right_xy)

            front_y_min = torch.minimum(left_y, right_y)
            front_y_max = torch.maximum(left_y, right_y)

            # 距离 payload 后缘 x 接触带。
            dx = torch.clamp(torch.abs(front_center_x - rear_face_x) - x_margin, min=0.0)

            # 距离 payload 横向可接触区间。若前边缘线段与区间重叠，则 dy=0。
            lower = -payload_half_y - y_margin
            upper = payload_half_y + y_margin
            dy_below = torch.clamp(lower - front_y_max, min=0.0)
            dy_above = torch.clamp(front_y_min - upper, min=0.0)
            dy = dy_below + dy_above

            errors.append(torch.sqrt(dx * dx + dy * dy))

        return torch.stack(errors, dim=1)

    def _compute_contact_geometry(self, payload_xy: torch.Tensor | None = None):
        """计算 AGV 与 payload 的前后接触几何关系。

        返回：
            heading_to_payload: 每台 AGV 车头方向与 AGV->payload 方向的点积，shape [num_envs, 3]
                > 0 表示车头大致朝向 payload；
                < 0 表示车尾大致朝向 payload。
            front_dists: 每台 AGV 前端点到 payload 中心的距离，shape [num_envs, 3]
            rear_dists: 每台 AGV 后端点到 payload 中心的距离，shape [num_envs, 3]
            v_actions: 每台 AGV 当前线速度动作，shape [num_envs, 3]
        """
        if payload_xy is None:
            payload_xy = self.payload.data.root_pos_w[:, :2]

        half_length = 0.5 * self.cfg.agv_size[0]

        heading_to_payload_list = []
        front_dist_list = []
        rear_dist_list = []
        v_action_list = []

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]

            agv_heading_xy = torch.stack(
                (
                    torch.cos(self.agv_yaw[:, i]),
                    torch.sin(self.agv_yaw[:, i]),
                ),
                dim=1,
            )

            to_payload = payload_xy - agv_xy
            to_payload_dist = torch.linalg.norm(
                to_payload,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-6)

            dir_to_payload = to_payload / to_payload_dist

            heading_to_payload = torch.sum(
                agv_heading_xy * dir_to_payload,
                dim=1,
            )

            front_xy = agv_xy + agv_heading_xy * half_length
            rear_xy = agv_xy - agv_heading_xy * half_length

            front_dist = torch.linalg.norm(front_xy - payload_xy, dim=1)
            rear_dist = torch.linalg.norm(rear_xy - payload_xy, dim=1)

            heading_to_payload_list.append(heading_to_payload)
            front_dist_list.append(front_dist)
            rear_dist_list.append(rear_dist)
            v_action_list.append(self.actions[:, 2 * i])

        heading_to_payload = torch.stack(heading_to_payload_list, dim=1)
        front_dists = torch.stack(front_dist_list, dim=1)
        rear_dists = torch.stack(rear_dist_list, dim=1)
        v_actions = torch.stack(v_action_list, dim=1)

        return heading_to_payload, front_dists, rear_dists, v_actions

    def _compute_bad_rear_push(self, payload_xy: torch.Tensor | None = None):
        """判断是否出现明显车尾倒推 payload。

        C0 阶段该项只用于 reward 软惩罚和评估记录；
        默认不作为 done 终止条件。
        """
        if payload_xy is None:
            payload_xy = self.payload.data.root_pos_w[:, :2]

        contact_flags = self._compute_contact_flags().float()

        (
            heading_to_payload,
            front_dists,
            rear_dists,
            v_actions,
        ) = self._compute_contact_geometry(payload_xy)

        rear_closer_than_front = (
            rear_dists + self.cfg.front_rear_margin < front_dists
        ).float()

        heading_away_from_payload = (
            heading_to_payload < self.cfg.bad_rear_heading_threshold
        ).float()

        reversing = (
            v_actions < self.cfg.bad_rear_reverse_threshold
        ).float()

        bad_rear_push_per_agv = (
            contact_flags
            * rear_closer_than_front
            * heading_away_from_payload
            * reversing
        )

        bad_rear_push = torch.any(
            bad_rear_push_per_agv > 0.5,
            dim=1,
        )

        return bad_rear_push_per_agv, bad_rear_push

    def _compute_contact_flag(self) -> torch.Tensor:
        """兼容旧接口：任意一台 AGV 接近 payload 即认为有接触。"""
        contact_flags = self._compute_contact_flags()
        return torch.any(contact_flags, dim=1)


    def _quat_to_yaw_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        """将 wxyz 四元数转换为 yaw。"""
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        return torch.atan2(siny_cosp, cosy_cosp)

    def _get_payload_yaw(self) -> torch.Tensor:
        """获取 payload 偏航角。"""
        payload_data = self.payload.data

        if hasattr(payload_data, "root_quat_w"):
            quat = payload_data.root_quat_w
        else:
            quat = payload_data.root_state_w[:, 3:7]

        return self._quat_to_yaw_wxyz(quat)

    def _get_payload_yaw_rate(self) -> torch.Tensor:
        """获取 payload yaw rate。"""
        payload_data = self.payload.data

        if hasattr(payload_data, "root_ang_vel_w"):
            return payload_data.root_ang_vel_w[:, 2]

        return payload_data.root_state_w[:, 12]

    def _compute_formation_target_xy(self) -> torch.Tensor:
        """计算三台 AGV 的目标队形点。

        Returns:
            Tensor shape = [num_envs, 3, 2]
        """
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()

        payload_to_target = target_xy - payload_xy
        dist_to_target = torch.linalg.norm(
            payload_to_target,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        push_dir = payload_to_target / dist_to_target

        lateral_dir = torch.stack(
            (-push_dir[:, 1], push_dir[:, 0]),
            dim=1,
        )

        stand_off_distances = torch.tensor(
            self.cfg.formation_stand_off_distances,
            device=self.device,
        )

        lateral_offsets = torch.tensor(
            self.cfg.formation_lateral_offsets,
            device=self.device,
        )

        desired_positions = []

        for i in range(3):
            desired_xy = (
                    payload_xy
                    - push_dir * stand_off_distances[i]
                    + lateral_dir * lateral_offsets[i]
            )
            desired_positions.append(desired_xy)

        return torch.stack(desired_positions, dim=1)

    def _compute_formation_errors(self) -> torch.Tensor:
        """计算三台 AGV 相对目标队形点的误差。

        Returns:
            Tensor shape = [num_envs, 3]
        """
        desired_positions = self._compute_formation_target_xy()

        errors = []

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            desired_xy = desired_positions[:, i, :]

            error = torch.linalg.norm(
                agv_xy - desired_xy,
                dim=1,
            )

            errors.append(error)

        return torch.stack(errors, dim=1)

    def _compute_agv_pairwise_distances(self) -> torch.Tensor:
        """计算三台 AGV 两两之间的距离。

        Returns:
            Tensor shape = [num_envs, 3]
            columns = [d12, d13, d23]
        """
        agv1_xy = self.agv1.data.root_pos_w[:, :2]
        agv2_xy = self.agv2.data.root_pos_w[:, :2]
        agv3_xy = self.agv3.data.root_pos_w[:, :2]

        d12 = torch.linalg.norm(agv1_xy - agv2_xy, dim=1)
        d13 = torch.linalg.norm(agv1_xy - agv3_xy, dim=1)
        d23 = torch.linalg.norm(agv2_xy - agv3_xy, dim=1)

        return torch.stack((d12, d13, d23), dim=1)

    def _compute_agv_collision(self) -> torch.Tensor:
        """判断 AGV 之间是否发生严重重叠。"""
        pairwise_distances = self._compute_agv_pairwise_distances()
        return torch.any(
            pairwise_distances < self.cfg.agv_collision_distance,
            dim=1,
        )

    def _compute_out_of_bounds(self) -> torch.Tensor:
        """判断三台 AGV 或 payload 是否超出当前环境工作空间。"""
        env_xy = self.scene.env_origins[:, :2]

        payload_rel_xy = self.payload.data.root_pos_w[:, :2] - env_xy
        payload_oob = torch.any(
            torch.abs(payload_rel_xy) > self.cfg.workspace_limit,
            dim=1,
        )

        agv_oob = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        for agv in self.agvs:
            agv_rel_xy = agv.data.root_pos_w[:, :2] - env_xy
            agv_oob = agv_oob | torch.any(
                torch.abs(agv_rel_xy) > self.cfg.workspace_limit,
                dim=1,
            )

        return payload_oob | agv_oob

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """把 yaw 角转换成 wxyz 四元数。"""
        half_yaw = 0.5 * yaw
        quat = torch.zeros((yaw.shape[0], 4), device=self.device)
        quat[:, 0] = torch.cos(half_yaw)  # w
        quat[:, 1] = 0.0  # x
        quat[:, 2] = 0.0  # y
        quat[:, 3] = torch.sin(half_yaw)  # z
        return quat

    def _wrap_to_pi(self, angle: torch.Tensor) -> torch.Tensor:
        """角度归一化到 [-pi, pi]。"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))