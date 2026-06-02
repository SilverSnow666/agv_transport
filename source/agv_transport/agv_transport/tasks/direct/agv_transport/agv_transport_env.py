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
    """单 AGV 推箱子任务。

    当前版本是最小闭环：
    - 一个蓝色方块代表 AGV；
    - 一个橙色方块代表 payload；
    - AGV 使用 [v, w] 差速运动学控制；
    - 目标是把 payload 推到固定目标点。
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
        # V4.0：每个环境当前跟踪的 waypoint 编号
        self.current_waypoint_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.path_segment_idx = torch.zeros(
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
        target_xy, _, path_progress, _ = self._compute_path_tracking_quantities(
            update_segment=False
        )

        payload_xy = self.payload.data.root_pos_w[:, :2]
        agv_xy = self.agv.data.root_pos_w[:, :2]

        self.prev_path_progress[:] = path_progress.detach()
        self.prev_payload_goal_dist[:] = torch.linalg.norm(
            payload_xy - target_xy,
            dim=1,
        )
        self.prev_agv_payload_dist[:] = torch.linalg.norm(
            agv_xy - payload_xy,
            dim=1,
        )

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
            self._compute_path_tracking_quantities(update_segment=True)
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
        contact_count = contact_flags.float().sum(dim=1)
        all_three_contact = torch.all(
            contact_flags,
            dim=1,
        ).float()

        # 只有沿路径真实前进时，才给接触奖励
        progress_gate = torch.clamp(
            path_progress_delta / 0.005,
            min=0.0,
            max=1.0,
        )

        contact_reward = progress_gate * (
                0.25 * contact_count
                + 0.8 * all_three_contact
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

        reward = (
            # 核心目标：payload 向目标移动
                30.0 * path_progress_delta

                # 目标距离
                - 0.8 * payload_goal_dist

                # 距离最终终点
                - 0.6 * final_goal_dist

                - 0.3 * path_lateral_error

                # payload 姿态稳定
                - 2.0 * payload_yaw_abs

                # 接近目标后加强位姿稳定
                - 2.0 * near_goal * payload_yaw_abs
                - 0.8 * near_goal * payload_yaw_rate_abs
                - 0.4 * near_goal * payload_speed
                # 鼓励多车接近 payload，但不要主导训练
                + contact_reward

                # 队形约束
                - 0.5 * formation_error_mean
                - 0.15 * formation_error_max

                # 角色相关朝向奖励
                + 0.45 * heading_alignment_mean
                + 0.15 * heading_alignment_min

                # 动作平滑
                - 0.005 * action_penalty
                - 0.015 * action_rate_penalty

                # 抑制角速度抖动
                - 0.010 * angular_action_penalty
                - 0.020 * angular_rate_penalty

                # AGV 间软分离
                - 2.0 * agv_overlap_penalty

                # 成功奖励
                + 150.0 * success.float()
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

        # 保存 terminal 判断时刻的真实状态，供评估脚本读取
        self.last_success[:] = success.detach()
        self.last_position_success[:] = position_success.detach()
        self.last_yaw_success[:] = yaw_success.detach()
        self.last_out_of_bounds[:] = out_of_bounds.detach()
        self.last_payload_goal_dist[:] = final_goal_dist.detach()
        self.last_payload_yaw_abs[:] = torch.abs(payload_yaw).detach()

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
        self.path_segment_idx[env_ids_tensor] = 0

        target_xy, _, path_progress, _ = self._compute_path_tracking_quantities(
            update_segment=False
        )

        target_xy = target_xy[env_ids_tensor]
        path_progress = path_progress[env_ids_tensor]

        payload_xy = payload_state[:, :2]

        self.prev_path_progress[env_ids_tensor] = path_progress.detach()

        self.prev_payload_goal_dist[env_ids_tensor] = torch.linalg.norm(
            payload_xy - target_xy,
            dim=1,
        )

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

    def _compute_path_tracking_quantities(self, update_segment: bool = False):
        """计算连续路径跟踪所需的前视目标点、横向误差和路径进度。

        V4.1B-OrderedPath:
            - 不再使用“全局最近路径段”；
            - 每个环境维护 path_segment_idx；
            - payload 只能按路径段顺序前进；
            - 防止 payload 直线抄近路、跳到后面的路径段。

        Args:
            update_segment:
                True 时允许更新 path_segment_idx。
                只应该在 _get_rewards() 中设为 True，
                其他地方保持 False，避免一次 step 内多次跳段。

        Returns:
            target_xy: lookahead 目标点, shape = [num_envs, 2]
            lateral_error: payload 到当前路径段的横向误差, shape = [num_envs]
            progress_s: payload 沿路径的累计进度, shape = [num_envs]
            current_idx: 当前路径段编号, shape = [num_envs]
        """
        payload_xy = self.payload.data.root_pos_w[:, :2]

        path_points = self._get_path_points_xy()

        segment_start_all = path_points[:, :-1, :]
        segment_end_all = path_points[:, 1:, :]

        segment_vec_all = segment_end_all - segment_start_all

        segment_len_all = torch.linalg.norm(
            segment_vec_all,
            dim=2,
        ).clamp_min(1e-6)

        num_segments = segment_start_all.shape[1]

        env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        current_idx = torch.clamp(
            self.path_segment_idx,
            min=0,
            max=num_segments - 1,
        )

        # 只在当前段上投影
        segment_start = segment_start_all[env_ids, current_idx]
        segment_vec = segment_vec_all[env_ids, current_idx]
        segment_len = segment_len_all[env_ids, current_idx]

        payload_vec = payload_xy - segment_start

        t = torch.sum(
            payload_vec * segment_vec,
            dim=1,
        ) / (segment_len * segment_len)

        t = torch.clamp(t, 0.0, 1.0)

        # 只有 update_segment=True 时，才允许路径段向前切换
        if update_segment:
            switch_t = getattr(
                self.cfg,
                "path_segment_switch_t",
                0.85,
            )

            should_advance = (
                    (t > switch_t)
                    & (current_idx < num_segments - 1)
            )

            if torch.any(should_advance):
                current_idx = torch.where(
                    should_advance,
                    current_idx + 1,
                    current_idx,
                )

                self.path_segment_idx[:] = current_idx

                # 切换到新段后重新计算投影
                segment_start = segment_start_all[env_ids, current_idx]
                segment_vec = segment_vec_all[env_ids, current_idx]
                segment_len = segment_len_all[env_ids, current_idx]

                payload_vec = payload_xy - segment_start

                t = torch.sum(
                    payload_vec * segment_vec,
                    dim=1,
                ) / (segment_len * segment_len)

                t = torch.clamp(t, 0.0, 1.0)
            else:
                self.path_segment_idx[:] = current_idx

        closest_point = segment_start + t.unsqueeze(-1) * segment_vec

        lateral_error = torch.linalg.norm(
            payload_xy - closest_point,
            dim=1,
        )

        cumulative_start = torch.cat(
            (
                torch.zeros(
                    (self.num_envs, 1),
                    device=self.device,
                ),
                torch.cumsum(segment_len_all[:, :-1], dim=1),
            ),
            dim=1,
        )

        progress_s = (
                cumulative_start[env_ids, current_idx]
                + t * segment_len
        )

        total_length = torch.sum(
            segment_len_all,
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

        cumulative_end = cumulative_start + segment_len_all

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
            target_local_s / segment_len_all[env_ids, target_segment_idx],
            0.0,
            1.0,
        )

        target_xy = (
                segment_start_all[env_ids, target_segment_idx]
                + target_t.unsqueeze(-1)
                * segment_vec_all[env_ids, target_segment_idx]
        )

        # 继续用 current_waypoint_idx 给 eval CSV 记录路径段编号
        self.current_waypoint_idx[:] = current_idx.detach()

        return target_xy, lateral_error, progress_s, current_idx

    def _get_target_xy(self) -> torch.Tensor:
        """返回连续路径跟踪的 lookahead target。"""
        target_xy, _, _, _ = self._compute_path_tracking_quantities()
        return target_xy

    def _compute_path_lateral_error(self) -> torch.Tensor:
        """返回 payload 到当前路径的横向误差。"""
        _, lateral_error, _, _ = self._compute_path_tracking_quantities()
        return lateral_error

    def _compute_contact_flags(self) -> torch.Tensor:
        """返回三台 AGV 的近似接触状态，shape = [num_envs, 3]。"""
        payload_xy = self.payload.data.root_pos_w[:, :2]

        contact_flags = []

        for agv in self.agvs:
            agv_xy = agv.data.root_pos_w[:, :2]
            agv_payload_dist = torch.linalg.norm(
                agv_xy - payload_xy,
                dim=1,
            )

            contact_flags.append(
                agv_payload_dist < self.cfg.train_contact_threshold
            )

        return torch.stack(contact_flags, dim=1)

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