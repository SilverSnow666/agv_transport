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
        self.prev_agv_payload_dist = torch.zeros(self.num_envs, device=self.device)

        # AGV 朝向角，单位 rad
        self.agv_yaw = torch.zeros((self.num_envs, 3), device=self.device)

        self._agv_height = self.cfg.agv_init_pos[2]
        self._identity_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(self.num_envs, 1)
        # 初始化一次距离缓存
        target_xy = self._get_target_xy()
        payload_xy = self.payload.data.root_pos_w[:, :2]
        agv_xy = self.agv.data.root_pos_w[:, :2]

        self.prev_payload_goal_dist[:] = torch.linalg.norm(payload_xy - target_xy, dim=1)
        self.prev_agv_payload_dist[:] = torch.linalg.norm(agv_xy - payload_xy, dim=1)

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
        target_marker_cfg = sim_utils.CuboidCfg(
            size=(0.18, 0.18, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                metallic=0.0,
            ),
        )

        target_marker_cfg.func(
            "/World/envs/env_0/TargetMarker",
            target_marker_cfg,
            translation=(self.cfg.target_pos[0], self.cfg.target_pos[1], 0.02),
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

        obs_parts.extend(
            [
                payload_heading,
                payload_yaw_rate,
                contact_flags,
            ]
        )

        obs = torch.cat(obs_parts, dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()

        payload_to_target = target_xy - payload_xy
        payload_goal_dist = torch.linalg.norm(
            payload_to_target,
            dim=1,
        )

        push_dir = payload_to_target / payload_goal_dist.unsqueeze(-1).clamp_min(1e-6)

        # payload 前进量
        payload_progress = self.prev_payload_goal_dist - payload_goal_dist
        self.prev_payload_goal_dist = payload_goal_dist.detach()

        positive_progress = torch.clamp(payload_progress, min=0.0)
        negative_progress = torch.clamp(-payload_progress, min=0.0)

        payload_yaw = self._get_payload_yaw()

        contact_flags = self._compute_contact_flags()
        contact_values = contact_flags.float()
        contact_count = contact_values.sum(dim=1)

        at_least_two_contact = (contact_count >= 2.0).float()
        all_three_contact = (contact_count >= 3.0).float()

        formation_errors = self._compute_formation_errors()
        formation_error_mean = formation_errors.mean(dim=1)
        formation_error_max = formation_errors.max(dim=1).values

        # 三台 AGV 车头朝向目标推送方向
        heading_alignments = []
        desired_yaw = torch.atan2(push_dir[:, 1], push_dir[:, 0])

        for i in range(3):
            agv_yaw_i = self.agv_yaw[:, i]

            yaw_error = torch.atan2(
                torch.sin(desired_yaw - agv_yaw_i),
                torch.cos(desired_yaw - agv_yaw_i),
            )

            heading_alignments.append(torch.cos(yaw_error))

        heading_alignment_mean = torch.stack(
            heading_alignments,
            dim=1,
        ).mean(dim=1)

        success = payload_goal_dist < self.cfg.target_radius
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

        # 关键：三车协同推动奖励，必须和 payload 正向前进绑定
        cooperative_push_reward = positive_progress * (
                1.0
                + 0.8 * contact_count
                + 1.2 * at_least_two_contact
                + 0.8 * all_three_contact
        )

        # 卡住惩罚：多车接触但 payload 几乎不动
        stuck = (
                (contact_count >= 2.0)
                & (positive_progress < 0.0005)
                & (~success)
        ).float()

        reward = (
            # 主要目标：payload 朝目标前进
                35.0 * positive_progress
                - 12.0 * negative_progress
                - 0.8 * payload_goal_dist

                # 协同奖励必须服务于“推动前进”
                + 60.0 * cooperative_push_reward

                # 保留少量接触奖励，但不能主导
                + 0.10 * contact_count
                + 0.20 * at_least_two_contact
                + 0.10 * all_three_contact

                # 姿态稳定
                - 2.0 * torch.abs(payload_yaw)

                # 队形约束降低，避免过度保守
                - 0.25 * formation_error_mean
                - 0.20 * formation_error_max

                # 车头方向适度奖励
                + 0.30 * heading_alignment_mean

                # 防止三车贴住但不推
                - 0.30 * stuck

                # 动作惩罚降低，避免车不敢用力
                - 0.003 * action_penalty
                - 0.006 * action_rate_penalty
                - 0.006 * angular_action_penalty

                # 时间惩罚，鼓励尽快完成
                - 0.03

                # 成功奖励
                + 120.0 * success.float()
                - 30.0 * out_of_bounds.float()
        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()

        payload_goal_dist = torch.linalg.norm(payload_xy - target_xy, dim=1)
        success = payload_goal_dist < self.cfg.target_radius

        out_of_bounds = self._compute_out_of_bounds()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = success | out_of_bounds

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

        target_xy = self._get_target_xy()[env_ids_tensor]
        payload_xy = payload_state[:, :2]

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

    def _get_target_xy(self) -> torch.Tensor:
        """返回每个环境中的目标点世界坐标 xy。"""
        target_offset_xy = torch.tensor(self.cfg.target_pos[:2], device=self.device).unsqueeze(0)
        return self.scene.env_origins[:, :2] + target_offset_xy

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

    def _compute_formation_errors(self) -> torch.Tensor:
        """计算三台 AGV 相对 payload 后方目标队形点的误差。

        Returns:
            Tensor shape = [num_envs, 3]
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

        errors = []

        for i, agv in enumerate(self.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]

            desired_xy = (
                    payload_xy
                    - push_dir * stand_off_distances[i]
                    + lateral_dir * lateral_offsets[i]
            )

            error = torch.linalg.norm(
                agv_xy - desired_xy,
                dim=1,
            )

            errors.append(error)

        return torch.stack(errors, dim=1)

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