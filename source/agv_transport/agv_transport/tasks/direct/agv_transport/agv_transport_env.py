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
        self.agv_yaw = torch.zeros(self.num_envs, device=self.device)

        self._agv_height = self.cfg.agv_init_pos[2]
        self._identity_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(self.num_envs, 1)
        # 初始化一次距离缓存
        target_xy = self._get_target_xy()
        payload_xy = self.payload.data.root_pos_w[:, :2]
        agv_xy = self.agv.data.root_pos_w[:, :2]

        self.prev_payload_goal_dist[:] = torch.linalg.norm(payload_xy - target_xy, dim=1)
        self.prev_agv_payload_dist[:] = torch.linalg.norm(agv_xy - payload_xy, dim=1)

    def _hide_agv_collision_visual(self) -> None:
        """隐藏简化 AGV 碰撞体的视觉显示，只保留小车 USD 外观。"""

        stage = sim_utils.get_current_stage()

        for env_id in range(self.num_envs):
            agv_root_path = f"/World/envs/env_{env_id}/AGV"
            agv_root = stage.GetPrimAtPath(agv_root_path)

            if not agv_root.IsValid():
                continue

            for prim in Usd.PrimRange(agv_root):
                prim_path = prim.GetPath().pathString

                # 不隐藏小车外观模型
                if "/Visual" in prim_path:
                    continue

                # 不隐藏 AGV 根节点，否则子节点 Visual 也可能受影响
                if prim_path == agv_root_path:
                    continue

                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
    def _setup_scene(self):
        # 创建刚体
        self.agv = RigidObject(self.cfg.agv_cfg)
        self.payload = RigidObject(self.cfg.payload_cfg)

        # 先注册到 scene，再 clone environments
        self.scene.rigid_objects["agv"] = self.agv
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
        self.cfg.agv_visual_cfg.func(
            "/World/envs/env_0/AGV/Visual",
            self.cfg.agv_visual_cfg,
            translation=(0.0, 0.0, -0.03),
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
        """把 action=[v, w] 转换成差速 AGV 运动。

        action[:, 0] -> 归一化线速度，范围 [-1, 1]
        action[:, 1] -> 归一化角速度，范围 [-1, 1]
        """
        dt = self.cfg.sim.dt * self.cfg.decimation

        # 当前 AGV 根状态
        agv_state = self.agv.data.root_state_w.clone()

        # 动作缩放
        linear_speed = self.actions[:, 0] * self.cfg.max_agv_linear_speed
        angular_speed = self.actions[:, 1] * self.cfg.max_agv_angular_speed

        # 更新 yaw
        self.agv_yaw = self.agv_yaw + angular_speed * dt
        self.agv_yaw = torch.atan2(torch.sin(self.agv_yaw), torch.cos(self.agv_yaw))

        # 根据差速运动学更新位置
        heading_x = torch.cos(self.agv_yaw)
        heading_y = torch.sin(self.agv_yaw)

        delta_x = linear_speed * heading_x * dt
        delta_y = linear_speed * heading_y * dt

        new_xy = agv_state[:, :2] + torch.stack((delta_x, delta_y), dim=1)

        # 限制在每个环境局部工作空间内
        env_xy = self.scene.env_origins[:, :2]
        rel_xy = new_xy - env_xy
        rel_xy = torch.clamp(rel_xy, -self.cfg.workspace_limit, self.cfg.workspace_limit)
        new_xy = env_xy + rel_xy

        # yaw -> quaternion, Isaac Lab / USD 常用 wxyz
        quat = self._yaw_to_quat(self.agv_yaw)

        # 写回 AGV 姿态和速度
        agv_state[:, :2] = new_xy
        agv_state[:, 2] = self._agv_height
        agv_state[:, 3:7] = quat

        agv_state[:, 7] = linear_speed * heading_x
        agv_state[:, 8] = linear_speed * heading_y
        agv_state[:, 9] = 0.0
        agv_state[:, 10] = 0.0
        agv_state[:, 11] = 0.0
        agv_state[:, 12] = angular_speed

        self.agv.write_root_pose_to_sim(agv_state[:, :7])
        self.agv.write_root_velocity_to_sim(agv_state[:, 7:])

    def _get_observations(self) -> dict:
        env_xy = self.scene.env_origins[:, :2]

        agv_xy = self.agv.data.root_pos_w[:, :2]
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()

        agv_vel_xy = self.agv.data.root_lin_vel_w[:, :2]
        payload_vel_xy = self.payload.data.root_lin_vel_w[:, :2]

        agv_xy_rel = agv_xy - env_xy
        payload_xy_rel = payload_xy - env_xy
        target_xy_rel = target_xy - env_xy

        agv_to_payload_xy = payload_xy - agv_xy
        payload_to_target_xy = target_xy - payload_xy

        # AGV 朝向向量
        agv_heading_xy = torch.stack(
            (
                torch.cos(self.agv_yaw),
                torch.sin(self.agv_yaw),
            ),
            dim=1,
        )

        obs = torch.cat(
            (
                agv_xy_rel,
                payload_xy_rel,
                target_xy_rel,
                agv_to_payload_xy,
                payload_to_target_xy,
                agv_heading_xy,
                agv_vel_xy,
                payload_vel_xy,
            ),
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        agv_xy = self.agv.data.root_pos_w[:, :2]
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()

        payload_to_target = target_xy - payload_xy
        agv_to_payload = payload_xy - agv_xy

        payload_goal_dist = torch.linalg.norm(payload_to_target, dim=1)
        agv_payload_dist = torch.linalg.norm(agv_to_payload, dim=1)

        # 1. payload 向目标移动的进度奖励
        payload_progress = self.prev_payload_goal_dist - payload_goal_dist
        self.prev_payload_goal_dist = payload_goal_dist.detach()

        # 2. AGV 靠近 payload 的进度奖励
        approach_progress = self.prev_agv_payload_dist - agv_payload_dist
        self.prev_agv_payload_dist = agv_payload_dist.detach()

        # 3. 接触判断
        contact_flag = self._compute_contact_flag()

        # 4. 计算期望推送位姿：payload 后方
        payload_to_target_dir = payload_to_target / payload_goal_dist.unsqueeze(-1).clamp_min(1e-6)

        stand_off_distance = 0.75
        desired_agv_xy = payload_xy - payload_to_target_dir * stand_off_distance
        push_pose_dist = torch.linalg.norm(agv_xy - desired_agv_xy, dim=1)

        # 5. AGV 朝向与期望推送方向对齐奖励
        agv_heading_xy = torch.stack(
            (
                torch.cos(self.agv_yaw),
                torch.sin(self.agv_yaw),
            ),
            dim=1,
        )

        heading_alignment = torch.sum(agv_heading_xy * payload_to_target_dir, dim=1)
        heading_alignment = torch.clamp(heading_alignment, min=0.0, max=1.0)

        # 6. 成功与失败
        success = payload_goal_dist < self.cfg.target_radius
        out_of_bounds = self._compute_out_of_bounds()

        # 7. 动作惩罚：重点抑制角速度抖动
        linear_action = self.actions[:, 0]
        angular_action = self.actions[:, 1]

        action_penalty = (
                0.003 * torch.square(linear_action)
                + 0.02 * torch.square(angular_action)
        )

        # 8. 动作变化率惩罚：抑制前后动作突变，减少车头摆动
        action_rate_penalty = torch.sum(
            torch.square(self.actions - self.prev_actions),
            dim=1,
        )

        # 9. 分阶段 reward
        reward = (
                15.0 * payload_progress
                + 3.0 * approach_progress
                - 0.8 * push_pose_dist
                + 0.5 * heading_alignment
                + 1.0 * contact_flag.float()
                - 0.5 * payload_goal_dist
                - action_penalty
                - 0.02 * action_rate_penalty
                + 50.0 * success.float()
                - 20.0 * out_of_bounds.float()
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
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # 每个环境的原点
        env_origins = self.scene.env_origins[env_ids]

        # AGV 初始状态
        agv_state = self.agv.data.default_root_state[env_ids].clone()
        agv_offset = torch.tensor(
            self.cfg.agv_init_pos,
            device=self.device,
        ).repeat(num_reset, 1)

        # 随机化 AGV 初始 y 位置，用于提高策略泛化能力
        if self.cfg.randomize_agv_init_y:
            y_min, y_max = self.cfg.agv_init_y_range
            random_y = y_min + (y_max - y_min) * torch.rand(
                num_reset,
                device=self.device,
            )
            agv_offset[:, 1] = random_y

        agv_state[:, :3] = env_origins + agv_offset
        agv_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(num_reset, 1)
        agv_state[:, 7:] = 0.0

        # Payload 初始状态
        payload_state = self.payload.data.default_root_state[env_ids].clone()
        payload_offset = torch.tensor(self.cfg.payload_init_pos, device=self.device).repeat(num_reset, 1)
        payload_state[:, :3] = env_origins + payload_offset
        payload_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(num_reset, 1)
        payload_state[:, 7:] = 0.0

        # 写入仿真
        self.agv.write_root_pose_to_sim(agv_state[:, :7], env_ids)
        self.agv.write_root_velocity_to_sim(agv_state[:, 7:], env_ids)
        self.payload.write_root_pose_to_sim(payload_state[:, :7], env_ids)
        self.payload.write_root_velocity_to_sim(payload_state[:, 7:], env_ids)

        self.agv.reset(env_ids)
        self.payload.reset(env_ids)

        # 重置动作和 AGV 朝向
        self.actions[env_ids] = 0.0
        self.agv_yaw[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        target_xy = self._get_target_xy()[env_ids]
        payload_xy = payload_state[:, :2]

        agv_xy = agv_state[:, :2]

        self.prev_payload_goal_dist[env_ids] = torch.linalg.norm(payload_xy - target_xy, dim=1)
        self.prev_agv_payload_dist[env_ids] = torch.linalg.norm(agv_xy - payload_xy, dim=1)

    def _get_target_xy(self) -> torch.Tensor:
        """返回每个环境中的目标点世界坐标 xy。"""
        target_offset_xy = torch.tensor(self.cfg.target_pos[:2], device=self.device).unsqueeze(0)
        return self.scene.env_origins[:, :2] + target_offset_xy

    def _compute_contact_flag(self) -> torch.Tensor:
        """用距离近似判断 AGV 是否与 payload 接触。"""
        agv_xy = self.agv.data.root_pos_w[:, :2]
        payload_xy = self.payload.data.root_pos_w[:, :2]

        agv_payload_dist = torch.linalg.norm(agv_xy - payload_xy, dim=1)

        # 当前 AGV 底盘和 payload 尺寸下，0.90 m 左右可作为近似接触阈值
        contact_flag = agv_payload_dist < 0.75

        return contact_flag

    def _compute_out_of_bounds(self) -> torch.Tensor:
        """判断 AGV 或 payload 是否超出当前环境工作空间。"""
        env_xy = self.scene.env_origins[:, :2]

        agv_rel_xy = self.agv.data.root_pos_w[:, :2] - env_xy
        payload_rel_xy = self.payload.data.root_pos_w[:, :2] - env_xy

        agv_oob = torch.any(torch.abs(agv_rel_xy) > self.cfg.workspace_limit, dim=1)
        payload_oob = torch.any(torch.abs(payload_rel_xy) > self.cfg.workspace_limit, dim=1)

        return agv_oob | payload_oob

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