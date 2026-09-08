from __future__ import annotations

from collections.abc import Sequence
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from pxr import Gf, Usd, UsdGeom, Vt

from .agv_level_carry_env_cfg import AgvLevelCarryEnvCfg

class AgvLevelCarryEnv(DirectRLEnv):
    """V6.2.6 在 V6.2.5 基础上增加后侧 AGV 横向位置闭环。

    与 V5.x 推送环境不同，本环境把 payload 放在三台 AGV 顶部，依靠接触、重力和摩擦
    进行承载运输。V6.1 在平地矩形 payload 成功 baseline 上打开轻度支撑面高度扰动：
    - 三台 AGV 保持三角支撑队形；
    - payload 在轻度起伏支撑面上到达目标；
    - 额外记录 terrain/support-z-gap/payload vertical motion 指标。

    该环境不依赖真实接触传感器，训练 reward 中的 contact_flags 是解析近似量：
    AGV 位于对应支撑目标附近，且 AGV 顶面接近 payload 底面。
    """

    cfg: AgvLevelCarryEnvCfg

    def __init__(self, cfg: AgvLevelCarryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.agv_yaw = torch.zeros((self.num_envs, 3), device=self.device)
        self.agv_planar_vel = torch.zeros((self.num_envs, 3, 2), device=self.device)
        self.base_z_disturbance = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_target_height = torch.full(
            (self.num_envs, 3), float(self.cfg.lift_neutral_height), device=self.device
        )
        self.lift_height = torch.full(
            (self.num_envs, 3), float(self.cfg.lift_neutral_height), device=self.device
        )
        self.lift_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.agv_terrain_roll = torch.zeros((self.num_envs, 3), device=self.device)
        self.agv_terrain_pitch = torch.zeros((self.num_envs, 3), device=self.device)
        self.agv_terrain_samples = torch.zeros((self.num_envs, 3, 4), device=self.device)
        self._update_native_lift_visuals()

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

        # V6.2.6 后侧横向闭环的执行层诊断量。
        self.last_rear_lateral_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_rear_lateral_correction = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_applied_angular_speed = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_virtual_carry_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_virtual_stabilization_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    # ---------------------------------------------------------------------
    # Scene
    # ---------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.agv1 = RigidObject(self.cfg.agv1_cfg)
        self.agv2 = RigidObject(self.cfg.agv2_cfg)
        self.agv3 = RigidObject(self.cfg.agv3_cfg)
        self.lift1 = RigidObject(self.cfg.lift1_cfg)
        self.lift2 = RigidObject(self.cfg.lift2_cfg)
        self.lift3 = RigidObject(self.cfg.lift3_cfg)
        self.payload = RigidObject(self.cfg.payload_cfg)
        self.cargo = RigidObject(self.cfg.cargo_cfg) if bool(self.cfg.enable_cargo) else None
        self.agvs = [self.agv1, self.agv2, self.agv3]
        self.lifts = [self.lift1, self.lift2, self.lift3]

        self.scene.rigid_objects["agv1"] = self.agv1
        self.scene.rigid_objects["agv2"] = self.agv2
        self.scene.rigid_objects["agv3"] = self.agv3
        self.scene.rigid_objects["lift1"] = self.lift1
        self.scene.rigid_objects["lift2"] = self.lift2
        self.scene.rigid_objects["lift3"] = self.lift3
        self.scene.rigid_objects["payload"] = self.payload
        if self.cargo is not None:
            self.scene.rigid_objects["cargo"] = self.cargo

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self._move_ground_plane_for_visual_terrain()
        self._spawn_visual_bumpy_terrain()

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

        # 黄色 USD 外观与 AGV 根节点保持既有标定。
        for agv_name in ["AGV1", "AGV2", "AGV3"]:
            self.cfg.agv_visual_cfg.func(
                f"/World/envs/env_0/{agv_name}/Visual",
                self.cfg.agv_visual_cfg,
                translation=tuple(float(v) for v in self.cfg.agv_visual_translation),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        # Board 的黄色可视子节点与物理刚体共用局部原点。
        self.cfg.payload_visual_cfg.func(
            "/World/envs/env_0/Payload/Visual",
            self.cfg.payload_visual_cfg,
            translation=tuple(float(v) for v in self.cfg.payload_visual_translation),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        self.scene.clone_environments(copy_from_source=False)
        self._configure_native_lift_visuals()
        self._configure_proxy_visual_visibility()

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_native_lift_visuals(self) -> None:
        """Cache the visual-only iwhub Lift transforms for runtime articulation.

        The source USD contains a single visible Lift mesh and a sibling
        ``/lift/Collision`` prim.  Only ``/lift/Lift`` is moved here, so this
        changes rendering without altering the AGV or hidden Lift collision.
        """
        self._native_lift_visual_ops = {}
        # Skip thousands of USD edits during large headless training jobs, but
        # keep small headless validation scenes inspectable.
        if self.num_envs > 8 and not (self.sim.has_gui() or self.sim.has_rtx_sensors()):
            return

        stage = sim_utils.get_current_stage()
        for env_id in range(self.num_envs):
            for lift_index, agv_name in enumerate(("AGV1", "AGV2", "AGV3")):
                path = f"/World/envs/env_{env_id}/{agv_name}/Visual/lift/Lift"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f"Missing native AGV Lift visual prim: {path}")
                for descendant in Usd.PrimRange(prim):
                    if any(
                        "CollisionAPI" in schema or "RigidBodyAPI" in schema
                        for schema in descendant.GetAppliedSchemas()
                    ):
                        raise RuntimeError(
                            f"Native AGV Lift branch is not visual-only: {descendant.GetPath()}"
                        )
                xformable = UsdGeom.Xformable(prim)
                transform_ops = [
                    op
                    for op in xformable.GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeTransform
                ]
                if len(transform_ops) != 1:
                    raise RuntimeError(f"Expected one native-Lift transform op at {path}")
                op = transform_ops[0]
                base_matrix = Gf.Matrix4d(op.Get())
                self._native_lift_visual_ops[(env_id, lift_index)] = (
                    prim,
                    op,
                    base_matrix,
                    base_matrix.ExtractTranslation(),
                )

    def _native_lift_visual_offsets(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the native Lift translation needed to touch the Board underside."""
        if env_ids is None:
            env_ids = self.payload._ALL_INDICES
        source_top_from_root = (
            float(self.cfg.agv_native_lift_visual_top_z) - float(self.cfg.agv_center_z)
        )
        target_top_from_root = (
            0.5 * float(self.cfg.agv_size[2])
            + self.lift_height[env_ids]
            + float(self.cfg.lift_plate_size[2])
            + float(self.cfg.board_support_clearance)
        )
        return target_top_from_root - source_top_from_root

    def _update_native_lift_visuals(self, env_ids: torch.Tensor | None = None) -> None:
        """Drive each iwhub Lift mesh from its corresponding ideal actuator height."""
        if not getattr(self, "_native_lift_visual_ops", None):
            return
        if env_ids is None:
            env_ids = self.payload._ALL_INDICES
        selected_envs = [int(v) for v in env_ids.detach().cpu().tolist()]
        offsets = self._native_lift_visual_offsets(env_ids).detach().cpu()
        scale_z = float(self.cfg.agv_visual_cfg.scale[2])
        if scale_z <= 0.0:
            raise ValueError("AGV visual Z scale must be positive")

        for row, env_id in enumerate(selected_envs):
            for lift_index in range(3):
                _, op, base_matrix, base_translation = self._native_lift_visual_ops[
                    (env_id, lift_index)
                ]
                matrix = Gf.Matrix4d(base_matrix)
                matrix.SetTranslateOnly(
                    Gf.Vec3d(
                        float(base_translation[0]),
                        float(base_translation[1]),
                        float(base_translation[2]) + float(offsets[row, lift_index]) / scale_z,
                    )
                )
                op.Set(matrix)

    def _set_native_lift_visual_visibility(
        self, visible: bool, env_ids: torch.Tensor | None = None
    ) -> None:
        """Show the native Lift for B/C/D or hide it for direct-support Case A."""
        if not getattr(self, "_native_lift_visual_ops", None):
            return
        if env_ids is None:
            env_ids = self.payload._ALL_INDICES
        for env_id in (int(v) for v in env_ids.detach().cpu().tolist()):
            for lift_index in range(3):
                prim = self._native_lift_visual_ops[(env_id, lift_index)][0]
                imageable = UsdGeom.Imageable(prim)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

    def _move_ground_plane_for_visual_terrain(self) -> None:
        """将默认 ground plane 下移，避免遮挡可视化崎岖地形 mesh。

        V6.2 的 terrain mesh 主要用于视觉展示；AGV 的 z 高度仍由 _terrain_height()
        解析函数决定。默认无限 ground plane 如果保持在 z=0，会遮住负高度谷底，导致
        mesh 看起来仍像平地。因此只在 enable_visual_terrain_mesh=True 时将 ground plane
        稍微下移，作为掉落后的兜底平面。
        """
        if not bool(getattr(self.cfg, "enable_visual_terrain_mesh", False)):
            return
        stage = sim_utils.get_current_stage()
        ground_prim = stage.GetPrimAtPath("/World/ground")
        if not ground_prim.IsValid():
            return
        xformable = UsdGeom.Xformable(ground_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(self.cfg.visual_terrain_ground_z)))

    def _spawn_visual_bumpy_terrain(self) -> None:
        """在 env_0 下生成可见崎岖地形 mesh，并随 scene.clone_environments 复制到各环境。

        重要说明：当前 mesh 主要用于视觉展示，不作为真实轮地碰撞来源。训练中的 AGV
        高度仍使用 _terrain_height(local_xy) 计算。为了避免“视觉路面”和“训练高度”不一致，
        mesh 顶点 z 使用 _terrain_height_scalar() 生成。
        """
        if not bool(getattr(self.cfg, "enable_visual_terrain_mesh", False)):
            return

        stage = sim_utils.get_current_stage()
        prim_path = "/World/envs/env_0/BumpyTerrainVisual"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)

        nx = max(int(self.cfg.visual_terrain_grid_x), 2)
        ny = max(int(self.cfg.visual_terrain_grid_y), 2)
        x_min = float(self.cfg.visual_terrain_x_min)
        x_max = float(self.cfg.visual_terrain_x_max)
        y_min = float(self.cfg.visual_terrain_y_min)
        y_max = float(self.cfg.visual_terrain_y_max)
        z_offset = float(self.cfg.visual_terrain_z_offset)
        z_scale = float(self.cfg.visual_terrain_height_scale)

        points = []
        for iy in range(ny):
            y = y_min + (y_max - y_min) * iy / float(ny - 1)
            for ix in range(nx):
                x = x_min + (x_max - x_min) * ix / float(nx - 1)
                z = z_offset + z_scale * self._terrain_height_scalar(x, y)
                points.append(Gf.Vec3f(float(x), float(y), float(z)))

        face_vertex_counts = []
        face_vertex_indices = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                v00 = iy * nx + ix
                v10 = iy * nx + ix + 1
                v01 = (iy + 1) * nx + ix
                v11 = (iy + 1) * nx + ix + 1
                # 每个网格拆成两个三角面。
                face_vertex_counts.extend([3, 3])
                face_vertex_indices.extend([v00, v10, v11, v00, v11, v01])

        mesh.CreatePointsAttr(Vt.Vec3fArray(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_vertex_counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_vertex_indices))
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(*tuple(float(c) for c in self.cfg.visual_terrain_color))])
        )

    def _configure_proxy_visual_visibility(self) -> None:
        """隐藏物理代理原始几何，只保留独立的 AGV/payload 外观。

        该函数只改变 USD 可见性，不改变碰撞、质量、刚体位姿、观测或奖励。
        调试时可通过 cfg 中的开关显示对应物理代理。
        """
        stage = sim_utils.get_current_stage()
        show_agv_proxy = bool(getattr(self.cfg, "debug_show_agv_collision_proxies", False))
        show_payload_proxy = bool(getattr(self.cfg, "debug_show_payload_collision_proxy", False))
        show_lift_proxy = bool(getattr(self.cfg, "debug_show_lift_collision_proxies", False))

        for env_id in range(self.num_envs):
            if not show_agv_proxy:
                for agv_name in ["AGV1", "AGV2", "AGV3"]:
                    agv_root_path = f"/World/envs/env_{env_id}/{agv_name}"
                    agv_root = stage.GetPrimAtPath(agv_root_path)
                    if not agv_root.IsValid():
                        continue
                    for prim in Usd.PrimRange(agv_root):
                        prim_path = prim.GetPath().pathString
                        if prim_path == agv_root_path or "/Visual" in prim_path:
                            continue
                        imageable = UsdGeom.Imageable(prim)
                        if imageable:
                            imageable.MakeInvisible()

            if not show_payload_proxy:
                payload_root_path = f"/World/envs/env_{env_id}/Payload"
                payload_root = stage.GetPrimAtPath(payload_root_path)
                if not payload_root.IsValid():
                    continue
                for prim in Usd.PrimRange(payload_root):
                    prim_path = prim.GetPath().pathString
                    if prim_path == payload_root_path or "/Visual" in prim_path:
                        continue
                    imageable = UsdGeom.Imageable(prim)
                    if imageable:
                        imageable.MakeInvisible()

            if not show_lift_proxy:
                for lift_name in ["Lift1", "Lift2", "Lift3"]:
                    lift_root_path = f"/World/envs/env_{env_id}/{lift_name}"
                    lift_root = stage.GetPrimAtPath(lift_root_path)
                    if not lift_root.IsValid():
                        continue
                    for prim in Usd.PrimRange(lift_root):
                        if prim.GetPath().pathString == lift_root_path:
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
        # DirectRLEnv calls _apply_action() once per physics substep.  Use the
        # physics dt here; multiplying by decimation would apply the configured
        # AGV and Lift velocities ``decimation`` times too fast.
        dt = self.cfg.sim.dt
        env_xy = self.scene.env_origins[:, :2]

        # 平地直线驮运阶段采用后侧线速度共享先验：
        # AGV2/AGV3 的纵向命令先取平均，再用于运动学写入。
        # 这样直接消除后侧两车一快一慢的失效模式；raw_cmds 仍保留给 reward，
        # 用于鼓励策略本身输出更一致的后侧速度命令。
        raw_linear_cmds, applied_linear_cmds = self._compute_linear_cmds()
        applied_angular_speeds = self._compute_angular_speeds()

        for i, agv in enumerate(self.agvs):
            agv_state = agv.data.root_state_w.clone()

            linear_cmd = applied_linear_cmds[:, i]
            linear_speed = linear_cmd * self.cfg.max_agv_linear_speed

            # 角速度由策略命令与 V6.2.6 后侧横向位置闭环共同决定。
            # AGV1 保持原策略角速度；AGV2/AGV3 在横向偏差超限时获得回正修正。
            angular_speed = applied_angular_speeds[:, i]

            self.agv_yaw[:, i] = self._wrap_to_pi(self.agv_yaw[:, i] + angular_speed * dt)
            heading = torch.stack((torch.cos(self.agv_yaw[:, i]), torch.sin(self.agv_yaw[:, i])), dim=1)
            new_xy = agv_state[:, :2] + linear_speed.unsqueeze(-1) * heading * dt

            rel_xy = torch.clamp(new_xy - env_xy, -self.cfg.workspace_limit, self.cfg.workspace_limit)
            new_xy = env_xy + rel_xy
            terrain_center, terrain_roll, terrain_pitch, terrain_samples = self._sample_agv_terrain_plane(
                new_xy, self.agv_yaw[:, i]
            )
            self.agv_terrain_roll[:, i] = terrain_roll
            self.agv_terrain_pitch[:, i] = terrain_pitch
            self.agv_terrain_samples[:, i, :] = terrain_samples
            quat = self._rpy_to_quat(terrain_roll, terrain_pitch, self.agv_yaw[:, i])
            local_z = self._quat_rotate_z(quat)
            z = (
                terrain_center
                + self.base_z_disturbance[:, i]
                + 0.5 * float(self.cfg.agv_size[2]) * local_z[:, 2]
            )
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

        # V7.0-B: independent ideal lift actuators track bounded targets at a
        # finite speed.  This update runs once per physics substep.
        lift_target = torch.clamp(
            self.lift_target_height,
            min=float(self.cfg.lift_min_height),
            max=float(self.cfg.lift_max_height),
        )
        lift_error = lift_target - self.lift_height
        self.lift_velocity[:] = torch.clamp(
            float(self.cfg.lift_position_kp) * lift_error,
            min=-float(self.cfg.max_lift_speed),
            max=float(self.cfg.max_lift_speed),
        )
        next_lift_height = self.lift_height + self.lift_velocity * dt
        reached_target = (lift_error * (lift_target - next_lift_height)) <= 0.0
        self.lift_height[:] = torch.where(reached_target, lift_target, next_lift_height)
        self.lift_height.clamp_(
            min=float(self.cfg.lift_min_height), max=float(self.cfg.lift_max_height)
        )
        self.lift_velocity[:] = torch.where(
            reached_target, torch.zeros_like(self.lift_velocity), self.lift_velocity
        )
        self._update_lift_poses()

        if bool(getattr(self.cfg, "enable_virtual_friction_carry", False)):
            self._apply_virtual_friction_carry(dt)

    def _update_lift_poses(self, env_ids: torch.Tensor | None = None) -> None:
        """Place each Lift along its AGV local +Z axis with the full AGV attitude."""
        if env_ids is None:
            env_ids = self.payload._ALL_INDICES

        for i, (agv, lift) in enumerate(zip(self.agvs, self.lifts)):
            agv_state = agv.data.root_state_w[env_ids]
            lift_pose = torch.zeros((len(env_ids), 7), device=self.device)
            local_z = self._quat_rotate_z(agv_state[:, 3:7])
            axis_offset = (
                0.5 * float(self.cfg.agv_size[2])
                + self.lift_height[env_ids, i]
                + 0.5 * float(self.cfg.lift_plate_size[2])
            )
            lift_pose[:, 0:3] = agv_state[:, 0:3] + local_z * axis_offset.unsqueeze(-1)
            lift_pose[:, 3:7] = agv_state[:, 3:7]

            lift_velocity = torch.zeros((len(env_ids), 6), device=self.device)
            lift_velocity[:, 0:2] = agv_state[:, 7:9]
            lift_velocity[:, 0:3] += local_z * self.lift_velocity[env_ids, i].unsqueeze(-1)
            lift_velocity[:, 3:6] = agv_state[:, 10:13]
            lift.write_root_pose_to_sim(lift_pose, env_ids=env_ids)
            lift.write_root_velocity_to_sim(lift_velocity, env_ids=env_ids)
        self._update_native_lift_visuals(env_ids)

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
        carry_valid = contact_count >= float(self.cfg.virtual_friction_min_contacts)
        support_margin = self._compute_support_polygon_margin(payload_xy)
        stabilization_valid = (
            contact_count >= float(self.cfg.virtual_stabilization_min_contacts)
        ) & (support_margin > float(self.cfg.virtual_stabilization_support_margin))
        self.last_virtual_carry_active[:] = carry_valid.detach()
        self.last_virtual_stabilization_active[:] = stabilization_valid.detach()
        if not torch.any(carry_valid):
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
        payload_state[:, 7:9] = torch.where(carry_valid.unsqueeze(-1), new_vxy, current_vxy)

        # Artificial vertical/attitude damping is stricter than planar carry.
        # With only two supports, or once the CoM leaves the support polygon,
        # gravity and PhysX contacts must be free to tip/drop the Board.
        vz = payload_state[:, 9]
        roll_w = payload_state[:, 10]
        pitch_w = payload_state[:, 11]
        payload_state[:, 9] = torch.where(
            stabilization_valid, vz * (1.0 - float(self.cfg.payload_vertical_damping)), vz
        )
        payload_state[:, 10] = torch.where(
            stabilization_valid,
            roll_w * (1.0 - float(self.cfg.payload_roll_pitch_damping)),
            roll_w,
        )
        payload_state[:, 11] = torch.where(
            stabilization_valid,
            pitch_w * (1.0 - float(self.cfg.payload_roll_pitch_damping)),
            pitch_w,
        )

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
        payload_state[:, 12] = torch.where(stabilization_valid, new_wz, current_wz)

        self.payload.write_root_velocity_to_sim(payload_state[:, 7:])

    def _compute_angular_speeds(self) -> torch.Tensor:
        """返回三台 AGV 实际执行的角速度（rad/s）。

        V6.2.5 已解决 AGV1 纵向抢跑，但诊断显示接近终点时 AGV2 的横向误差
        从 -0.157 m 持续扩大到 -0.236 m，并因 XY 误差失去 contact。这里在不改变
        reward、contact margin、视觉和物理代理的前提下，对 AGV2/AGV3 增加执行层
        横向位置闭环。

        lateral_error 使用与 contact 诊断完全相同的 payload 局部横向坐标：
        - error > 0：AGV 位于支撑目标左侧，需要向右转（负角速度）；
        - error < 0：AGV 位于支撑目标右侧，需要向左转（正角速度）。
        因而统一使用 correction = -Kp * error，不需要为 AGV2/AGV3 人工反转符号。
        """
        max_angular = float(self.cfg.max_agv_angular_speed)
        side_scale = float(getattr(self.cfg, "side_agv_angular_scale", 1.0))

        applied = torch.empty((self.num_envs, 3), device=self.device, dtype=self.actions.dtype)
        applied[:, 0] = self.actions[:, 1] * max_angular
        applied[:, 1] = self.actions[:, 3] * max_angular * side_scale
        applied[:, 2] = self.actions[:, 5] * max_angular * side_scale

        rear_errors = torch.zeros((self.num_envs, 2), device=self.device, dtype=self.actions.dtype)
        corrections = torch.zeros_like(rear_errors)

        if bool(getattr(self.cfg, "enable_rear_lateral_guard", False)):
            payload_xy = self.payload.data.root_pos_w[:, :2]
            target_xy = self._get_target_xy()
            move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)
            support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
            agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in self.agvs], dim=1)
            support_error = agv_xy - support_targets
            lateral_error = torch.sum(support_error * lateral_dir.unsqueeze(1), dim=2)
            rear_errors = lateral_error[:, 1:3]

            deadband = max(0.0, float(getattr(self.cfg, "rear_lateral_guard_deadband", 0.06)))
            kp = max(0.0, float(getattr(self.cfg, "rear_lateral_guard_kp", 1.6)))
            max_correction = max(0.0, float(getattr(self.cfg, "rear_lateral_guard_max_correction", 0.24)))
            hard_limit = max(deadband, float(getattr(self.cfg, "rear_lateral_guard_hard_limit", 0.14)))
            hard_min_speed = max(0.0, float(getattr(self.cfg, "rear_lateral_guard_hard_min_angular_speed", 0.22)))
            max_applied = max(1.0e-6, float(getattr(self.cfg, "rear_lateral_guard_max_applied_angular_speed", 0.32)))

            abs_error = torch.abs(rear_errors)
            excess = torch.clamp(abs_error - deadband, min=0.0)
            correction_mag = torch.clamp(kp * excess, min=0.0, max=max_correction)
            correction_dir = -torch.sign(rear_errors)
            corrections = correction_dir * correction_mag

            rear_applied = applied[:, 1:3] + corrections

            # 进入硬偏差区后，不能让策略给出的反向角速度抵消回正动作。
            # 保证最终角速度沿回正方向至少达到 hard_min_speed，直到误差退出硬区。
            hard_mask = abs_error > hard_limit
            corrective_component = rear_applied * correction_dir
            hard_rear_applied = torch.where(
                corrective_component < hard_min_speed,
                correction_dir * hard_min_speed,
                rear_applied,
            )
            rear_applied = torch.where(hard_mask, hard_rear_applied, rear_applied)
            rear_applied = torch.clamp(rear_applied, min=-max_applied, max=max_applied)
            applied[:, 1:3] = rear_applied

        self.last_rear_lateral_error[:] = rear_errors.detach()
        self.last_rear_lateral_correction[:] = corrections.detach()
        self.last_applied_angular_speed[:] = applied.detach()
        return applied

    def _compute_linear_cmds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 raw/appplied 线速度命令，范围约为 [0, 1]。

        raw_linear_cmds 是策略直接输出动作映射后的命令；applied_linear_cmds 是实际写入
        kinematic AGV 的命令。当前阶段默认把 AGV2/AGV3 的线速度做共享，并限制 AGV1 抢跑，
        以消除后侧两车速率不一致造成的拖拽/脱离趋势。
        """
        linear_actions = self.actions[:, 0::2]
        if bool(getattr(self.cfg, "forward_only_linear_speed", True)):
            raw_linear_cmds = 0.5 * (linear_actions + 1.0)
        else:
            raw_linear_cmds = torch.clamp(linear_actions, min=0.0, max=1.0)

        applied_linear_cmds = raw_linear_cmds.clone()
        if bool(getattr(self.cfg, "tie_rear_linear_speed", False)):
            blend = float(getattr(self.cfg, "tie_rear_linear_speed_blend", 1.0))
            blend = max(0.0, min(1.0, blend))
            rear_mean = 0.5 * (raw_linear_cmds[:, 1] + raw_linear_cmds[:, 2])
            applied_linear_cmds[:, 1] = (1.0 - blend) * raw_linear_cmds[:, 1] + blend * rear_mean
            applied_linear_cmds[:, 2] = (1.0 - blend) * raw_linear_cmds[:, 2] + blend * rear_mean

        # 第一层：固定命令领先上限。它只能限制瞬时速度差，无法阻止小速度差长期积分成
        # 位置误差，因此 V6.2.4 中 AGV1 仍会在约 ep_step=293 时越过 XY contact margin。
        rear_applied_mean = 0.5 * (applied_linear_cmds[:, 1] + applied_linear_cmds[:, 2])
        max_front_lead = float(getattr(self.cfg, "max_front_command_lead", 0.08))
        max_front_lead = max(0.0, min(1.0, max_front_lead))
        if bool(getattr(self.cfg, "limit_front_linear_speed", False)):
            front_cmd_ceiling = torch.clamp(rear_applied_mean + max_front_lead, min=0.0, max=1.0)
            applied_linear_cmds[:, 0] = torch.minimum(applied_linear_cmds[:, 0], front_cmd_ceiling)

        # V6.2.5 第二层：基于 AGV1 相对自身支撑目标的纵向位置误差做闭环限速。
        # front_long_error > 0 表示 AGV1 已跑到支撑目标前方。误差超过 deadband 后，
        # 按比例降低 AGV1 命令；接近 contact 边界时让 AGV1 明确慢于后车，使后侧支撑
        # 与 payload 有机会追上。该约束不改变 AGV2/AGV3，也不修改视觉或物理几何。
        if bool(getattr(self.cfg, "enable_front_position_guard", False)):
            front_long_error = self._compute_front_longitudinal_error()
            deadband = float(getattr(self.cfg, "front_position_deadband", 0.17))
            kp = float(getattr(self.cfg, "front_position_kp", 2.0))
            max_slowdown = float(getattr(self.cfg, "front_position_max_slowdown", 0.30))
            hard_limit = float(getattr(self.cfg, "front_position_hard_limit", 0.22))
            recovery_margin = float(getattr(self.cfg, "front_position_recovery_margin", 0.10))

            position_excess = torch.clamp(front_long_error - deadband, min=0.0)
            slowdown = torch.clamp(kp * position_excess, min=0.0, max=max_slowdown)
            position_ceiling = rear_applied_mean + max_front_lead - slowdown

            # 进入硬恢复区时，AGV1 至少比后车平均命令低 recovery_margin，
            # 避免误差停留在 contact 阈值外侧而无法恢复。
            hard_recovery_ceiling = rear_applied_mean - recovery_margin
            position_ceiling = torch.where(
                front_long_error > hard_limit,
                torch.minimum(position_ceiling, hard_recovery_ceiling),
                position_ceiling,
            )
            position_ceiling = torch.clamp(position_ceiling, min=0.0, max=1.0)
            applied_linear_cmds[:, 0] = torch.minimum(applied_linear_cmds[:, 0], position_ceiling)

        return raw_linear_cmds, applied_linear_cmds

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
        """V6.2.1 可视化崎岖支撑面三点支撑稳定化奖励。

        上一版把 payload 速度奖励硬性绑定到三车完全有效支撑，训练早期几乎拿不到
        正反馈；同时未限幅的 support polygon dense penalty 可能在三车退化/共线时
        产生极大负奖励。本版改成：
        - 运输过程使用 soft transport_quality，而不是 0/1 硬门控；
        - dense reward 中暂时不使用 support polygon 惩罚；
        - support polygon 仍保留在 success 条件和 debug 指标中；
        - formation、slip、动作惩罚均使用温和、限幅形式；
        - 额外给三车支撑结构朝目标方向的平均前进速度正反馈，避免原地低风险策略；
        - 对 support_drive_speed 过低、线速度命令过低增加小惩罚，直接打破停车局部最优；
        - episode 前期降低 formation 惩罚，避免一动就被队形误差压回原地；
        - 增加左右侧 AGV 横向外扩/横向速度惩罚，抑制侧车向外逃逸。
        """
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()
        goal_dist = torch.linalg.norm(target_xy - payload_xy, dim=1)

        # progress 是主驱动力，但做限幅，避免 reset 或偶发碰撞造成异常大回报。
        raw_progress = self.prev_goal_dist - goal_dist
        progress = torch.clamp(raw_progress, min=-0.05, max=0.05)
        self.prev_goal_dist[:] = goal_dist.detach()

        move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)
        support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
        formation_errors = self._compute_formation_errors(support_targets)
        contact_flags = self._compute_support_contact_flags(support_targets)
        contact_count = contact_flags.float().sum(dim=1)
        slip_error = self._compute_slip_error(payload_xy, move_dir, lateral_dir)
        support_margin = self._compute_support_polygon_margin(payload_xy)
        roll, pitch, _ = self._get_payload_rpy()

        # V6.1 rough-rect diagnostics: AGV 支撑面高度随位置变化后，必须显式观察
        # payload 垂向运动和 payload-bottom / AGV-top 的 z gap。该项用于轻度崎岖支撑面
        # 过渡阶段，避免只看 xy goal 而忽略承载高度不一致。
        payload_vertical_speed_abs = torch.abs(self.payload.data.root_lin_vel_w[:, 2])
        vertical_vel_violation = torch.clamp(
            payload_vertical_speed_abs - float(self.cfg.payload_vertical_velocity_threshold),
            min=0.0,
            max=0.50,
        )
        support_z_gaps = self._compute_support_z_gaps()
        support_z_gap_mean = torch.mean(support_z_gaps, dim=1)
        support_z_gap_max = torch.max(support_z_gaps, dim=1).values
        support_z_gap_violation = torch.clamp(
            support_z_gap_mean - float(self.cfg.support_z_gap_penalty_deadband),
            min=0.0,
            max=0.30,
        )

        required_contacts = float(self.cfg.required_support_contacts)

        # 软运输质量：不再要求三车完美支撑才给速度奖励，否则训练早期没有正反馈。
        # contact_quality: 三车全在支撑区为 1，两车约 0.67，一车约 0.33。
        # formation/slip 使用指数衰减，误差越小，payload 前进奖励越完整。
        contact_quality = torch.clamp(contact_count / required_contacts, min=0.0, max=1.0)
        formation_error = torch.mean(formation_errors, dim=1)
        formation_quality = torch.exp(-float(self.cfg.formation_quality_gain) * formation_error)
        slip_quality = torch.exp(-float(self.cfg.slip_quality_gain) * slip_error)
        transport_quality = torch.clamp(contact_quality * formation_quality * slip_quality, min=0.0, max=1.0)

        payload_vel_xy = self.payload.data.root_lin_vel_w[:, :2]
        payload_speed_to_goal = torch.sum(payload_vel_xy * move_dir, dim=1)
        positive_payload_speed = torch.clamp(payload_speed_to_goal, min=0.0, max=self.cfg.max_payload_planar_speed)

        roll_pitch_abs = torch.maximum(torch.abs(roll), torch.abs(pitch))
        roll_pitch_violation = torch.clamp(
            roll_pitch_abs - self.cfg.stable_roll_pitch_radius,
            min=0.0,
            max=0.50,
        )
        slip_violation = torch.clamp(
            slip_error - self.cfg.slip_penalty_threshold,
            min=0.0,
            max=0.50,
        )
        formation_violation = torch.clamp(
            formation_error - self.cfg.formation_error_deadband,
            min=0.0,
            max=0.50,
        )
        missing_support = torch.clamp(required_contacts - contact_count, min=0.0, max=required_contacts)

        # 直接奖励三车支撑结构朝目标方向运动。
        # 只奖励 payload 速度时，训练早期 payload 尚未被带动，策略容易坍缩到原地低风险动作；
        # 这项给 AGV 前进提供更早的正反馈。
        # 用 MIN 而非 MEAN：MEAN 时 PPO 只动 2/3 车就能拿 67% 奖励，
        # 导致"前方 AGV 不动、两侧 AGV 先动"的投机策略。MIN 要求三车都动才有奖励。
        agv_speed_to_goal = torch.sum(self.agv_planar_vel * move_dir.unsqueeze(1), dim=2)
        support_drive_speed = torch.clamp(torch.min(agv_speed_to_goal, dim=1).values, min=0.0, max=self.cfg.max_agv_linear_speed)

        # 左右侧 AGV 横向约束：只抑制“向外侧逃逸”，不强行压制正常纵向前进。
        # lateral_dir 指向 payload 左侧：AGV2 在左，lateral_error>0 表示继续向左外扩；
        # AGV3 在右，lateral_error<0 表示继续向右外扩。
        agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in self.agvs], dim=1)
        support_error_vec = agv_xy - support_targets
        longitudinal_error = torch.sum(support_error_vec * move_dir.unsqueeze(1), dim=2)
        lateral_error = torch.sum(support_error_vec * lateral_dir.unsqueeze(1), dim=2)

        # V6.2.5 前车位置闭环诊断。这里仅记录当前误差和理论限速量，
        # 实际 applied command 已在 _compute_linear_cmds() 中完成裁剪。
        front_long_error = longitudinal_error[:, 0]
        front_position_excess = torch.clamp(
            front_long_error - float(getattr(self.cfg, "front_position_deadband", 0.17)),
            min=0.0,
        )
        front_position_slowdown = torch.clamp(
            float(getattr(self.cfg, "front_position_kp", 2.0)) * front_position_excess,
            min=0.0,
            max=float(getattr(self.cfg, "front_position_max_slowdown", 0.30)),
        )
        side_deadband = float(self.cfg.side_lateral_deadband)
        agv2_outward = torch.clamp(lateral_error[:, 1] - side_deadband, min=0.0, max=0.50)
        agv3_outward = torch.clamp(-lateral_error[:, 2] - side_deadband, min=0.0, max=0.50)
        side_outward_cost = torch.square(agv2_outward) + torch.square(agv3_outward)

        agv_lateral_vel = torch.sum(self.agv_planar_vel * lateral_dir.unsqueeze(1), dim=2)
        side_lateral_velocity_cost = torch.square(agv_lateral_vel[:, 1]) + torch.square(agv_lateral_vel[:, 2])

        # AGV 车头朝向约束：抑制 v 很小、w 很大的原地转圈策略。
        agv_headings = torch.stack(
            [
                torch.stack((torch.cos(self.agv_yaw[:, i]), torch.sin(self.agv_yaw[:, i])), dim=1)
                for i in range(3)
            ],
            dim=1,
        )
        heading_dot = torch.sum(agv_headings * move_dir.unsqueeze(1), dim=2)
        agv_heading_cost = torch.mean(1.0 - torch.clamp(heading_dot, -1.0, 1.0), dim=1)

        # 后侧 AGV 同向偏航惩罚：side_outward_penalty 只能捕获 AGV2 向左外扩
        # 和 AGV3 向右外扩，但当两车同时左偏时 AGV3 的左偏是"内向"的，不被惩罚。
        # 此项直接惩罚后两车相对 move_dir 的偏航之和的平方：
        # 同向偏航 → 和值大 → 重罚；异向偏航（正常转弯）→ 和值≈0 → 不罚。
        move_yaw = torch.atan2(move_dir[:, 1], move_dir[:, 0])
        rear1_yaw_dev = self._wrap_to_pi(self.agv_yaw[:, 1] - move_yaw)
        rear2_yaw_dev = self._wrap_to_pi(self.agv_yaw[:, 2] - move_yaw)
        rear_yaw_sum = torch.clamp(rear1_yaw_dev + rear2_yaw_dev, min=-math.pi, max=math.pi)
        rear_heading_sync_cost = torch.square(rear_yaw_sum)

        linear_actions = self.actions[:, 0::2]
        angular_actions = self.actions[:, 1::2]
        raw_linear_cmds, applied_linear_cmds = self._compute_linear_cmds()

        # command-level 速度同步约束：使用 [0, 1] 量级的线速度命令，而不是 m/s 物理速度，
        # 避免 max_agv_linear_speed 较小时速度差平方过小，reward 几乎不起作用。
        rear_cmd_sync_cost = torch.square(raw_linear_cmds[:, 1] - raw_linear_cmds[:, 2])
        cmd_mean = torch.mean(raw_linear_cmds, dim=1, keepdim=True)
        all_cmd_sync_cost = torch.mean(torch.square(raw_linear_cmds - cmd_mean), dim=1)

        # 前车抢跑约束的策略侧 shaping：执行层已经限制 applied command，
        # 这里仅惩罚 raw command 超过允许领先量的部分，促使 PPO 自身学会同步输出。
        rear_raw_cmd_mean = 0.5 * (raw_linear_cmds[:, 1] + raw_linear_cmds[:, 2])
        front_cmd_lead_raw = raw_linear_cmds[:, 0] - rear_raw_cmd_mean
        allowed_front_lead = float(getattr(self.cfg, "max_front_command_lead", 0.08))
        front_cmd_lead_excess = torch.clamp(front_cmd_lead_raw - allowed_front_lead, min=0.0, max=1.0)
        front_cmd_lead_cost = torch.square(front_cmd_lead_excess)

        # per-AGV 低前进命令惩罚：用 applied_linear_cmds 检查实际执行命令。
        # 这样任何一台 AGV 停车（cmd≈0）都会触发惩罚，不被其他两车的高 cmd 掩盖。
        per_agv_low_cmd = torch.clamp(float(self.cfg.min_forward_cmd) - applied_linear_cmds, min=0.0)
        low_forward_cmd = torch.max(per_agv_low_cmd, dim=1).values

        # per-AGV 静止惩罚：任何一台 AGV 速度低于阈值即触发，
        # 防止 PPO 通过让 2/3 车运动来规避静止惩罚。
        min_agv_speed = torch.min(agv_speed_to_goal, dim=1).values
        stationary = (min_agv_speed < float(self.cfg.stationary_speed_threshold)) & (goal_dist > self.cfg.target_radius)

        linear_action_cost = torch.mean(torch.square(linear_actions), dim=1)
        angular_action_cost = torch.mean(torch.square(angular_actions), dim=1)
        action_rate_cost = torch.mean(torch.square(self.actions - self.prev_actions), dim=1)

        warmup = self.episode_length_buf < int(self.cfg.formation_warmup_steps)
        formation_penalty_weight = torch.where(
            warmup,
            torch.full_like(goal_dist, float(self.cfg.formation_error_penalty_scale) * float(self.cfg.formation_warmup_scale)),
            torch.full_like(goal_dist, float(self.cfg.formation_error_penalty_scale)),
        )

        dropped = self.payload.data.root_pos_w[:, 2] < self.cfg.payload_min_z
        tipped = roll_pitch_abs > self.cfg.tip_roll_pitch_threshold
        out_of_bounds = self._compute_out_of_bounds(payload_xy)
        support_lost = (
            (contact_count < float(self.cfg.critical_support_contacts))
            & (self.episode_length_buf > self.cfg.support_loss_grace_steps)
        )
        success = self._compute_success(goal_dist, roll, pitch, slip_error, contact_count, support_margin)

        # progress 不完全硬门控：即使 transport_quality 低，也保留 25% 的进展反馈，
        # 防止策略重新坍缩到原地不动；payload 速度奖励则按 transport_quality 连续缩放。
        progress_quality = 0.25 + 0.75 * transport_quality

        # 近终点 heading 松弛：payload 接近目标时 move_dir = (target-payload)/norm
        # 变得不稳定——微小位移导致方向剧变，heading penalty 强迫 AGV 急转。
        # goal_dist > 0.5 时满罚；< 0.15 时不罚；中间线性过渡。
        near_goal_heading_scale = torch.clamp((goal_dist - 0.15) / 0.35, min=0.0, max=1.0)

        reward = (
            self.cfg.progress_reward_scale * progress * progress_quality
            - self.cfg.distance_penalty_scale * goal_dist
            + self.cfg.payload_speed_reward_scale * positive_payload_speed * transport_quality
            + self.cfg.support_drive_reward_scale * support_drive_speed * (0.20 + 0.80 * transport_quality)
            - self.cfg.stationary_penalty_scale * stationary.float()
            - self.cfg.low_forward_cmd_penalty_scale * low_forward_cmd
            - self.cfg.rear_cmd_sync_penalty_scale * rear_cmd_sync_cost
            - self.cfg.all_cmd_sync_penalty_scale * all_cmd_sync_cost
            - self.cfg.front_command_lead_penalty_scale * front_cmd_lead_cost
            - self.cfg.missing_support_penalty_scale * missing_support
            - formation_penalty_weight * formation_violation
            - self.cfg.side_outward_penalty_scale * side_outward_cost
            - self.cfg.side_lateral_velocity_penalty_scale * side_lateral_velocity_cost
            - self.cfg.roll_pitch_penalty_scale * torch.square(roll_pitch_violation)
            - self.cfg.vertical_velocity_penalty_scale * torch.square(vertical_vel_violation)
            - self.cfg.support_z_gap_penalty_scale * torch.square(support_z_gap_violation)
            - self.cfg.slip_penalty_scale * slip_violation
            - self.cfg.linear_action_penalty_scale * linear_action_cost
            - self.cfg.angular_action_penalty_scale * angular_action_cost
            - self.cfg.agv_heading_penalty_scale * near_goal_heading_scale * agv_heading_cost
            - self.cfg.rear_heading_sync_penalty_scale * near_goal_heading_scale * rear_heading_sync_cost
            - self.cfg.action_rate_penalty_scale * action_rate_cost
            + self.cfg.success_reward_scale * success.float()
            - self.cfg.support_loss_penalty_scale * support_lost.float()
            - self.cfg.drop_penalty * dropped.float()
            - self.cfg.tip_penalty * tipped.float()
            - self.cfg.out_of_bounds_penalty * out_of_bounds.float()
        )

        # -------------------------------------------------------------
        # TensorBoard / evaluation metrics only. These logs do NOT change reward.
        # They are used to judge whether the policy is really stable instead of
        # judging the model only from the total reward curve.
        # -------------------------------------------------------------
        reward_progress = self.cfg.progress_reward_scale * progress * progress_quality
        reward_distance = -self.cfg.distance_penalty_scale * goal_dist
        reward_payload_speed = self.cfg.payload_speed_reward_scale * positive_payload_speed * transport_quality
        reward_support_drive = self.cfg.support_drive_reward_scale * support_drive_speed * (0.20 + 0.80 * transport_quality)
        penalty_stationary = -self.cfg.stationary_penalty_scale * stationary.float()
        penalty_low_forward_cmd = -self.cfg.low_forward_cmd_penalty_scale * low_forward_cmd
        penalty_rear_cmd_sync = -self.cfg.rear_cmd_sync_penalty_scale * rear_cmd_sync_cost
        penalty_all_cmd_sync = -self.cfg.all_cmd_sync_penalty_scale * all_cmd_sync_cost
        penalty_front_command_lead = -self.cfg.front_command_lead_penalty_scale * front_cmd_lead_cost
        penalty_missing_support = -self.cfg.missing_support_penalty_scale * missing_support
        penalty_formation = -formation_penalty_weight * formation_violation
        penalty_side_outward = -self.cfg.side_outward_penalty_scale * side_outward_cost
        penalty_side_lateral_velocity = -self.cfg.side_lateral_velocity_penalty_scale * side_lateral_velocity_cost
        penalty_roll_pitch = -self.cfg.roll_pitch_penalty_scale * torch.square(roll_pitch_violation)
        penalty_vertical_velocity = -self.cfg.vertical_velocity_penalty_scale * torch.square(vertical_vel_violation)
        penalty_support_z_gap = -self.cfg.support_z_gap_penalty_scale * torch.square(support_z_gap_violation)
        penalty_slip = -self.cfg.slip_penalty_scale * slip_violation
        penalty_angular_action = -self.cfg.angular_action_penalty_scale * angular_action_cost
        penalty_heading = -self.cfg.agv_heading_penalty_scale * near_goal_heading_scale * agv_heading_cost
        penalty_rear_heading_sync = -self.cfg.rear_heading_sync_penalty_scale * near_goal_heading_scale * rear_heading_sync_cost
        penalty_action_rate = -self.cfg.action_rate_penalty_scale * action_rate_cost
        reward_success = self.cfg.success_reward_scale * success.float()
        penalty_support_lost = -self.cfg.support_loss_penalty_scale * support_lost.float()
        penalty_drop = -self.cfg.drop_penalty * dropped.float()
        penalty_tip = -self.cfg.tip_penalty * tipped.float()
        penalty_oob = -self.cfg.out_of_bounds_penalty * out_of_bounds.float()

        rear_speed_diff = torch.abs(agv_speed_to_goal[:, 1] - agv_speed_to_goal[:, 2])
        rear_cmd_diff = torch.abs(raw_linear_cmds[:, 1] - raw_linear_cmds[:, 2])
        applied_rear_cmd_diff = torch.abs(applied_linear_cmds[:, 1] - applied_linear_cmds[:, 2])
        rear_speed_mean = 0.5 * (agv_speed_to_goal[:, 1] + agv_speed_to_goal[:, 2])
        front_speed_lead = agv_speed_to_goal[:, 0] - rear_speed_mean
        rear_applied_cmd_mean = 0.5 * (applied_linear_cmds[:, 1] + applied_linear_cmds[:, 2])
        front_cmd_lead_applied = applied_linear_cmds[:, 0] - rear_applied_cmd_mean
        agv_forward_speed_mean = torch.mean(agv_speed_to_goal, dim=1)
        agv_forward_speed_std = torch.std(agv_speed_to_goal, dim=1, unbiased=False)
        min_contact = torch.min(contact_count)

        env_xy = self.scene.env_origins[:, :2]
        agv_local_xy = agv_xy - env_xy.unsqueeze(1)
        agv_terrain_heights = self._terrain_height(agv_local_xy.reshape(-1, 2)).reshape(self.num_envs, 3)
        payload_terrain_height = self._terrain_height(payload_xy - env_xy)
        agv_height_std = torch.std(agv_terrain_heights, dim=1, unbiased=False)
        agv_height_span = torch.max(agv_terrain_heights, dim=1).values - torch.min(agv_terrain_heights, dim=1).values

        self.extras["log"] = {
            # Main task metrics
            "Carry/success_rate": success.float().mean().detach(),
            "Carry/goal_dist_mean": goal_dist.mean().detach(),
            "Carry/payload_speed_to_goal_mean": payload_speed_to_goal.mean().detach(),
            "Carry/positive_payload_speed_mean": positive_payload_speed.mean().detach(),
            "Carry/support_drive_speed_mean": support_drive_speed.mean().detach(),
            "Carry/transport_quality_mean": transport_quality.mean().detach(),
            "Carry/contact_count_mean": contact_count.mean().detach(),
            "Carry/contact_count_min": min_contact.detach(),
            "Support/agv1_contact_rate": contact_flags[:, 0].float().mean().detach(),
            "Support/agv2_contact_rate": contact_flags[:, 1].float().mean().detach(),
            "Support/agv3_contact_rate": contact_flags[:, 2].float().mean().detach(),
            "Support/agv1_formation_error_mean": formation_errors[:, 0].mean().detach(),
            "Support/agv2_formation_error_mean": formation_errors[:, 1].mean().detach(),
            "Support/agv3_formation_error_mean": formation_errors[:, 2].mean().detach(),
            "Carry/slip_error_mean": slip_error.mean().detach(),
            "Carry/support_margin_mean": support_margin.mean().detach(),
            "Carry/roll_abs_mean": torch.abs(roll).mean().detach(),
            "Carry/pitch_abs_mean": torch.abs(pitch).mean().detach(),
            "Carry/payload_vertical_speed_abs_mean": payload_vertical_speed_abs.mean().detach(),
            # Terrain / rough-support diagnostics
            "Terrain/agv_height_mean": agv_terrain_heights.mean().detach(),
            "Terrain/agv_height_std_mean": agv_height_std.mean().detach(),
            "Terrain/agv_height_span_mean": agv_height_span.mean().detach(),
            "Terrain/payload_height_ref_mean": payload_terrain_height.mean().detach(),
            "Support/z_gap_mean": support_z_gap_mean.mean().detach(),
            "Support/z_gap_max": support_z_gap_max.max().detach(),
            # Coordination metrics
            "Coord/rear_speed_diff_mean": rear_speed_diff.mean().detach(),
            "Coord/rear_speed_diff_max": rear_speed_diff.max().detach(),
            "Coord/rear_cmd_diff_mean_raw": rear_cmd_diff.mean().detach(),
            "Coord/rear_cmd_diff_mean_applied": applied_rear_cmd_diff.mean().detach(),
            "Coord/front_speed_lead_mean": front_speed_lead.mean().detach(),
            "Coord/front_speed_lead_positive_mean": torch.clamp(front_speed_lead, min=0.0).mean().detach(),
            "Coord/front_cmd_lead_raw_mean": front_cmd_lead_raw.mean().detach(),
            "Coord/front_cmd_lead_applied_mean": front_cmd_lead_applied.mean().detach(),
            "Coord/front_longitudinal_error_mean": front_long_error.mean().detach(),
            "Coord/front_longitudinal_error_max": front_long_error.max().detach(),
            "Coord/front_position_guard_excess_mean": front_position_excess.mean().detach(),
            "Coord/front_position_guard_slowdown_mean": front_position_slowdown.mean().detach(),
            "Coord/agv2_lateral_error_mean": lateral_error[:, 1].mean().detach(),
            "Coord/agv3_lateral_error_mean": lateral_error[:, 2].mean().detach(),
            "Coord/agv2_lateral_error_abs_mean": torch.abs(lateral_error[:, 1]).mean().detach(),
            "Coord/agv3_lateral_error_abs_mean": torch.abs(lateral_error[:, 2]).mean().detach(),
            "Coord/agv2_lateral_guard_correction_mean": self.last_rear_lateral_correction[:, 0].mean().detach(),
            "Coord/agv3_lateral_guard_correction_mean": self.last_rear_lateral_correction[:, 1].mean().detach(),
            "Coord/agv2_applied_angular_speed_mean": self.last_applied_angular_speed[:, 1].mean().detach(),
            "Coord/agv3_applied_angular_speed_mean": self.last_applied_angular_speed[:, 2].mean().detach(),
            "Coord/all_speed_std_mean": agv_forward_speed_std.mean().detach(),
            "Coord/agv_forward_speed_mean": agv_forward_speed_mean.mean().detach(),
            "Coord/agv1_speed_to_goal_mean": agv_speed_to_goal[:, 0].mean().detach(),
            "Coord/agv2_speed_to_goal_mean": agv_speed_to_goal[:, 1].mean().detach(),
            "Coord/agv3_speed_to_goal_mean": agv_speed_to_goal[:, 2].mean().detach(),
            "Coord/side_outward_cost_mean": side_outward_cost.mean().detach(),
            "Coord/side_lateral_velocity_cost_mean": side_lateral_velocity_cost.mean().detach(),
            "Coord/rear_heading_sync_cost_mean": rear_heading_sync_cost.mean().detach(),
            "Coord/agv_heading_cost_mean": agv_heading_cost.mean().detach(),
            # Done/failure diagnostics
            "Done/support_lost_rate": support_lost.float().mean().detach(),
            "Done/drop_rate": dropped.float().mean().detach(),
            "Done/tip_rate": tipped.float().mean().detach(),
            "Done/out_of_bounds_rate": out_of_bounds.float().mean().detach(),
            # Reward components
            "RewardTerms/progress": reward_progress.mean().detach(),
            "RewardTerms/distance": reward_distance.mean().detach(),
            "RewardTerms/payload_speed": reward_payload_speed.mean().detach(),
            "RewardTerms/support_drive": reward_support_drive.mean().detach(),
            "RewardTerms/stationary": penalty_stationary.mean().detach(),
            "RewardTerms/low_forward_cmd": penalty_low_forward_cmd.mean().detach(),
            "RewardTerms/rear_cmd_sync": penalty_rear_cmd_sync.mean().detach(),
            "RewardTerms/all_cmd_sync": penalty_all_cmd_sync.mean().detach(),
            "RewardTerms/front_command_lead": penalty_front_command_lead.mean().detach(),
            "RewardTerms/missing_support": penalty_missing_support.mean().detach(),
            "RewardTerms/formation": penalty_formation.mean().detach(),
            "RewardTerms/side_outward": penalty_side_outward.mean().detach(),
            "RewardTerms/side_lateral_velocity": penalty_side_lateral_velocity.mean().detach(),
            "RewardTerms/roll_pitch": penalty_roll_pitch.mean().detach(),
            "RewardTerms/vertical_velocity": penalty_vertical_velocity.mean().detach(),
            "RewardTerms/support_z_gap": penalty_support_z_gap.mean().detach(),
            "RewardTerms/slip": penalty_slip.mean().detach(),
            "RewardTerms/angular_action": penalty_angular_action.mean().detach(),
            "RewardTerms/heading": penalty_heading.mean().detach(),
            "RewardTerms/rear_heading_sync": penalty_rear_heading_sync.mean().detach(),
            "RewardTerms/action_rate": penalty_action_rate.mean().detach(),
            "RewardTerms/success": reward_success.mean().detach(),
            "RewardTerms/support_lost": penalty_support_lost.mean().detach(),
            "RewardTerms/drop": penalty_drop.mean().detach(),
            "RewardTerms/tip": penalty_tip.mean().detach(),
            "RewardTerms/out_of_bounds": penalty_oob.mean().detach(),
        }

        self.last_success[:] = success.detach()
        self.last_payload_dropped[:] = dropped.detach()
        self.last_payload_tipped[:] = tipped.detach()
        self.last_support_lost[:] = support_lost.detach()
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
        roll, pitch, _ = self._get_payload_rpy()

        success = self._compute_success(goal_dist, roll, pitch, slip_error, contact_count, support_margin)
        dropped = self.payload.data.root_pos_w[:, 2] < self.cfg.payload_min_z
        tipped = torch.maximum(torch.abs(roll), torch.abs(pitch)) > self.cfg.tip_roll_pitch_threshold
        out_of_bounds = self._compute_out_of_bounds(payload_xy)
        support_lost = (
            (contact_count < float(self.cfg.critical_support_contacts))
            & (self.episode_length_buf > self.cfg.support_loss_grace_steps)
        )

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
        self.base_z_disturbance[env_ids] = 0.0
        self.lift_target_height[env_ids] = float(self.cfg.lift_neutral_height)
        self.lift_height[env_ids] = float(self.cfg.lift_neutral_height)
        self.lift_velocity[env_ids] = 0.0
        self.agv_terrain_roll[env_ids] = 0.0
        self.agv_terrain_pitch[env_ids] = 0.0
        self.agv_terrain_samples[env_ids] = 0.0

        # Payload 初始状态。
        payload_init_pos = torch.tensor(self.cfg.payload_init_pos, device=self.device, dtype=torch.float32)
        payload_pose = torch.zeros((num_reset, 7), device=self.device)
        payload_pose[:, :3] = env_origins + payload_init_pos
        payload_pose[:, 3] = 1.0
        payload_vel = torch.zeros((num_reset, 6), device=self.device)
        self.payload.write_root_pose_to_sim(payload_pose, env_ids=env_ids)
        self.payload.write_root_velocity_to_sim(payload_vel, env_ids=env_ids)

        # Optional V7.2 Cargo is a free dynamic rigid body.  Reset only sets its
        # pose/velocity; gravity, collision and friction govern all later motion.
        if self.cargo is not None:
            cargo_init_pos = torch.tensor(self.cfg.cargo_init_pos, device=self.device, dtype=torch.float32)
            cargo_pose = torch.zeros((num_reset, 7), device=self.device)
            cargo_pose[:, :3] = env_origins + cargo_init_pos
            cargo_pose[:, 3] = 1.0
            cargo_vel = torch.zeros((num_reset, 6), device=self.device)
            self.cargo.write_root_pose_to_sim(cargo_pose, env_ids=env_ids)
            self.cargo.write_root_velocity_to_sim(cargo_vel, env_ids=env_ids)

        # 根据 payload->target 方向初始化三台 AGV 的支撑三角形与车头朝向。
        # 这样即使后续把 target 改到别的方向，也不会出现开局三车顶着货物原地大角度旋转。
        target_xy = self._get_target_xy()[env_ids]
        payload_xy = payload_pose[:, :2]
        direction = target_xy - payload_xy
        direction_norm = torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
        move_dir = direction / direction_norm
        lateral_dir = torch.stack((-move_dir[:, 1], move_dir[:, 0]), dim=1)
        init_yaw = torch.atan2(move_dir[:, 1], move_dir[:, 0])
        self.agv_yaw[env_ids] = init_yaw.unsqueeze(1).repeat(1, 3)

        offsets = torch.tensor(self.cfg.support_offsets_xy, device=self.device, dtype=torch.float32)
        for i, agv in enumerate(self.agvs):
            pose = torch.zeros((num_reset, 7), device=self.device)
            support_xy = payload_xy + offsets[i, 0] * move_dir + offsets[i, 1] * lateral_dir
            pose[:, 0:2] = support_xy
            terrain_center, terrain_roll, terrain_pitch, terrain_samples = self._sample_agv_terrain_plane(
                support_xy, init_yaw, env_origins[:, :2]
            )
            self.agv_terrain_roll[env_ids, i] = terrain_roll
            self.agv_terrain_pitch[env_ids, i] = terrain_pitch
            self.agv_terrain_samples[env_ids, i, :] = terrain_samples
            quat = self._rpy_to_quat(terrain_roll, terrain_pitch, init_yaw)
            local_z = self._quat_rotate_z(quat)
            pose[:, 2] = (
                terrain_center
                + self.base_z_disturbance[env_ids, i]
                + 0.5 * float(self.cfg.agv_size[2]) * local_z[:, 2]
            )
            pose[:, 3:7] = quat
            vel = torch.zeros((num_reset, 6), device=self.device)
            agv.write_root_pose_to_sim(pose, env_ids=env_ids)
            agv.write_root_velocity_to_sim(vel, env_ids=env_ids)

        self._update_lift_poses(env_ids)

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
        self.last_rear_lateral_error[env_ids] = 0.0
        self.last_rear_lateral_correction[env_ids] = 0.0
        self.last_applied_angular_speed[env_ids] = 0.0
        self.last_virtual_carry_active[env_ids] = False
        self.last_virtual_stabilization_active[env_ids] = False

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

    def _compute_front_longitudinal_error(self) -> torch.Tensor:
        """AGV1 相对其 payload 支撑目标的纵向误差。

        正值表示 AGV1 位于目标前方（抢跑），负值表示落后。该量用于 V6.2.5
        的位置闭环限速，不改变 contact margin 或支撑点几何。
        """
        payload_xy = self.payload.data.root_pos_w[:, :2]
        target_xy = self._get_target_xy()
        move_dir, lateral_dir = self._compute_move_frame(payload_xy, target_xy)
        support_targets = self._compute_support_targets(payload_xy, move_dir, lateral_dir)
        front_xy = self.agvs[0].data.root_pos_w[:, :2]
        front_error_vec = front_xy - support_targets[:, 0, :]
        return torch.sum(front_error_vec * move_dir, dim=1)

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

    def _compute_support_z_gaps(self) -> torch.Tensor:
        """payload 底面与三块 Lift Plate 顶面的绝对 z gap。"""
        payload_bottom_z = self.payload.data.root_pos_w[:, 2] - 0.5 * self.cfg.payload_size[2]
        gaps = []
        for lift in self.lifts:
            local_z = self._quat_rotate_z(lift.data.root_quat_w)
            lift_top_z = (
                lift.data.root_pos_w[:, 2]
                + 0.5 * float(self.cfg.lift_plate_size[2]) * local_z[:, 2]
            )
            gaps.append(torch.abs(payload_bottom_z - lift_top_z))
        return torch.stack(gaps, dim=1)

    def _compute_support_contact_flags(self, support_targets: torch.Tensor) -> torch.Tensor:
        formation_errors = self._compute_formation_errors(support_targets)
        xy_ok = formation_errors < self.cfg.support_contact_xy_margin
        z_ok = self._compute_support_z_gaps() < self.cfg.support_contact_z_margin
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
        训练早期三台 AGV 可能接近共线，原始 barycentric 分母很小，直接除以
        1e-6 会得到极大负值并污染 reward 曲线。因此这里对退化三角形返回 -1，
        并对输出做限幅；该值仍可用于 success/debug，但不再作为未限幅 dense reward。
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
        degenerate = torch.abs(den) < 1e-4
        den_safe = torch.where(degenerate, torch.ones_like(den), den)
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den_safe
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den_safe
        w = 1.0 - u - v
        margin = torch.minimum(torch.minimum(u, v), w)
        margin = torch.where(degenerate, torch.full_like(margin, -1.0), margin)
        return torch.clamp(margin, min=-1.0, max=1.0)

    def _compute_success(
        self,
        goal_dist: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        slip_error: torch.Tensor,
        contact_count: torch.Tensor,
        support_margin: torch.Tensor,
    ) -> torch.Tensor:
        """V6.1.1 轻度崎岖矩形货物阶段的成功条件。

        崎岖支撑面会使解析 contact flag 在终点附近短时抖动。
        因此成功条件不再要求终止瞬间三车全部 contact=True，而是要求：
        - payload 已进入目标半径；
        - roll/pitch 与 slip 仍处于稳定范围；
        - 至少 success_required_support_contacts 台 AGV 有效支撑；
        - payload CoM 未严重越出支撑三角形。

        持续支撑丢失、掉落、倾覆和越界仍由 _get_dones() 的 failure flags 处理，
        不在 success 中放宽。
        """
        return (
            (goal_dist < self.cfg.target_radius)
            & (torch.abs(roll) < self.cfg.stable_roll_pitch_radius)
            & (torch.abs(pitch) < self.cfg.stable_roll_pitch_radius)
            & (slip_error < self.cfg.slip_success_threshold)
            & (contact_count >= float(self.cfg.success_required_support_contacts))
            & (support_margin > float(self.cfg.success_support_margin_min))
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
        """解析轻度起伏支撑面高度。

        说明：当前阶段仍是 kinematic AGV support，不是完整轮地接触 terrain。
        该函数只让三台支撑平台的 z 高度随 x/y 变化，用于验证三点支撑在
        低幅崎岖扰动下是否仍能稳定承载 payload。
        """
        if not bool(getattr(self.cfg, "enable_bumpy_support", False)):
            return torch.zeros(local_xy.shape[0], device=self.device)
        amp = float(self.cfg.bump_amplitude)
        kx = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_x), 1e-6)
        ky = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_y), 1e-6)
        phase_x = float(getattr(self.cfg, "bump_phase_x", 0.0))
        phase_y = float(getattr(self.cfg, "bump_phase_y", 0.0))
        return amp * torch.sin(kx * local_xy[:, 0] + phase_x) * torch.sin(ky * local_xy[:, 1] + phase_y)

    def _terrain_height_scalar(self, x: float, y: float) -> float:
        """与 _terrain_height() 完全一致的标量版，用于 USD 可视化 mesh 顶点。"""
        if not bool(getattr(self.cfg, "enable_bumpy_support", False)):
            return 0.0
        amp = float(self.cfg.bump_amplitude)
        kx = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_x), 1e-6)
        ky = 2.0 * math.pi / max(float(self.cfg.bump_wavelength_y), 1e-6)
        phase_x = float(getattr(self.cfg, "bump_phase_x", 0.0))
        phase_y = float(getattr(self.cfg, "bump_phase_y", 0.0))
        return amp * math.sin(kx * float(x) + phase_x) * math.sin(ky * float(y) + phase_y)

    def _sample_agv_terrain_plane(
        self, world_xy: torch.Tensor, yaw: torch.Tensor, env_xy: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample four equivalent contacts and fit the local support-plane attitude.

        Sample order is front-left, front-right, rear-left, rear-right.  The
        analytic terrain is expressed in environment-local coordinates while
        the footprint offsets rotate with the commanded AGV yaw.
        """
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
        offset_x = cos_yaw * local_contacts[None, :, 0] - sin_yaw * local_contacts[None, :, 1]
        offset_y = sin_yaw * local_contacts[None, :, 0] + cos_yaw * local_contacts[None, :, 1]
        sample_xy_w = world_xy[:, None, :] + torch.stack((offset_x, offset_y), dim=2)
        if env_xy is None:
            env_xy = self.scene.env_origins[:, :2]
        samples = self._terrain_height(
            (sample_xy_w - env_xy[:, None, :]).reshape(-1, 2)
        ).reshape(-1, 4)

        front_mean = 0.5 * (samples[:, 0] + samples[:, 1])
        rear_mean = 0.5 * (samples[:, 2] + samples[:, 3])
        left_mean = 0.5 * (samples[:, 0] + samples[:, 2])
        right_mean = 0.5 * (samples[:, 1] + samples[:, 3])
        slope_x = (front_mean - rear_mean) / max(2.0 * half_wheelbase, 1.0e-6)
        slope_y = (left_mean - right_mean) / max(2.0 * half_track, 1.0e-6)
        # Isaac/robotics ZYX convention: uphill along local +X is negative pitch;
        # uphill along local +Y is positive roll.
        pitch = -torch.atan(slope_x)
        roll = torch.atan(slope_y * torch.cos(pitch))
        return samples.mean(dim=1), roll, pitch, samples

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), device=yaw.device)
        quat[:, 0] = torch.cos(0.5 * yaw)
        quat[:, 3] = torch.sin(0.5 * yaw)
        return quat

    @staticmethod
    def _rpy_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Return scalar-first quaternion for intrinsic roll/pitch/yaw (ZYX)."""
        cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
        cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
        cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
        return torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dim=1,
        )

    @staticmethod
    def _quat_rotate_z(quat: torch.Tensor) -> torch.Tensor:
        """Rotate local unit +Z by scalar-first quaternions."""
        qw, qx, qy, qz = quat.unbind(dim=1)
        return torch.stack(
            (
                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ),
            dim=1,
        )

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
