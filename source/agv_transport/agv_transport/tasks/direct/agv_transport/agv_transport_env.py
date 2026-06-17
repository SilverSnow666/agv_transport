from __future__ import annotations

from collections.abc import Sequence
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .agv_transport_env_cfg import AgvTransportEnvCfg
from pxr import Usd, UsdGeom

class AgvTransportEnv(DirectRLEnv):
    """三 AGV 无连接协同推送 payload 的 DirectRLEnv 环境。

    当前版本：V5.2-B0-train-safe-narrow-corridor-clearance。

    目标：在 V5.2-A5 物理边界 zero-shot 成功基础上，小幅收窄物理通道，
    并加入轻量 wall-clearance penalty，验证 AGV2/AGV3 侧推策略在更强边界约束下是否仍稳定。

    核心设计：
    - 集中式单智能体控制三台差速 AGV，动作为 [v1, w1, v2, w2, v3, w3]。
    - 使用 active waypoint 子目标驱动 payload 按顺序经过路径点。
    - 允许任意两台 AGV 形成有效推动，不强制三车全程同时接触。
    - 在右转/下拐段招募 AGV2，在左转/上拐段招募 AGV3。
    - 有效推动奖励与 contact flag 均使用 AGV 与 payload 车身朝向平行度，
      避免侧车为了最大化朝向/接触奖励而转向 payload 质心形成 V 型挤压。
    - 保留 AGV 指向 payload 质心的几何量，仅用于车尾倒推等异常诊断。
    - 在路径两侧生成低矮物理边界。V5.2-B0 仍不改变 payload 形状、质量、摩擦或质心。
    - 使用手动左右边界控制点、稠密短墙段和 joint cap 生成连续 U 型通道。
    - 从收窄通道阶段开始加入 wall-clearance penalty，约束 kinematic AGV 贴墙/穿墙风险。
    - 默认不为每个墙段/cap 生成独立 PhysX/visual material，避免 2048 并行环境下触发 64K material limit。
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

        # D0A0h-plus：active waypoint 子目标编号。
        # 当前 target、队形方向和子目标距离奖励都由 active_goal_idx 决定。
        self.active_goal_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.prev_active_goal_dist = torch.zeros(self.num_envs, device=self.device)

        # D0A0g：waypoint gate 进度。
        # next_gate_idx 表示当前 episode 中 payload 下一步必须经过的 waypoint 编号。
        # 只有依次通过所有 waypoint gate，最终 success 才成立，避免开放空间 shortcut。
        self.next_gate_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

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
        active_goal_xy = self._get_active_goal_xy()
        self.prev_active_goal_dist[:] = torch.linalg.norm(
            payload_xy - active_goal_xy,
            dim=1,
        )
        self.prev_agv_payload_dist[:] = torch.linalg.norm(
            agv_xy - payload_xy,
            dim=1,
        )
        self.last_agv_escaped = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
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

    def _spawn_path_boundary_walls(self) -> None:
        """在 env_0 中生成连续、低矮、宽通道物理边界墙。

        V5.2-A4：优先使用手动左右边界控制点生成 U 型通道。

        之前的中心线 offset 方法在大 offset 距离和内弯组合下会产生
        offset 曲线自交，表现为某一侧墙体出现 X 型交叉或局部折返。
        joint cap 只能遮盖连接点，不能消除这种拓扑自交。

        本版默认采用显式 left/right boundary control points：
        - 左右墙分别独立定义，不再由中心线自动外扩；
        - 分别做 Chaikin 平滑和稠密采样；
        - 用短墙段 + joint cap 生成连续低矮物理边界；
        - 保留中心线 offset fallback，便于后续快速测试其它路径。
        """
        if not getattr(self.cfg, "enable_physical_path_boundaries", False):
            return

        wall_thickness = float(self.cfg.path_boundary_wall_thickness)
        wall_height = float(self.cfg.path_boundary_wall_height)
        smoothing_iterations = int(getattr(self.cfg, "path_boundary_smoothing_iterations", 3))
        sample_step = max(float(getattr(self.cfg, "path_boundary_sample_step", 0.20)), 0.05)
        segment_overlap = max(float(getattr(self.cfg, "path_boundary_segment_overlap", 0.0)), 0.0)
        use_joint_caps = bool(getattr(self.cfg, "path_boundary_use_joint_caps", True))
        joint_cap_radius = float(getattr(self.cfg, "path_boundary_joint_cap_radius", 0.5 * wall_thickness))
        joint_cap_radius = max(joint_cap_radius, 0.25 * wall_thickness)

        # IMPORTANT for large-scale training:
        # Do not create per-wall physics/visual materials by default.  With
        # num_envs=2048, each spawned segment/cap is cloned many times; if each
        # prim owns its own material, PhysX can hit the 64K material limit during
        # scene replication.  The walls use the default material unless
        # path_boundary_enable_materials=True is explicitly enabled for visual
        # inspection with small num_envs.
        enable_boundary_materials = bool(getattr(self.cfg, "path_boundary_enable_materials", False))
        wall_material = None
        visual_material = None
        if enable_boundary_materials:
            wall_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(getattr(self.cfg, "path_boundary_static_friction", 1.0)),
                dynamic_friction=float(getattr(self.cfg, "path_boundary_dynamic_friction", 1.0)),
                restitution=0.0,
            )
            visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(getattr(self.cfg, "path_boundary_color", (0.25, 0.25, 0.25))),
                metallic=0.0,
            )

        def _as_point_list(values) -> list[tuple[float, float]]:
            points: list[tuple[float, float]] = []
            for item in values:
                if len(item) < 2:
                    continue
                point = (float(item[0]), float(item[1]))
                if points and math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) <= 1e-6:
                    continue
                points.append(point)
            return points

        def _lerp(
            a: tuple[float, float],
            b: tuple[float, float],
            t: float,
        ) -> tuple[float, float]:
            return (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)

        def _chaikin_smooth(
            points: list[tuple[float, float]],
            iterations: int,
        ) -> list[tuple[float, float]]:
            """Chaikin corner-cutting，保留首尾点，生成更连续的墙体中心线。"""
            smoothed = points
            for _ in range(max(iterations, 0)):
                if len(smoothed) < 3:
                    break
                new_points: list[tuple[float, float]] = [smoothed[0]]
                for p0, p1 in zip(smoothed[:-1], smoothed[1:]):
                    new_points.append(_lerp(p0, p1, 0.25))
                    new_points.append(_lerp(p0, p1, 0.75))
                new_points.append(smoothed[-1])
                smoothed = new_points
            return smoothed

        def _dense_sample_polyline(
            points: list[tuple[float, float]],
            step: float,
        ) -> list[tuple[float, float]]:
            """按近似等距重采样，使墙体由均匀短段组成。"""
            if len(points) < 2:
                return points

            sampled: list[tuple[float, float]] = [points[0]]
            for p0, p1 in zip(points[:-1], points[1:]):
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                length = math.hypot(dx, dy)
                if length < 1e-9:
                    continue
                num_segments = max(1, int(math.ceil(length / step)))
                for k in range(1, num_segments + 1):
                    sampled.append((p0[0] + dx * k / num_segments, p0[1] + dy * k / num_segments))
            return sampled

        def _normalize(vec: tuple[float, float]) -> tuple[float, float]:
            length = math.hypot(vec[0], vec[1])
            if length < 1e-9:
                return (0.0, 0.0)
            return (vec[0] / length, vec[1] / length)

        def _build_auto_offset_edges() -> dict[str, list[tuple[float, float]]]:
            """自动 offset fallback。复杂弯道可能自交，默认不作为 V5.2-A4 主方案。"""
            payload_start_xy = tuple(float(v) for v in self.cfg.payload_init_pos[:2])
            start_extension = float(getattr(self.cfg, "path_boundary_start_extension", 0.0))
            extended_start_xy = (
                payload_start_xy[0] - start_extension,
                payload_start_xy[1],
            )

            raw_path_points = [
                extended_start_xy,
                payload_start_xy,
                *[tuple(float(v) for v in waypoint) for waypoint in self.cfg.waypoints],
            ]
            center_points = _as_point_list(raw_path_points)
            if len(center_points) < 2:
                return {"left": [], "right": []}

            smooth_centerline = _chaikin_smooth(center_points, smoothing_iterations)
            dense_centerline = _dense_sample_polyline(smooth_centerline, sample_step)
            if len(dense_centerline) < 2:
                return {"left": [], "right": []}

            segment_normals: list[tuple[float, float]] = []
            for p0, p1 in zip(dense_centerline[:-1], dense_centerline[1:]):
                direction = _normalize((p1[0] - p0[0], p1[1] - p0[1]))
                segment_normals.append((-direction[1], direction[0]))

            vertex_normals: list[tuple[float, float]] = []
            for idx in range(len(dense_centerline)):
                if idx == 0:
                    normal = segment_normals[0]
                elif idx == len(dense_centerline) - 1:
                    normal = segment_normals[-1]
                else:
                    normal = _normalize((
                        segment_normals[idx - 1][0] + segment_normals[idx][0],
                        segment_normals[idx - 1][1] + segment_normals[idx][1],
                    ))
                    if math.hypot(normal[0], normal[1]) < 1e-9:
                        normal = segment_normals[idx]
                vertex_normals.append(normal)

            inner_half_width = float(self.cfg.path_boundary_inner_half_width)
            wall_center_offset = inner_half_width + 0.5 * wall_thickness

            return {
                "left": [
                    (p[0] + n[0] * wall_center_offset, p[1] + n[1] * wall_center_offset)
                    for p, n in zip(dense_centerline, vertex_normals)
                ],
                "right": [
                    (p[0] - n[0] * wall_center_offset, p[1] - n[1] * wall_center_offset)
                    for p, n in zip(dense_centerline, vertex_normals)
                ],
            }

        def _build_manual_edges() -> dict[str, list[tuple[float, float]]]:
            left_points = _as_point_list(getattr(self.cfg, "path_boundary_left_points", ()))
            right_points = _as_point_list(getattr(self.cfg, "path_boundary_right_points", ()))
            if len(left_points) < 2 or len(right_points) < 2:
                return _build_auto_offset_edges()

            manual_smoothing_iterations = int(
                getattr(self.cfg, "path_boundary_manual_smoothing_iterations", smoothing_iterations)
            )
            return {
                "left": _dense_sample_polyline(
                    _chaikin_smooth(left_points, manual_smoothing_iterations),
                    sample_step,
                ),
                "right": _dense_sample_polyline(
                    _chaikin_smooth(right_points, manual_smoothing_iterations),
                    sample_step,
                ),
            }

        segment_kwargs = {
            "size": (sample_step, wall_thickness, wall_height),
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            "mass_props": sim_utils.MassPropertiesCfg(mass=1000.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
        }
        if enable_boundary_materials:
            segment_kwargs["physics_material"] = wall_material
            segment_kwargs["visual_material"] = visual_material
        segment_cfg = sim_utils.CuboidCfg(**segment_kwargs)

        cap_common_kwargs = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            "mass_props": sim_utils.MassPropertiesCfg(mass=1000.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
        }
        if enable_boundary_materials:
            cap_common_kwargs["physics_material"] = wall_material
            cap_common_kwargs["visual_material"] = visual_material

        if hasattr(sim_utils, "CylinderCfg"):
            cap_cfg = sim_utils.CylinderCfg(
                radius=joint_cap_radius,
                height=wall_height,
                axis="Z",
                **cap_common_kwargs,
            )
        else:
            cap_cfg = sim_utils.CuboidCfg(
                size=(2.0 * joint_cap_radius, 2.0 * joint_cap_radius, wall_height),
                **cap_common_kwargs,
            )

        def _spawn_wall_segment(
            prim_name: str,
            p0: tuple[float, float],
            p1: tuple[float, float],
        ) -> None:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                return

            yaw = math.atan2(dy, dx)
            quat_w = math.cos(0.5 * yaw)
            quat_z = math.sin(0.5 * yaw)
            center_x = 0.5 * (p0[0] + p1[0])
            center_y = 0.5 * (p0[1] + p1[1])

            segment_cfg.size = (length + 2.0 * segment_overlap, wall_thickness, wall_height)
            segment_cfg.func(
                prim_name,
                segment_cfg,
                translation=(center_x, center_y, 0.5 * wall_height),
                orientation=(quat_w, 0.0, 0.0, quat_z),
            )

        def _spawn_joint_cap(
            prim_name: str,
            point: tuple[float, float],
        ) -> None:
            cap_cfg.func(
                prim_name,
                cap_cfg,
                translation=(point[0], point[1], 0.5 * wall_height),
            )

        if bool(getattr(self.cfg, "path_boundary_use_manual_edges", True)):
            wall_edges = _build_manual_edges()
        else:
            wall_edges = _build_auto_offset_edges()

        for side_name in ("left", "right"):
            wall_points = wall_edges.get(side_name, [])
            if len(wall_points) < 2:
                continue

            for seg_idx, (p0, p1) in enumerate(zip(wall_points[:-1], wall_points[1:])):
                _spawn_wall_segment(
                    f"/World/envs/env_0/PathBoundary/{side_name}/segment_{seg_idx}",
                    p0,
                    p1,
                )

            if use_joint_caps:
                for cap_idx, point in enumerate(wall_points):
                    _spawn_joint_cap(
                        f"/World/envs/env_0/PathBoundary/{side_name}/cap_{cap_idx}",
                        point,
                    )


    def _spawn_irregular_payload_lobes(self) -> None:
        """Spawn light irregular payload lobes as child colliders of Payload.

        V5.3-A0 keeps the main payload rigid body, mass and CoM unchanged.  The
        added lobes are child collision shapes under the payload prim, so they
        act as a mild compound-contact irregularity without introducing a new
        dynamic object or a welded joint.  This is intentionally conservative:
        the goal is to test whether the V5.2 boundary-capable policy transfers
        to a non-rectangular contact outline before introducing strong CoM
        shifts or a full custom USD asset.
        """
        if not bool(getattr(self.cfg, "enable_irregular_payload", False)):
            return

        lobes = getattr(self.cfg, "irregular_payload_lobes", ())
        if not lobes:
            return

        use_materials = bool(getattr(self.cfg, "irregular_payload_enable_materials", True))
        visual_material = None
        physics_material = None
        if use_materials:
            visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(getattr(self.cfg, "irregular_payload_color", (1.0, 0.32, 0.08))),
                metallic=0.0,
            )
            physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(getattr(self.cfg, "irregular_payload_static_friction", 0.9)),
                dynamic_friction=float(getattr(self.cfg, "irregular_payload_dynamic_friction", 0.8)),
                restitution=0.0,
            )

        for lobe_idx, lobe in enumerate(lobes):
            if len(lobe) != 2:
                raise ValueError(
                    "Each irregular_payload_lobes entry must be "
                    "(local_pos_xyz, size_xyz)."
                )
            local_pos, size = lobe
            local_pos = tuple(float(v) for v in local_pos)
            size = tuple(float(v) for v in size)
            if len(local_pos) != 3 or len(size) != 3:
                raise ValueError(
                    "irregular payload local_pos and size must both have length 3."
                )
            if any(v <= 0.0 for v in size):
                raise ValueError(f"Invalid irregular payload lobe size: {size}")

            lobe_cfg_kwargs = {
                "size": size,
                "collision_props": sim_utils.CollisionPropertiesCfg(),
            }
            if use_materials:
                lobe_cfg_kwargs["physics_material"] = physics_material
                lobe_cfg_kwargs["visual_material"] = visual_material

            lobe_cfg = sim_utils.CuboidCfg(**lobe_cfg_kwargs)
            lobe_cfg.func(
                f"/World/envs/env_0/Payload/IrregularLobe_{lobe_idx}",
                lobe_cfg,
                translation=local_pos,
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

    def _get_manual_boundary_world_points(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return manual physical boundary centerlines in world frame.

        The points are wall centerlines defined in local env coordinates.
        They are used only for soft wall-clearance diagnostics/reward.  The
        actual collision geometry is spawned separately in env_0 and cloned.
        """
        if not bool(getattr(self.cfg, "enable_physical_path_boundaries", False)):
            return None, None
        if not bool(getattr(self.cfg, "path_boundary_use_manual_edges", True)):
            return None, None

        left_points = getattr(self.cfg, "path_boundary_left_points", None)
        right_points = getattr(self.cfg, "path_boundary_right_points", None)
        if left_points is None or right_points is None:
            return None, None
        if len(left_points) < 2 or len(right_points) < 2:
            return None, None

        left_local = torch.tensor(left_points, dtype=torch.float32, device=self.device)
        right_local = torch.tensor(right_points, dtype=torch.float32, device=self.device)
        origins = self.scene.env_origins[:, None, :2]
        return origins + left_local[None, :, :], origins + right_local[None, :, :]

    @staticmethod
    def _point_to_polyline_distance(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
        """Compute minimum 2D distance from each point to a batched polyline.

        Args:
            points: Tensor with shape [num_envs, 2].
            polyline: Tensor with shape [num_envs, num_points, 2].

        Returns:
            Tensor with shape [num_envs].
        """
        if polyline is None or polyline.shape[1] < 2:
            return torch.full((points.shape[0],), float("nan"), device=points.device)

        p0 = polyline[:, :-1, :]
        p1 = polyline[:, 1:, :]
        segment = p1 - p0
        segment_len_sq = torch.sum(segment * segment, dim=2).clamp_min(1e-12)
        rel = points[:, None, :] - p0
        t = torch.sum(rel * segment, dim=2) / segment_len_sq
        t = torch.clamp(t, 0.0, 1.0)
        projection = p0 + t[:, :, None] * segment
        return torch.linalg.norm(points[:, None, :] - projection, dim=2).min(dim=1).values

    def _compute_wall_clearance_penalties(
        self,
        payload_xy: torch.Tensor,
        agv_payload_dists: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized wall-clearance penalties for payload and AGVs.

        V5.2.3 changes the B0 clearance penalty from raw meter-squared
        violations to normalized margin deficits.  This makes small but
        important AGV-wall clearance violations visible to PPO.

        The AGV term uses the worst AGV in each environment instead of the
        mean/sum over all three AGVs.  In the current policy AGV2 is the
        wall-critical robot, while AGV1 and AGV3 usually remain far from the
        wall; averaging would dilute AGV2's safety signal.
        """
        del agv_payload_dists  # reserved for future role-conditioned variants

        zeros = torch.zeros(self.num_envs, device=self.device)
        if not bool(getattr(self.cfg, "enable_wall_clearance_penalty", False)):
            return zeros, zeros

        left_world, right_world = self._get_manual_boundary_world_points()
        if left_world is None or right_world is None:
            return zeros, zeros

        wall_half_thickness = 0.5 * float(getattr(self.cfg, "path_boundary_wall_thickness", 0.08))
        payload_half_width = 0.5 * float(getattr(self.cfg, "payload_size", (0.90, 1.20, 0.30))[1])
        agv_half_width = 0.5 * float(getattr(self.cfg, "agv_size", (0.70, 0.45, 0.06))[1])

        payload_dist_to_wall = torch.minimum(
            self._point_to_polyline_distance(payload_xy, left_world),
            self._point_to_polyline_distance(payload_xy, right_world),
        )
        payload_clearance = payload_dist_to_wall - wall_half_thickness - payload_half_width

        payload_margin = max(float(getattr(self.cfg, "payload_wall_clearance_margin", 0.20)), 1e-6)
        payload_deficit = torch.clamp(
            (payload_margin - payload_clearance) / payload_margin,
            min=0.0,
            max=1.0,
        )
        payload_penalty = torch.square(payload_deficit)

        agv_xy = torch.stack(
            (
                self.agv1.data.root_pos_w[:, :2],
                self.agv2.data.root_pos_w[:, :2],
                self.agv3.data.root_pos_w[:, :2],
            ),
            dim=1,
        )

        agv_margin = max(float(getattr(self.cfg, "agv_wall_clearance_margin", 0.08)), 1e-6)
        agv_deficits = []
        for agv_idx in range(3):
            agv_dist_to_wall = torch.minimum(
                self._point_to_polyline_distance(agv_xy[:, agv_idx, :], left_world),
                self._point_to_polyline_distance(agv_xy[:, agv_idx, :], right_world),
            )
            agv_clearance = agv_dist_to_wall - wall_half_thickness - agv_half_width
            agv_deficit = torch.clamp(
                (agv_margin - agv_clearance) / agv_margin,
                min=0.0,
                max=1.0,
            )
            agv_deficits.append(agv_deficit)

        agv_deficit = torch.stack(agv_deficits, dim=1)

        # Use the worst AGV rather than an average.  This prevents AGV2's
        # near-wall behavior from being masked by AGV1/AGV3, which usually
        # have large positive clearances.
        agv_penalty = torch.max(torch.square(agv_deficit), dim=1).values

        return payload_penalty, agv_penalty

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

        # V5.2-A：生成路径两侧宽通道低矮物理边界。
        self._spawn_path_boundary_walls()

        # V5.3-A0：在 Payload 根 prim 下添加轻度异形凸起子碰撞体。
        # 该凸起会随 env_0 一同 clone 到全部并行环境。
        self._spawn_irregular_payload_lobes()

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

        _, path_lateral_error, path_progress, _ = (
            self._compute_path_tracking_quantities()
        )
        total_path_length = self._compute_path_total_length()
        path_progress_ratio = path_progress / total_path_length

        # D0A0h-plus：当前控制目标使用 active waypoint 子目标。
        # 路径投影量仍保留给评估、progress ratio 和轻量路径误差项。
        target_xy = self._get_target_xy()

        payload_to_target = target_xy - payload_xy

        payload_goal_dist = torch.linalg.norm(
            payload_to_target,
            dim=1,
        )

        push_dir = payload_to_target / payload_goal_dist.unsqueeze(-1).clamp_min(1e-6)

        # 路径投影进度只用于评估和最大进度保持；主推进奖励改为 active subgoal 距离减少。
        path_progress_delta = path_progress - self.prev_path_progress
        self.prev_path_progress[:] = path_progress.detach()

        (
            active_goal_dist,
            subgoal_progress_delta,
            subgoal_reached,
            intermediate_subgoal_reached,
            is_final_active_goal,
        ) = self._update_active_subgoal(payload_xy)

        self.prev_payload_goal_dist[:] = payload_goal_dist.detach()

        payload_yaw = self._get_payload_yaw()
        payload_yaw_abs = torch.abs(payload_yaw)
        payload_yaw_rate_abs = torch.abs(self._get_payload_yaw_rate())

        payload_speed = torch.linalg.norm(
            self.payload.data.root_lin_vel_w[:, :2],
            dim=1,
        )

        # D0A0h-plus：当前 active segment 信息，用于接近子目标减速、越过子目标惩罚和 yaw 引导。
        (
            prev_subgoal_xy,
            active_subgoal_xy,
            active_segment_dir,
            active_segment_len,
        ) = self._get_active_segment_info()

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
            heading_to_payload_center,
            heading_parallel_to_payload,
            front_dists,
            rear_dists,
            v_actions,
        ) = self._compute_contact_geometry(payload_xy)

        # 有效推动朝向不再要求侧车指向 payload 质心。
        # 对宽矩形 payload，侧车若追求“车头指向质心”，会自然形成 V 型内扣，
        # 压缩中心车空间。这里改为奖励 AGV 车头与 payload 车身朝向一致，
        # 使三车以近似平行姿态推送。
        front_facing_score = torch.clamp(
            (
                heading_parallel_to_payload - self.cfg.front_contact_heading_min
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

        # 接触时车头若未与 payload 车身方向保持一致，给软惩罚。
        # 该项与 push_utility 使用同一平行朝向指标，避免重新引入侧车内扣梯度。
        bad_contact_heading_penalty = torch.sum(
            contact_flags_float
            * torch.square(
                torch.clamp(
                    self.cfg.front_contact_heading_min - heading_parallel_to_payload,
                    min=0.0,
                )
            ),
            dim=1,
        ) / 3.0

        _, bad_rear_push = self._compute_bad_rear_push(payload_xy)
        self.last_bad_rear_push[:] = bad_rear_push.detach()

        # D0A0h：主推进信号来自 active waypoint 子目标距离减少。
        # 这样模型必须先接近当前 waypoint，再切换到下一个 waypoint，不能只沿直线找终点。
        positive_progress = torch.clamp(subgoal_progress_delta, min=0.0)
        negative_progress = torch.clamp(-subgoal_progress_delta, min=0.0)
        progress_gate = torch.clamp(
            positive_progress / 0.005,
            min=0.0,
            max=1.0,
        )

        # D0A0g-easy：温和软路径走廊惩罚。
        # 横向误差在较宽 corridor 半宽内不额外惩罚，超出后按平方轻惩罚。
        path_corridor_violation = torch.clamp(
            path_lateral_error - self.cfg.path_corridor_half_width,
            min=0.0,
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
        base_two_pusher_gate = torch.clamp(
            second_push_utility / self.cfg.two_pusher_gate_threshold,
            min=0.0,
            max=1.0,
        )

        # 提取当前路径段是否处于显著转弯状态
        is_turning = (torch.abs(active_segment_dir[:, 1]) > self.cfg.turn_role_y_threshold).float()

        # 转弯豁免机制：如果是转弯段，强行提振 two_pusher_gate，允许外侧单车合法发力推进而不被重罚
        two_pusher_gate = torch.where(
            is_turning > 0.5,
            torch.clamp(base_two_pusher_gate + 0.6, max=1.0),
            base_two_pusher_gate
        )

        # D0A0g-easy：soft corridor-gated progress reward。
        # 与上一版不同，这里不把偏离路径时的 progress reward 直接压到 0，
        # 而是使用 0.5~1.0 的混合 gate，逐步压小 shortcut，同时保护已学到的两车推送能力。
        corridor_gate = torch.clamp(
            1.0 - path_lateral_error / self.cfg.progress_corridor_width,
            min=0.0,
            max=1.0,
        )
        progress_gate_soft = 0.5 + 0.5 * corridor_gate

        progress_reward = (
            progress_gate_soft
            * (
                self.cfg.subgoal_progress_reward_scale * positive_progress
                + self.cfg.progress_two_pusher_bonus_scale * positive_progress * two_pusher_gate
            )
            - self.cfg.subgoal_backward_penalty_scale * negative_progress
        )

        # D0A0h-plus：接近 active waypoint 时减速，避免到第 4 个 waypoint 后继续高速直推。
        near_subgoal = torch.clamp(
            1.0 - active_goal_dist / self.cfg.subgoal_slow_radius,
            min=0.0,
            max=1.0,
        )

        # D0A0h-plus：如果 payload 已经沿当前段方向越过 active waypoint，则给 overshoot 惩罚。
        # 这项直接针对“到第 4 个 waypoint 后继续往前推、越过终点”的现象。
        overshoot = torch.sum(
            (payload_xy - active_subgoal_xy) * active_segment_dir,
            dim=1,
        )
        subgoal_overshoot = torch.clamp(overshoot, min=0.0)

        # D0A0h-plus：payload yaw 与当前 active segment 方向对齐，帮助转弯段提前调整姿态。
        desired_segment_yaw = torch.atan2(
            active_segment_dir[:, 1],
            active_segment_dir[:, 0],
        )
        subgoal_yaw_error = torch.atan2(
            torch.sin(payload_yaw - desired_segment_yaw),
            torch.cos(payload_yaw - desired_segment_yaw),
        )
        subgoal_yaw_error_abs = torch.abs(subgoal_yaw_error)

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
        # 增加以下双向对称角色切换逻辑：
        # 1. 右转/下拐判定 (主动招募左侧车 AGV2 发力，压制右侧车 AGV3)
        right_turn_gate = (
                active_segment_dir[:, 1] < -self.cfg.turn_role_y_threshold
        ).float()

        agv2_right_turn_push_reward = right_turn_gate * push_utility[:, 1]
        agv2_right_turn_contact_reward = right_turn_gate * torch.clamp(
            1.0 - contact_zone_errors[:, 1] / self.cfg.turn_role_contact_zone_norm,
            min=0.0,
            max=1.0,
        )
        agv3_right_turn_penalty = right_turn_gate * push_utility[:, 2]

        # 2. 左转/上拐判定 (主动招募右侧车 AGV3 发力，压制左侧车 AGV2)
        left_turn_gate = (
                active_segment_dir[:, 1] > self.cfg.turn_role_y_threshold
        ).float()

        agv3_left_turn_push_reward = left_turn_gate * push_utility[:, 2]
        agv3_left_turn_contact_reward = left_turn_gate * torch.clamp(
            1.0 - contact_zone_errors[:, 2] / self.cfg.turn_role_contact_zone_norm,
            min=0.0,
            max=1.0,
        )
        agv2_left_turn_penalty = left_turn_gate * push_utility[:, 1]

        # 3. 汇总为对称的统一控制项
        turn_push_reward = agv2_right_turn_push_reward + agv3_left_turn_push_reward
        turn_contact_reward = agv2_right_turn_contact_reward + agv3_left_turn_contact_reward
        turn_opposite_penalty = agv3_right_turn_penalty + agv2_left_turn_penalty

        # ======== 【新增段落：AGV1 转弯让权判定】 ========
        is_turning_gate = torch.clamp(right_turn_gate + left_turn_gate, max=1.0)
        # 如果处于转弯段，AGV1 若继续强行输出正向推力，将受到计算
        agv1_turn_penalty = is_turning_gate * push_utility[:, 0]
        # ===================================================

        # D0A0g-easy：waypoint gate 先关闭，避免从 v4.3f 到强路径约束时任务骤崩。
        # 后续当 path_lat_max 降到 0.25~0.30 后，再把 cfg.enable_waypoint_gate 设为 True。
        if getattr(self.cfg, "enable_waypoint_gate", False):
            waypoint_gate_passed, waypoint_gate_dist, all_waypoint_gates_passed = (
                self._update_waypoint_gate(payload_xy)
            )
            waypoint_gate_reward = (
                self.cfg.waypoint_gate_reward_scale * waypoint_gate_passed.float()
            )
        else:
            waypoint_gate_passed = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            waypoint_gate_dist = torch.zeros(self.num_envs, device=self.device)
            all_waypoint_gates_passed = torch.ones(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            waypoint_gate_reward = torch.zeros(self.num_envs, device=self.device)

        # D0A0h-plus：通过中间子目标时给稠密而明确的小奖励。
        # 只对中间 waypoint 给奖励；最终 goal 由 success reward 处理。
        subgoal_reach_reward = (
            self.cfg.subgoal_reach_reward_scale
            * intermediate_subgoal_reached.float()
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

        agv_escaped = torch.any(
            agv_payload_dists > getattr(self.cfg, "agv_escape_dist_threshold", 2.0),
            dim=1,
        )

        payload_wall_clearance_penalty, agv_wall_clearance_penalty = (
            self._compute_wall_clearance_penalties(payload_xy)
        )

        # D0A0c：第三台低贡献 AGV 不必强行参与，但已有两台有效推动时，
        # 低贡献 AGV 应保持低动作、低动作变化率，并处于可重新加入的待命距离。
        low_utility_weight = torch.clamp(
            (self.cfg.idle_low_utility_threshold - push_utility)
            / self.cfg.idle_low_utility_threshold,
            min=0.0,
            max=1.0,
        )
        idle_gate = (two_pusher_gate > self.cfg.idle_two_pusher_gate_threshold).float()

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

        # D0A0h-plus：已有两车有效推动时，低贡献、未接触 payload 的 AGV 不应持续向前空跑。
        idle_forward_no_contact_penalty = torch.sum(
            idle_gate.unsqueeze(1)
            * low_utility_weight
            * (~contact_flags).float()
            * torch.clamp(v_actions, min=0.0),
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

        progress_success = path_progress_ratio > self.cfg.success_progress_ratio

        path_success = path_lateral_error < self.cfg.success_path_lateral_error

        # D0A0h-plus：最终成功要求 active waypoint 已推进到最后一个目标。
        # 中间路径由 active_goal_idx 顺序切换保证，不再依赖强 waypoint gate。
        success = (
            is_final_active_goal
            & position_success
            & yaw_success
        )

        out_of_bounds = self._compute_out_of_bounds()

        action_penalty = torch.sum(
            torch.square(self.actions),
            dim=1,
        )

        # D0A0h-plus：接近子目标时降低 AGV 总动作幅度，给 payload 转向和下一段对齐留出时间。
        subgoal_action_slow_penalty = near_subgoal * action_penalty
        subgoal_speed_penalty = near_subgoal * payload_speed

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

                - self.cfg.path_lateral_error_scale * path_lateral_error
                - self.cfg.path_corridor_penalty_scale * path_corridor_violation * path_corridor_violation

                # D0A0h-plus：子目标转向稳定项。
                - self.cfg.subgoal_speed_penalty_scale * subgoal_speed_penalty
                - self.cfg.subgoal_action_slow_penalty_scale * subgoal_action_slow_penalty
                - self.cfg.subgoal_overshoot_penalty_scale * subgoal_overshoot * subgoal_overshoot
                - self.cfg.subgoal_yaw_alignment_scale * subgoal_yaw_error_abs

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

                # 替换为以下对称计算项：
                + self.cfg.turn_role_contact_zone_reward_scale * turn_contact_reward
                + self.cfg.turn_role_push_reward_scale * turn_push_reward
                - self.cfg.turn_opposite_push_penalty_scale * turn_opposite_penalty

                - getattr(self.cfg, "turn_center_push_penalty_scale", 2.0) * agv1_turn_penalty

                + waypoint_gate_reward
                + subgoal_reach_reward
                + self.cfg.contact_persistence_reward_scale * contact_persistence_reward
                - self.cfg.single_pusher_progress_penalty_scale * single_pusher_progress_penalty
                - self.cfg.progress_drop_penalty_scale * progress_drop
                - self.cfg.idle_action_penalty_scale * idle_action_penalty
                - self.cfg.idle_action_rate_penalty_scale * idle_action_rate_penalty
                - self.cfg.idle_standby_penalty_scale * idle_standby_penalty
                - self.cfg.idle_forward_no_contact_penalty_scale * idle_forward_no_contact_penalty

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

                - getattr(self.cfg, "agv_escape_penalty_scale", 50.0) * agv_escaped.float()

                # V5.2-B0：物理通道收窄后，轻量约束 AGV/payload 与墙体的安全间隙。
                - getattr(self.cfg, "payload_wall_clearance_penalty_scale", 0.0) * payload_wall_clearance_penalty
                - getattr(self.cfg, "agv_wall_clearance_penalty_scale", 0.0) * agv_wall_clearance_penalty

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

        _, path_lateral_error, path_progress, _ = self._compute_path_tracking_quantities()
        total_path_length = self._compute_path_total_length()
        path_progress_ratio = path_progress / total_path_length

        final_target_xy = self._get_final_target_xy()

        final_goal_dist = torch.linalg.norm(
            payload_xy - final_target_xy,
            dim=1,
        )

        payload_yaw = self._get_payload_yaw()

        position_success = final_goal_dist < self.cfg.target_radius

        yaw_success = torch.abs(payload_yaw) < self.cfg.target_yaw_radius

        progress_success = path_progress_ratio > self.cfg.success_progress_ratio

        path_success = path_lateral_error < self.cfg.success_path_lateral_error

        # D0A0h：中间 waypoint 是否按顺序通过由 active_goal_idx 记录。
        # 只有 active goal 已推进到最后一个 waypoint 后，才允许最终 success。
        final_idx = max(len(self.cfg.waypoints) - 1, 0)
        is_final_active_goal = self.active_goal_idx >= final_idx

        # --- [新增] 修复切角捷径：要求依次通过所有 waypoint 门限 ---
        if getattr(self.cfg, "enable_waypoint_gate", False):
            all_gates_passed = self._all_waypoint_gates_passed()
            success = (
                    is_final_active_goal
                    & position_success
                    & yaw_success
                    & all_gates_passed
            )
        else:
            success = (
                    is_final_active_goal
                    & position_success
                    & yaw_success
            )

        out_of_bounds = self._compute_out_of_bounds()

        _, bad_rear_push = self._compute_bad_rear_push(payload_xy)
        self.last_bad_rear_push[:] = bad_rear_push.detach()

        # --- [新增] 修复角色逃逸：计算 AGV 到 payload 的距离并判定是否逃逸 ---
        agv1_dist = torch.linalg.norm(self.agv1.data.root_pos_w[:, :2] - payload_xy, dim=1)
        agv2_dist = torch.linalg.norm(self.agv2.data.root_pos_w[:, :2] - payload_xy, dim=1)
        agv3_dist = torch.linalg.norm(self.agv3.data.root_pos_w[:, :2] - payload_xy, dim=1)

        agv_payload_dists = torch.stack((agv1_dist, agv2_dist, agv3_dist), dim=1)

        agv_escaped = torch.any(
            agv_payload_dists > getattr(self.cfg, "agv_escape_dist_threshold", 2.0),
            dim=1,
        )

        # 保存 terminal 判断时刻的真实状态，供评估脚本读取
        self.last_success[:] = success.detach()
        self.last_position_success[:] = position_success.detach()
        self.last_yaw_success[:] = yaw_success.detach()
        self.last_out_of_bounds[:] = out_of_bounds.detach()
        self.last_payload_goal_dist[:] = final_goal_dist.detach()
        self.last_payload_yaw_abs[:] = torch.abs(payload_yaw).detach()
        self.last_agv_escaped[:] = agv_escaped.detach()

        # 基础终止判定
        if getattr(self.cfg, "terminate_on_bad_rear_push", False):
            terminated = success | out_of_bounds | bad_rear_push
        else:
            terminated = success | out_of_bounds

        # --- [新增] 挂载逃逸截断：如果某台车跑得太远，直接终止当前 episode ---
        if getattr(self.cfg, "terminate_on_agv_escape", False):
            terminated = terminated | agv_escaped

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
        self.active_goal_idx[env_ids_tensor] = 0
        self.next_gate_idx[env_ids_tensor] = 0
        self.episode_positive_progress[env_ids_tensor] = 0.0
        self.episode_two_pusher_progress[env_ids_tensor] = 0.0

        self.prev_payload_goal_dist[env_ids_tensor] = torch.linalg.norm(
            payload_xy - target_xy,
            dim=1,
        )

        active_goal_xy = self._get_active_goal_xy()[env_ids_tensor]
        self.prev_active_goal_dist[env_ids_tensor] = torch.linalg.norm(
            payload_xy - active_goal_xy,
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

    def _get_active_segment_info(self):
        """返回当前 active waypoint 所在路径段的信息。

        active_goal_idx=k 对应路径点序列中的第 k+1 个点：
        path_points = [payload_init_pos, waypoint_0, waypoint_1, ...]
        因此当前段为 path_points[k] -> path_points[k+1]。

        Returns:
            prev_goal_xy: 当前段起点，shape=[num_envs, 2]
            active_goal_xy: 当前 active waypoint，shape=[num_envs, 2]
            segment_dir: 当前段单位方向，shape=[num_envs, 2]
            segment_len: 当前段长度，shape=[num_envs]
        """
        path_points = self._get_path_points_xy()
        num_points = path_points.shape[1]
        env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        goal_path_idx = torch.clamp(
            self.active_goal_idx + 1,
            min=1,
            max=num_points - 1,
        )
        prev_path_idx = torch.clamp(
            goal_path_idx - 1,
            min=0,
            max=num_points - 1,
        )

        prev_goal_xy = path_points[env_ids, prev_path_idx]
        active_goal_xy = path_points[env_ids, goal_path_idx]
        segment_vec = active_goal_xy - prev_goal_xy
        segment_len = torch.linalg.norm(
            segment_vec,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        segment_dir = segment_vec / segment_len

        return prev_goal_xy, active_goal_xy, segment_dir, segment_len.squeeze(-1)

    def _get_active_goal_xy(self) -> torch.Tensor:
        """返回当前 active waypoint 子目标坐标。"""
        waypoints_xy = self._get_waypoints_xy()
        env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        num_goals = len(self.cfg.waypoints)
        if num_goals <= 0:
            return self._get_final_target_xy()

        goal_idx = torch.clamp(
            self.active_goal_idx,
            min=0,
            max=num_goals - 1,
        )

        return waypoints_xy[env_ids, goal_idx]

    def _get_active_goal_radius(self) -> torch.Tensor:
        """返回当前 active waypoint 的通过半径。"""
        num_goals = len(self.cfg.waypoints)
        final_idx = max(num_goals - 1, 0)

        is_final_goal = self.active_goal_idx >= final_idx

        subgoal_radius = torch.full(
            (self.num_envs,),
            float(getattr(self.cfg, "subgoal_radius", 0.32)),
            device=self.device,
        )
        final_radius = torch.full(
            (self.num_envs,),
            float(getattr(self.cfg, "subgoal_final_radius", self.cfg.target_radius)),
            device=self.device,
        )

        return torch.where(is_final_goal, final_radius, subgoal_radius)

    def _update_active_subgoal(self, payload_xy: torch.Tensor):
        """根据 payload 位置更新 active waypoint 子目标。

        Returns:
            active_goal_dist: 更新前到当前子目标的距离
            subgoal_progress_delta: 相比上一控制步，当前子目标距离减少量
            subgoal_reached: 当前子目标是否已到达
            intermediate_subgoal_reached: 是否到达了中间子目标
            is_final_active_goal: 更新后 active goal 是否已经是最后一个 waypoint
        """
        num_goals = len(self.cfg.waypoints)
        if num_goals <= 0:
            zeros = torch.zeros(self.num_envs, device=self.device)
            flags = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return zeros, zeros, flags, flags, flags

        active_goal_xy = self._get_active_goal_xy()
        active_goal_dist = torch.linalg.norm(
            payload_xy - active_goal_xy,
            dim=1,
        )

        subgoal_progress_delta = self.prev_active_goal_dist - active_goal_dist

        final_idx = num_goals - 1
        is_final_before_update = self.active_goal_idx >= final_idx
        active_goal_radius = self._get_active_goal_radius()
        subgoal_reached = active_goal_dist < active_goal_radius
        intermediate_subgoal_reached = subgoal_reached & (~is_final_before_update)

        # 到达中间 waypoint 后切到下一个 waypoint；最终 waypoint 不再继续增加。
        self.active_goal_idx[:] = torch.where(
            intermediate_subgoal_reached,
            torch.clamp(self.active_goal_idx + 1, max=final_idx),
            self.active_goal_idx,
        )

        new_active_goal_xy = self._get_active_goal_xy()
        self.prev_active_goal_dist[:] = torch.linalg.norm(
            payload_xy - new_active_goal_xy,
            dim=1,
        ).detach()

        is_final_active_goal = self.active_goal_idx >= final_idx

        return (
            active_goal_dist,
            subgoal_progress_delta,
            subgoal_reached,
            intermediate_subgoal_reached,
            is_final_active_goal,
        )

    def _update_waypoint_gate(self, payload_xy: torch.Tensor):
        """更新并返回 waypoint gate 状态。

        Returns:
            gate_passed: 当前步是否通过下一 waypoint gate，shape=[num_envs]
            gate_dist: payload 到当前 gate 的距离，shape=[num_envs]
            all_gates_passed: 是否已经依次通过所有 waypoint gate，shape=[num_envs]
        """
        num_gates = len(self.cfg.waypoints)
        if num_gates <= 0:
            gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            gate_dist = torch.zeros(self.num_envs, device=self.device)
            all_gates_passed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return gate_passed, gate_dist, all_gates_passed

        waypoints_xy = self._get_waypoints_xy()
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        active_gate = self.next_gate_idx < num_gates
        gate_idx = torch.clamp(self.next_gate_idx, max=num_gates - 1)
        gate_xy = waypoints_xy[env_ids, gate_idx]
        gate_dist = torch.linalg.norm(payload_xy - gate_xy, dim=1)

        gate_passed = active_gate & (gate_dist < self.cfg.waypoint_gate_radius)
        self.next_gate_idx[:] = torch.clamp(
            self.next_gate_idx + gate_passed.long(),
            max=num_gates,
        )

        all_gates_passed = self.next_gate_idx >= num_gates
        return gate_passed, gate_dist, all_gates_passed

    def _all_waypoint_gates_passed(self) -> torch.Tensor:
        """返回是否已依次通过所有 waypoint gate。"""
        return self.next_gate_idx >= len(self.cfg.waypoints)

    def _compute_path_total_length(self) -> torch.Tensor:
        """返回每个环境中规划路径的总弧长，shape = [num_envs]."""
        path_points = self._get_path_points_xy()
        segment_vec = path_points[:, 1:, :] - path_points[:, :-1, :]
        segment_len = torch.linalg.norm(segment_vec, dim=2)
        return torch.sum(segment_len, dim=1).clamp_min(1e-6)

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
        """返回当前 active waypoint 子目标。"""
        if getattr(self.cfg, "enable_subgoal_waypoint", True):
            return self._get_active_goal_xy()

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
        payload_heading_xy = torch.stack(
            (
                cos_yaw,
                sin_yaw,
            ),
            dim=1,
        )

        payload_half_x = 0.5 * self.cfg.payload_size[0]
        payload_half_y = float(
            getattr(
                self.cfg,
                "payload_contact_half_width_y",
                0.5 * self.cfg.payload_size[1],
            )
        )
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

            # Contact flag 的朝向 gate 必须与有效推动奖励保持同一语义：
            # 前端接触车辆应与 payload 车身朝向平行，而不是朝向 payload 质心。
            # 否则侧车会重新被诱导向中心偏头，形成 V 型挤压。
            heading_parallel_to_payload = torch.sum(
                agv_heading_xy * payload_heading_xy,
                dim=1,
            )
            front_facing = heading_parallel_to_payload > self.cfg.front_contact_heading_min

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
        payload_half_y = float(
            getattr(
                self.cfg,
                "payload_contact_half_width_y",
                0.5 * self.cfg.payload_size[1],
            )
        )
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
        """计算 AGV 与 payload 的接触几何关系。

        返回：
            heading_to_payload_center:
                AGV 车头方向与 AGV->payload 质心方向的点积，shape=[num_envs, 3]。
                仅用于车尾倒推、异常接触等几何诊断。
            heading_parallel_to_payload:
                AGV 车头方向与 payload 当前车身朝向的点积，shape=[num_envs, 3]。
                用于有效前端推动奖励，避免侧车为了最大化奖励而朝 payload
                质心内扣。
            front_dists:
                每台 AGV 前端点到 payload 质心的距离，shape=[num_envs, 3]。
            rear_dists:
                每台 AGV 后端点到 payload 质心的距离，shape=[num_envs, 3]。
            v_actions:
                每台 AGV 当前线速度动作，shape=[num_envs, 3]。
        """
        if payload_xy is None:
            payload_xy = self.payload.data.root_pos_w[:, :2]

        half_length = 0.5 * self.cfg.agv_size[0]

        payload_yaw = self._get_payload_yaw()
        payload_heading_xy = torch.stack(
            (
                torch.cos(payload_yaw),
                torch.sin(payload_yaw),
            ),
            dim=1,
        )

        heading_to_payload_center_list = []
        heading_parallel_to_payload_list = []
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

            heading_to_payload_center = torch.sum(
                agv_heading_xy * dir_to_payload,
                dim=1,
            )
            heading_parallel_to_payload = torch.sum(
                agv_heading_xy * payload_heading_xy,
                dim=1,
            )

            front_xy = agv_xy + agv_heading_xy * half_length
            rear_xy = agv_xy - agv_heading_xy * half_length

            front_dist = torch.linalg.norm(front_xy - payload_xy, dim=1)
            rear_dist = torch.linalg.norm(rear_xy - payload_xy, dim=1)

            heading_to_payload_center_list.append(heading_to_payload_center)
            heading_parallel_to_payload_list.append(heading_parallel_to_payload)
            front_dist_list.append(front_dist)
            rear_dist_list.append(rear_dist)
            v_action_list.append(self.actions[:, 2 * i])

        heading_to_payload_center = torch.stack(
            heading_to_payload_center_list,
            dim=1,
        )
        heading_parallel_to_payload = torch.stack(
            heading_parallel_to_payload_list,
            dim=1,
        )
        front_dists = torch.stack(front_dist_list, dim=1)
        rear_dists = torch.stack(rear_dist_list, dim=1)
        v_actions = torch.stack(v_action_list, dim=1)

        return (
            heading_to_payload_center,
            heading_parallel_to_payload,
            front_dists,
            rear_dists,
            v_actions,
        )

    def _compute_bad_rear_push(self, payload_xy: torch.Tensor | None = None):
        """判断是否出现明显车尾倒推 payload。

        C0 阶段该项只用于 reward 软惩罚和评估记录；
        默认不作为 done 终止条件。
        """
        if payload_xy is None:
            payload_xy = self.payload.data.root_pos_w[:, :2]

        contact_flags = self._compute_contact_flags().float()

        (
            heading_to_payload_center,
            _,
            front_dists,
            rear_dists,
            v_actions,
        ) = self._compute_contact_geometry(payload_xy)

        rear_closer_than_front = (
            rear_dists + self.cfg.front_rear_margin < front_dists
        ).float()

        heading_away_from_payload = (
            heading_to_payload_center < self.cfg.bad_rear_heading_threshold
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