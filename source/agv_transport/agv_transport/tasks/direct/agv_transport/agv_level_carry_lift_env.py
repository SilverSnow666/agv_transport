from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .agv_level_carry_env import AgvLevelCarryEnv
from .agv_level_carry_lift_env_cfg import AgvLevelCarryLiftVisualEnvCfg


class AgvLevelCarryLiftVisualEnv(AgvLevelCarryEnv):
    """Level-carry environment with a visual-only telescopic Lift mechanism.

    Physics are inherited unchanged from :class:`AgvLevelCarryEnv`: the three
    hidden kinematic Lift plates still carry the Board. This subclass only
    replaces the awkward moving iwhub single-mesh Lift with a visual mechanism
    that has a fixed base, two extending guide posts and a moving head.
    """

    cfg: AgvLevelCarryLiftVisualEnvCfg

    def _all_env_ids(self) -> torch.Tensor:
        """Return environment indices without relying on asset initialization internals.

        During ``_setup_scene`` the RigidObject handles already exist, but Isaac
        Lab has not necessarily initialized private helpers such as
        ``payload._ALL_INDICES`` yet.  Visual setup must therefore derive the
        indices directly from ``num_envs``.
        """
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    def _setup_scene(self) -> None:
        super()._setup_scene()

        self._telescopic_lift_visual_visible = True
        self._telescopic_lift_visual_enabled = not (
            self.num_envs > 8 and not (self.sim.has_gui() or self.sim.has_rtx_sensors())
        )

        # The native iwhub Lift is one merged visual mesh. Moving it as one body
        # makes the whole frame float. Keep it hidden and draw a kinematically
        # meaningful visual overlay instead. The sibling/hidden physical Lift
        # plates are not changed here.
        #
        # Important: _setup_scene runs before RigidObject initialization is
        # complete, so pass explicit env ids instead of letting the parent
        # access payload._ALL_INDICES.
        super()._set_native_lift_visual_visibility(False, self._all_env_ids())

        if not self._telescopic_lift_visual_enabled:
            return

        base_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/LevelCarryLiftBases",
            markers={
                "base": sim_utils.CuboidCfg(
                    size=tuple(float(v) for v in self.cfg.lift_visual_base_size),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(float(v) for v in self.cfg.lift_visual_base_color),
                        metallic=0.55,
                        roughness=0.28,
                    ),
                )
            },
        )
        column_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/LevelCarryLiftColumns",
            markers={
                "column": sim_utils.CuboidCfg(
                    size=tuple(float(v) for v in self.cfg.lift_visual_column_size),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(float(v) for v in self.cfg.lift_visual_column_color),
                        metallic=0.75,
                        roughness=0.20,
                    ),
                )
            },
        )
        head_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/LevelCarryLiftHeads",
            markers={
                "head": sim_utils.CuboidCfg(
                    size=tuple(float(v) for v in self.cfg.lift_visual_head_size),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(float(v) for v in self.cfg.lift_visual_head_color),
                        metallic=0.60,
                        roughness=0.25,
                    ),
                )
            },
        )
        self._lift_base_visualizer = VisualizationMarkers(base_cfg)
        self._lift_column_visualizer = VisualizationMarkers(column_cfg)
        self._lift_head_visualizer = VisualizationMarkers(head_cfg)
        self._initialize_telescopic_lift_visuals()

    def _initialize_telescopic_lift_visuals(self) -> None:
        """Create the final marker counts before the first simulation step."""
        if not self._telescopic_lift_visual_enabled:
            return
        device = self.device
        base_count = 3 * int(self.num_envs)
        column_count = 2 * base_count
        base_pos = torch.zeros((base_count, 3), device=device)
        base_pos[:, 2] = -10.0
        base_quat = torch.zeros((base_count, 4), device=device)
        base_quat[:, 0] = 1.0
        column_pos = torch.zeros((column_count, 3), device=device)
        column_pos[:, 2] = -10.0
        column_quat = torch.zeros((column_count, 4), device=device)
        column_quat[:, 0] = 1.0
        column_scale = torch.ones((column_count, 3), device=device)
        column_scale[:, 2] = float(self.cfg.lift_visual_min_column_height)

        self._lift_base_visualizer.visualize(translations=base_pos, orientations=base_quat)
        self._lift_head_visualizer.visualize(translations=base_pos, orientations=base_quat)
        self._lift_column_visualizer.visualize(
            translations=column_pos,
            orientations=column_quat,
            scales=column_scale,
        )

    @staticmethod
    def _quat_rotate_vector(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Rotate world-batched vectors by scalar-first quaternions."""
        qvec = quat[:, 1:4]
        t = 2.0 * torch.cross(qvec, vector, dim=1)
        return vector + quat[:, 0:1] * t + torch.cross(qvec, t, dim=1)

    def lift_visual_column_lengths(self) -> torch.Tensor:
        """Return the exposed guide-post lengths for all environments/lifts."""
        roof = 0.5 * float(self.cfg.agv_size[2])
        plate_thickness = float(self.cfg.lift_plate_size[2])
        clearance = float(self.cfg.board_support_clearance)
        head_height = float(self.cfg.lift_visual_head_size[2])
        column_start = roof - float(self.cfg.lift_visual_column_embed_depth)
        head_bottom = roof + self.lift_height + plate_thickness + clearance - head_height
        return torch.clamp(
            head_bottom - column_start,
            min=float(self.cfg.lift_visual_min_column_height),
        )

    def _update_telescopic_lift_visuals(self) -> None:
        if not getattr(self, "_telescopic_lift_visual_enabled", False):
            return
        if not hasattr(self, "_lift_base_visualizer"):
            return

        if not getattr(self, "_telescopic_lift_visual_visible", True):
            self._initialize_telescopic_lift_visuals()
            return

        base_positions = []
        base_orientations = []
        head_positions = []
        head_orientations = []
        column_positions = []
        column_orientations = []
        column_scales = []

        roof = 0.5 * float(self.cfg.agv_size[2])
        base_height = float(self.cfg.lift_visual_base_size[2])
        base_exposed = float(self.cfg.lift_visual_base_exposed_height)
        head_height = float(self.cfg.lift_visual_head_size[2])
        plate_thickness = float(self.cfg.lift_plate_size[2])
        clearance = float(self.cfg.board_support_clearance)
        column_embed = float(self.cfg.lift_visual_column_embed_depth)
        column_y = float(self.cfg.lift_visual_column_y_offset)

        for lift_index, agv in enumerate(self.agvs):
            agv_state = agv.data.root_state_w
            agv_pos = agv_state[:, 0:3]
            agv_quat = agv_state[:, 3:7]
            local_z = self._quat_rotate_z(agv_quat)

            # Base is mostly buried below the visual roof.
            base_center_offset = roof - 0.5 * base_height + base_exposed
            base_positions.append(agv_pos + local_z * base_center_offset)
            base_orientations.append(agv_quat)

            # Head top follows physical plate top + nominal Board clearance.
            lift_h = self.lift_height[:, lift_index]
            head_top_offset = roof + lift_h + plate_thickness + clearance
            head_center_offset = head_top_offset - 0.5 * head_height
            head_bottom_offset = head_top_offset - head_height
            head_positions.append(agv_pos + local_z * head_center_offset.unsqueeze(-1))
            head_orientations.append(agv_quat)

            column_start_offset = roof - column_embed
            column_length = torch.clamp(
                head_bottom_offset - column_start_offset,
                min=float(self.cfg.lift_visual_min_column_height),
            )
            column_center_offset = column_start_offset + 0.5 * column_length
            column_center = agv_pos + local_z * column_center_offset.unsqueeze(-1)

            for side in (-1.0, 1.0):
                local_side = torch.zeros_like(agv_pos)
                local_side[:, 1] = side * column_y
                world_side = self._quat_rotate_vector(agv_quat, local_side)
                column_positions.append(column_center + world_side)
                column_orientations.append(agv_quat)
                scale = torch.ones((self.num_envs, 3), device=self.device)
                scale[:, 2] = column_length
                column_scales.append(scale)

        self._lift_base_visualizer.visualize(
            translations=torch.cat(base_positions, dim=0),
            orientations=torch.cat(base_orientations, dim=0),
        )
        self._lift_head_visualizer.visualize(
            translations=torch.cat(head_positions, dim=0),
            orientations=torch.cat(head_orientations, dim=0),
        )
        self._lift_column_visualizer.visualize(
            translations=torch.cat(column_positions, dim=0),
            orientations=torch.cat(column_orientations, dim=0),
            scales=torch.cat(column_scales, dim=0),
        )

    def _update_native_lift_visuals(self, env_ids: torch.Tensor | None = None) -> None:
        """Replace whole-mesh native Lift translation with the telescopic overlay."""
        self._update_telescopic_lift_visuals()

    def _set_native_lift_visual_visibility(
        self, visible: bool, env_ids: torch.Tensor | None = None
    ) -> None:
        """Compatibility hook used by the A/B/C/D ablation script."""
        if env_ids is None:
            env_ids = self._all_env_ids()
        super()._set_native_lift_visual_visibility(False, env_ids)
        self._telescopic_lift_visual_visible = bool(visible)
        self._update_telescopic_lift_visuals()

    def set_lift_stack_initial_height(
        self,
        height: float,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        reposition_cargo: bool = True,
    ) -> None:
        """Place Lift + Board consistently at one starting height for demos/tests."""
        if env_ids is None:
            env_ids = self._all_env_ids()
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        height = float(height)
        lo = float(self.cfg.lift_min_height)
        hi = float(self.cfg.lift_max_height)
        if not lo <= height <= hi:
            raise ValueError(f"Lift height {height:.6f} m is outside [{lo:.6f}, {hi:.6f}] m")

        self.lift_height[env_ids] = height
        self.lift_target_height[env_ids] = height
        self.lift_velocity[env_ids] = 0.0
        self._update_lift_poses(env_ids)

        support_top_z = []
        roof = 0.5 * float(self.cfg.agv_size[2])
        plate_thickness = float(self.cfg.lift_plate_size[2])
        for agv in self.agvs:
            agv_state = agv.data.root_state_w[env_ids]
            local_z = self._quat_rotate_z(agv_state[:, 3:7])
            top_point = agv_state[:, 0:3] + local_z * (roof + height + plate_thickness)
            support_top_z.append(top_point[:, 2])
        support_top_z = torch.stack(support_top_z, dim=1).mean(dim=1)

        board_pose = self.payload.data.root_pose_w[env_ids].clone()
        board_pose[:, 2] = (
            support_top_z
            + float(self.cfg.board_support_clearance)
            + 0.5 * float(self.cfg.payload_size[2])
        )
        board_pose[:, 3:7] = 0.0
        board_pose[:, 3] = 1.0
        board_vel = torch.zeros((len(env_ids), 6), device=self.device)
        self.payload.write_root_pose_to_sim(board_pose, env_ids=env_ids)
        self.payload.write_root_velocity_to_sim(board_vel, env_ids=env_ids)

        if self.cargo is not None and reposition_cargo:
            cargo_pose = self.cargo.data.root_pose_w[env_ids].clone()
            cargo_pose[:, 0:2] = board_pose[:, 0:2]
            cargo_pose[:, 2] = (
                board_pose[:, 2]
                + 0.5 * float(self.cfg.payload_size[2])
                + 0.5 * float(self.cfg.cargo_size[2])
                + float(self.cfg.cargo_board_clearance)
            )
            cargo_pose[:, 3:7] = 0.0
            cargo_pose[:, 3] = 1.0
            cargo_vel = torch.zeros((len(env_ids), 6), device=self.device)
            self.cargo.write_root_pose_to_sim(cargo_pose, env_ids=env_ids)
            self.cargo.write_root_velocity_to_sim(cargo_vel, env_ids=env_ids)

        self._update_telescopic_lift_visuals()
