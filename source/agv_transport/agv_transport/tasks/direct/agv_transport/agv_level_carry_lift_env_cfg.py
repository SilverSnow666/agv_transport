from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .agv_level_carry_env_cfg import AgvLevelCarryEnvCfg


# Isaac Lab's @configclass converts the base configuration values into instance
# fields.  Accessing them as ``AgvLevelCarryEnvCfg.some_field`` is therefore not
# reliable after decoration (and caused an import-time AttributeError).  Keep a
# private base instance only for constructing this derived geometry config.
_BASE_CFG = AgvLevelCarryEnvCfg()


@configclass
class AgvLevelCarryLiftVisualEnvCfg(AgvLevelCarryEnvCfg):
    """V7 telescopic-Lift geometry with a thinner hidden support proxy.

    The active-leveling task still uses three hidden kinematic Lift plates as
    the physical Board supports. The old proxy was 40 mm thick, which made a
    30 mm neutral actuator height place the Board about 73 mm above the AGV
    roof (30 mm travel + 40 mm proxy + 3 mm reset clearance). That thickness
    was only an ideal collision proxy, not a real mechanical requirement.

    This config reduces the hidden Lift plate thickness to 15 mm while keeping
    the 0--100 mm actuator travel and 30 mm neutral working point unchanged.
    The nominal roof-to-Board-bottom distance therefore becomes about 48 mm:
    30 mm neutral travel + 15 mm support proxy + 3 mm clearance.

    All dependent initial heights and RigidObjectCfg instances are rebuilt here
    so visual geometry, physical Lift collision, Board reset height, Cargo reset
    height and drop threshold use the same 15 mm support thickness.
    """

    # ------------------------------------------------------------------
    # Physical Lift geometry override
    # ------------------------------------------------------------------
    lift_plate_size = (0.28, 0.28, 0.015)
    lift_plate_mass = float(_BASE_CFG.lift_plate_mass)

    # Recompute every height that the base class originally derived from the
    # old 40 mm plate. Do not simply override lift_plate_size: inherited
    # RigidObjectCfg defaults have already been constructed with the old size.
    payload_init_z = (
        float(_BASE_CFG.agv_top_z)
        + float(_BASE_CFG.lift_neutral_height)
        + lift_plate_size[2]
        + 0.5 * float(_BASE_CFG.payload_size[2])
        + float(_BASE_CFG.board_support_clearance)
    )
    payload_init_pos = (0.0, 0.0, payload_init_z)

    cargo_init_z = (
        payload_init_z
        + 0.5 * float(_BASE_CFG.payload_size[2])
        + 0.5 * float(_BASE_CFG.cargo_size[2])
        + float(_BASE_CFG.cargo_board_clearance)
    )
    cargo_init_pos = (0.0, 0.0, cargo_init_z)

    # Preserve the same allowed drop distance below the new nominal Board pose.
    payload_min_z = payload_init_z - 0.093

    lift_init_z = (
        float(_BASE_CFG.agv_top_z)
        + float(_BASE_CFG.lift_neutral_height)
        + 0.5 * lift_plate_size[2]
    )

    _support_offsets = tuple(tuple(float(v) for v in pair) for pair in _BASE_CFG.support_offsets_xy)

    # Recreate materials locally instead of depending on private decorated-class
    # attributes. Values match the base V7 carrying environment.
    _lift_material_thin = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.30,
        dynamic_friction=1.10,
        restitution=0.0,
    )
    _payload_material_thin = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.10,
        dynamic_friction=0.95,
        restitution=0.0,
    )
    _cargo_material_thin = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(_BASE_CFG.cargo_static_friction),
        dynamic_friction=float(_BASE_CFG.cargo_dynamic_friction),
        restitution=0.0,
    )

    lift1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Lift1",
        spawn=sim_utils.CuboidCfg(
            size=lift_plate_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=lift_plate_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_lift_material_thin,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.45, 0.85), metallic=0.15
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_support_offsets[0][0], _support_offsets[0][1], lift_init_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    lift2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Lift2",
        spawn=sim_utils.CuboidCfg(
            size=lift_plate_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=lift_plate_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_lift_material_thin,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.45, 0.85), metallic=0.15
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_support_offsets[1][0], _support_offsets[1][1], lift_init_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    lift3_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Lift3",
        spawn=sim_utils.CuboidCfg(
            size=lift_plate_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=lift_plate_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_lift_material_thin,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.45, 0.85), metallic=0.15
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_support_offsets[2][0], _support_offsets[2][1], lift_init_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Rebuild Board/Cargo configs because their inherited init-state objects
    # still contain the old 40 mm-stack heights even after overriding the scalar
    # payload_init_z/cargo_init_z values above.
    payload_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Payload",
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(v) for v in _BASE_CFG.payload_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(_BASE_CFG.payload_mass)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_payload_material_thin,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.55, 0.0), metallic=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=payload_init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    cargo_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cargo",
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(v) for v in _BASE_CFG.cargo_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(_BASE_CFG.cargo_mass)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_cargo_material_thin,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.65, 0.95), metallic=0.0, roughness=0.35
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=cargo_init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ------------------------------------------------------------------
    # Visual telescopic mechanism
    # ------------------------------------------------------------------
    lift_visual_base_size = (0.17, 0.15, 0.024)
    # Match the visible moving head thickness to the new thin support proxy.
    lift_visual_head_size = (0.18, 0.16, 0.015)
    # Unit-height cuboid. Runtime Z scale equals the exposed post length.
    lift_visual_column_size = (0.026, 0.026, 1.0)
    lift_visual_column_y_offset = 0.045
    # Put the base and posts slightly inside the roof so there is no floating gap.
    lift_visual_base_exposed_height = 0.004
    lift_visual_column_embed_depth = 0.002
    lift_visual_min_column_height = 0.002

    lift_visual_base_color = (0.30, 0.32, 0.35)
    lift_visual_column_color = (0.62, 0.64, 0.67)
    lift_visual_head_color = (0.24, 0.26, 0.29)
