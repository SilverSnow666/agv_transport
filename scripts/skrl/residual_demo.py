"""Opt-in visual demo helpers; training defaults and physics stay unchanged."""

import math


def configure_demo(cfg, args):
    """Freeze one reproducible condition, retaining terrain-aware warm reset."""
    if args.task.split(":")[-1] != "Template-Agv-Level-Residual-Direct-v0":
        raise ValueError("--residual-demo requires Template-Agv-Level-Residual-Direct-v0")
    if args.num_envs not in (None, 1):
        raise ValueError("--residual-demo supports exactly one environment")
    if args.ml_framework != "torch":
        raise ValueError("--residual-demo currently supports torch only")
    values = (args.demo_speed, args.demo_amplitude, args.demo_phase_x, args.demo_phase_y)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Demo parameters must be finite")
    if not 0 <= args.demo_speed <= cfg.max_agv_linear_speed:
        raise ValueError("Demo speed is outside the AGV speed range")
    if not 0 <= args.demo_amplitude <= 0.050:
        raise ValueError("Demo amplitude must be within 0..0.050 m")
    cfg.scene.num_envs = 1
    cfg.viewer.eye = (3.5, -4.0, 2.0)
    cfg.viewer.lookat = (0.7, 0.0, 0.15)
    cfg.enable_bumpy_support = True
    cfg.enable_visual_terrain_mesh = True
    cfg.visual_terrain_height_scale = 1.0
    cfg.visual_terrain_z_offset = 0.0
    # The base mesh option otherwise moves the physical catch plane to -80 mm.
    # Keep its training pose; hide its rendering only after scene creation.
    cfg.visual_terrain_ground_z = 0.0
    cfg.bump_amplitude = args.demo_amplitude
    cfg.bump_phase_x = args.demo_phase_x
    cfg.bump_phase_y = args.demo_phase_y
    cfg.residual_scripted_speed = args.demo_speed
    # Degenerate ranges disable variation, NOT the A.1 warm-start reset path.
    cfg.residual_domain_randomization = True
    cfg.residual_speed_range = (args.demo_speed,) * 2
    cfg.residual_terrain_amplitude_range = (args.demo_amplitude,) * 2
    cfg.residual_terrain_phase_x_range = (args.demo_phase_x,) * 2
    cfg.residual_terrain_phase_y_range = (args.demo_phase_y,) * 2
    cfg.residual_cargo_mass_range = (cfg.cargo_mass,) * 2
    cfg.residual_cargo_offset_x_range = (0.0, 0.0)
    cfg.residual_cargo_offset_y_range = (0.0, 0.0)
    print(
        f"[DEMO] Fixed terrain: speed={args.demo_speed:.3f} m/s, "
        f"amplitude={1000 * args.demo_amplitude:.1f} mm, "
        f"phase=({args.demo_phase_x:.3f}, {args.demo_phase_y:.3f}), "
        f"wavelength=({cfg.bump_wavelength_x}, {cfg.bump_wavelength_y}) m; "
        f"controller={'E zero residual' if args.zero_residual else 'F PPO mean action'}. "
        "Mesh is visual only; AGVs still follow analytical terrain."
    )


def prepare_and_check_visuals(env):
    """Validate actual USD vertices against the runtime terrain after each reset."""
    import torch
    from pxr import Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    ground = stage.GetPrimAtPath("/World/ground")
    UsdGeom.Imageable(ground).MakeInvisible()  # USD visibility does not disable collision.
    colliders = [prim for prim in Usd.PrimRange(ground) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders or not all(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() for prim in colliders):
        raise RuntimeError("Demo must retain the ground catch-plane collision")
    ground_z = UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
    if abs(ground_z) > 1e-7:
        raise RuntimeError(f"Demo unexpectedly moved ground collider: z={ground_z}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/envs/env_0/BumpyTerrainVisual"))
    if not mesh:
        raise RuntimeError("Demo terrain mesh is missing")
    for prim in Usd.PrimRange(mesh.GetPrim()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError("Demo mesh must not add physical collisions")
    # Prevent renderer subdivision from smoothing vertices away from the sampled surface.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    points = torch.tensor([tuple(p) for p in mesh.GetPointsAttr().Get()], device=env.device)
    error = torch.max(torch.abs(points[:, 2] - env._terrain_height(points[:, :2])))
    if float(error) > 1e-6:
        raise RuntimeError(f"Visible/analytical terrain mismatch: {float(error)} m")
    gap_error = torch.max(torch.abs(env.last_reset_support_gap - env.cfg.board_support_clearance))
    if float(gap_error) > 1e-5 or torch.any(env.last_reset_support_unreachable):
        raise RuntimeError("Demo warm reset has inconsistent support geometry")
    print(
        f"[DEMO CHECK] {len(points)} vertices, terrain max error={float(error) * 1000:.6f} mm; "
        f"reset gap={float(env.last_reset_support_gap.mean()) * 1000:.3f} mm; "
        "ground collision pose unchanged, flat ground hidden, visual mesh collision-free."
    )
