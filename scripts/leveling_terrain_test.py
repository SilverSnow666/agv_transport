# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V7.3/V7.4 Board-attitude feedback validation on terrain.

The three modes keep terrain, motion, support geometry, Cargo, and virtual-carry
settings identical. Only the Lift controller changes:

* ``no_leveling`` (C): fixed 30 mm neutral height.
* ``geometric`` (D): support-height feedforward.
* ``feedback`` (E): geometric feedforward plus Board-local roll/pitch PD.

Cargo support/contact values are analytical geometry proxies, not PhysX contact
sensor readings.

The V7.4 robustness suite repeats D/E under controlled speed, terrain, and
Cargo-mass variations. The terrain is analytic and deterministic, so changing
the reset seed alone would not create a different road profile. V7.5 can run
the same controller either in this script or through the reusable environment
implementation, which makes numerical parity directly testable.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Compare fixed, geometric, and Board-feedback Lift leveling on terrain."
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Template-Agv-Level-Carry-Direct-v0")
parser.add_argument(
    "--mode",
    choices=("no_leveling", "geometric", "feedback", "both", "all"),
    default="both",
)
parser.add_argument(
    "--suite",
    choices=("single", "robustness"),
    default="single",
    help="Run one requested mode set or the V7.4 D/E robustness matrix.",
)
parser.add_argument(
    "--controller_source",
    choices=("script", "environment"),
    default="script",
    help="Compute Lift targets in this benchmark script or in the reusable environment controller.",
)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--target_speed", type=float, default=0.10)
parser.add_argument("--log_dir", type=str, default="logs/v7_3")
parser.add_argument("--cargo_mass", type=float, default=None)
parser.add_argument("--bump_amplitude", type=float, default=None)
parser.add_argument("--bump_phase_x", type=float, default=None)
parser.add_argument("--bump_phase_y", type=float, default=None)
parser.add_argument("--feedback_roll_kp", type=float, default=None)
parser.add_argument("--feedback_pitch_kp", type=float, default=None)
parser.add_argument("--feedback_roll_kd", type=float, default=None)
parser.add_argument("--feedback_pitch_kd", type=float, default=None)
parser.add_argument("--feedback_max_correction_mm", type=float, default=None)
parser.add_argument("--feedback_filter_alpha", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import agv_transport.tasks  # noqa: F401


CSV_FIELDS = (
    "time",
    "agv1_z", "agv2_z", "agv3_z",
    "terrain_z1", "terrain_z2", "terrain_z3",
    "agv1_roll_deg", "agv2_roll_deg", "agv3_roll_deg",
    "agv1_pitch_deg", "agv2_pitch_deg", "agv3_pitch_deg",
    "agv1_fl_z", "agv1_fr_z", "agv1_rl_z", "agv1_rr_z",
    "agv2_fl_z", "agv2_fr_z", "agv2_rl_z", "agv2_rr_z",
    "agv3_fl_z", "agv3_fr_z", "agv3_rl_z", "agv3_rr_z",
    "lift1_height", "lift2_height", "lift3_height",
    "lift1_velocity", "lift2_velocity", "lift3_velocity",
    "lift_top_world_x1", "lift_top_world_y1", "lift_top_world_z1",
    "lift_top_world_x2", "lift_top_world_y2", "lift_top_world_z2",
    "lift_top_world_x3", "lift_top_world_y3", "lift_top_world_z3",
    "support_height_error_rms",
    "board_z", "board_roll_deg", "board_pitch_deg",
    "board_roll_rate", "board_pitch_rate",
    "board_rp_angular_speed", "board_vertical_accel",
    "feedback_roll_cmd", "feedback_pitch_cmd",
    "feedback_height1", "feedback_height2", "feedback_height3",
    "lift1_saturated", "lift2_saturated", "lift3_saturated", "board_yaw_deg",
    "cargo_x", "cargo_y", "cargo_z",
    "cargo_rel_x", "cargo_rel_y", "cargo_rel_z",
    "cargo_rel_dx", "cargo_rel_dy", "cargo_rel_dz",
    "cargo_relative_xy_displacement", "cargo_xy_slip_step",
    "cargo_xy_slip_distance", "cargo_max_xy_slip_distance",
    "cargo_roll_deg", "cargo_pitch_deg", "cargo_yaw_deg",
    "cargo_rel_roll_deg", "cargo_rel_pitch_deg", "cargo_rel_yaw_deg",
    "cargo_ang_vel_x", "cargo_ang_vel_y", "cargo_ang_vel_z", "cargo_ang_speed",
    "cargo_board_surface_gap",
    "cargo_center_over_board", "cargo_fully_supported", "cargo_contact",
    "cargo_dropped", "cargo_tipped",
)

ROBUSTNESS_CONDITIONS = (
    {
        "name": "baseline",
        "target_speed": 0.10,
        "bump_amplitude": 0.030,
        "bump_phase_x": 0.25,
        "bump_phase_y": 0.45,
        "cargo_mass": 4.0,
    },
    {
        "name": "slow",
        "target_speed": 0.05,
        "bump_amplitude": 0.030,
        "bump_phase_x": 0.25,
        "bump_phase_y": 0.45,
        "cargo_mass": 4.0,
    },
    {
        "name": "fast",
        "target_speed": 0.15,
        "bump_amplitude": 0.030,
        "bump_phase_x": 0.25,
        "bump_phase_y": 0.45,
        "cargo_mass": 4.0,
    },
    {
        "name": "rough",
        "target_speed": 0.10,
        "bump_amplitude": 0.045,
        "bump_phase_x": 0.25,
        "bump_phase_y": 0.45,
        "cargo_mass": 4.0,
    },
    {
        "name": "phase_shift",
        "target_speed": 0.10,
        "bump_amplitude": 0.030,
        "bump_phase_x": 1.10,
        "bump_phase_y": -0.35,
        "cargo_mass": 4.0,
    },
    {
        "name": "heavy_cargo",
        "target_speed": 0.10,
        "bump_amplitude": 0.030,
        "bump_phase_x": 0.25,
        "bump_phase_y": 0.45,
        "cargo_mass": 8.0,
    },
)

ROBUSTNESS_METRICS = (
    "roll_rms_deg",
    "pitch_rms_deg",
    "max_abs_roll_deg",
    "max_abs_pitch_deg",
    "board_rp_angular_speed_rms",
    "board_vertical_accel_rms",
    "support_height_rms_mm",
    "lift_velocity_rms_mm_s",
    "cargo_relative_xy_rms_mm",
    "cargo_max_relative_xy_mm",
    "cargo_cumulative_slip_mm",
    "cargo_ang_speed_rms",
    "cargo_contact_fraction",
    "cargo_dropped",
    "cargo_tipped",
    "lift_saturation_fraction",
    "terrain_span_mm",
)


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[1:4] = -result[1:4]
    return result


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind()
    rw, rx, ry, rz = right.unbind()
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat[1:4]
    twice_cross = 2.0 * torch.cross(quat_vector, vector, dim=0)
    return vector + quat[0] * twice_cross + torch.cross(quat_vector, twice_cross, dim=0)


def _quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.linalg.norm(quat).clamp_min(1.0e-12)
    qw, qx, qy, qz = quat.unbind()
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return torch.stack((roll, pitch, yaw))


def _cargo_board_state(raw_env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    board_pos = raw_env.payload.data.root_pos_w[0]
    board_quat = raw_env.payload.data.root_quat_w[0]
    cargo_pos = raw_env.cargo.data.root_pos_w[0]
    cargo_quat = raw_env.cargo.data.root_quat_w[0]
    board_inverse = _quat_conjugate(board_quat)
    relative_pos = _quat_rotate(board_inverse, cargo_pos - board_pos)
    relative_quat = _quat_multiply(board_inverse, cargo_quat)
    relative_quat = relative_quat / torch.linalg.norm(relative_quat).clamp_min(1.0e-12)

    # Project the Cargo oriented half-extents onto Board local +Z.  This gives
    # a face gap that remains meaningful while either body rolls or pitches.
    qw, qx, qy, qz = relative_quat.unbind()
    vertical_axis_projection = torch.stack((
        2.0 * (qx * qz - qw * qy),
        2.0 * (qy * qz + qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )).abs()
    cargo_half_size = 0.5 * torch.tensor(raw_env.cfg.cargo_size, device=raw_env.device)
    cargo_vertical_radius = torch.sum(vertical_axis_projection * cargo_half_size)
    surface_gap = relative_pos[2] - cargo_vertical_radius - 0.5 * float(raw_env.cfg.payload_size[2])
    return relative_pos, relative_quat, cargo_pos, surface_gap


def _make_actions(raw_env, target_speed: float) -> tuple[torch.Tensor, float, float]:
    max_speed = float(raw_env.cfg.max_agv_linear_speed)
    if max_speed <= 0.0:
        raise ValueError(f"cfg.max_agv_linear_speed must be positive, got {max_speed}")
    actual_target = min(max(float(target_speed), 0.0), max_speed)
    normalized_command = actual_target / max_speed
    if bool(getattr(raw_env.cfg, "forward_only_linear_speed", True)):
        linear_action = 2.0 * normalized_command - 1.0
    else:
        linear_action = normalized_command
    linear_action = min(max(linear_action, -1.0), 1.0)
    actions = torch.zeros(raw_env.action_space.shape, device=raw_env.device)
    actions[:, 0::2] = linear_action
    actions[:, 1::2] = 0.0
    return actions, actual_target, linear_action


def _lift_top_centers(raw_env) -> torch.Tensor:
    centers = []
    half_thickness = 0.5 * float(raw_env.cfg.lift_plate_size[2])
    for lift in raw_env.lifts:
        quat = lift.data.root_quat_w[0].unsqueeze(0)
        local_z = raw_env._quat_rotate_z(quat)[0]
        centers.append(lift.data.root_pos_w[0] + half_thickness * local_z)
    return torch.stack(centers)


def _geometric_lift_target(raw_env) -> tuple[torch.Tensor, torch.Tensor]:
    """Equalize the three physical support-top world heights."""
    top_before = _lift_top_centers(raw_env)
    top_error = top_before[:, 2].mean() - top_before[:, 2]
    agv_quat = torch.stack([agv.data.root_quat_w[0] for agv in raw_env.agvs])
    axis_z = raw_env._quat_rotate_z(agv_quat)[:, 2].clamp_min(0.25)
    return raw_env.lift_height[0] + top_error / axis_z, axis_z


def _feedback_lift_target(
    raw_env, filtered_height: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add a zero-mean Board roll/pitch PD correction to geometric leveling."""
    geometric_target, axis_z = _geometric_lift_target(raw_env)
    roll, pitch, _ = raw_env._get_payload_rpy()
    board_quat = raw_env.payload.data.root_quat_w[0]
    board_ang_vel_local = _quat_rotate(
        _quat_conjugate(board_quat), raw_env.payload.data.root_ang_vel_w[0]
    )

    roll_cmd = -(
        float(raw_env.cfg.leveling_feedback_roll_kp) * roll[0]
        + float(raw_env.cfg.leveling_feedback_roll_kd) * board_ang_vel_local[0]
    )
    pitch_cmd = -(
        float(raw_env.cfg.leveling_feedback_pitch_kp) * pitch[0]
        + float(raw_env.cfg.leveling_feedback_pitch_kd) * board_ang_vel_local[1]
    )

    offsets = torch.tensor(
        raw_env.cfg.support_offsets_xy,
        device=raw_env.device,
        dtype=raw_env.lift_height.dtype,
    )
    height = -offsets[:, 0] * torch.tan(pitch_cmd) + offsets[:, 1] * torch.tan(roll_cmd)
    height -= height.mean()
    max_height = float(raw_env.cfg.leveling_feedback_max_correction)
    max_abs = torch.max(torch.abs(height)).clamp_min(1.0e-9)
    height *= torch.clamp(
        torch.tensor(max_height, device=raw_env.device, dtype=height.dtype) / max_abs,
        max=1.0,
    )

    alpha = float(raw_env.cfg.leveling_feedback_filter_alpha)
    filtered_height = (1.0 - alpha) * filtered_height + alpha * height
    return (
        geometric_target + filtered_height / axis_z,
        filtered_height,
        roll_cmd,
        pitch_cmd,
    )


def _summarize(rows: list[dict[str, float]], step_dt: float) -> dict[str, float]:
    result = {
        "roll_rms_deg": _rms([row["board_roll_deg"] for row in rows]),
        "pitch_rms_deg": _rms([row["board_pitch_deg"] for row in rows]),
        "max_abs_roll_deg": max(abs(row["board_roll_deg"]) for row in rows),
        "max_abs_pitch_deg": max(abs(row["board_pitch_deg"]) for row in rows),
        "support_height_rms_mm": 1000.0 * _rms([row["support_height_error_rms"] for row in rows]),
        "board_rp_angular_speed_rms": _rms(
            [row["board_rp_angular_speed"] for row in rows]
        ),
        "board_vertical_accel_rms": _rms([row["board_vertical_accel"] for row in rows]),
        "lift_velocity_rms_mm_s": 1000.0
        * _rms(
            [
                row[f"lift{lift_index}_velocity"]
                for row in rows
                for lift_index in range(1, 4)
            ]
        ),
        "feedback_height_rms_mm": 1000.0
        * _rms(
            [
                row[f"feedback_height{lift_index}"]
                for row in rows
                for lift_index in range(1, 4)
            ]
        ),
    }
    result.update({
        "cargo_relative_xy_rms_mm": 1000.0 * _rms(
            [row["cargo_relative_xy_displacement"] for row in rows]
        ),
        "cargo_final_relative_xy_mm": 1000.0 * rows[-1]["cargo_relative_xy_displacement"],
        "cargo_max_relative_xy_mm": 1000.0 * max(row["cargo_relative_xy_displacement"] for row in rows),
        "cargo_cumulative_slip_mm": 1000.0 * rows[-1]["cargo_xy_slip_distance"],
        "cargo_roll_rms_deg": _rms([row["cargo_roll_deg"] for row in rows]),
        "cargo_pitch_rms_deg": _rms([row["cargo_pitch_deg"] for row in rows]),
        "cargo_ang_speed_rms": _rms([row["cargo_ang_speed"] for row in rows]),
        "cargo_contact_fraction": sum(row["cargo_contact"] for row in rows) / len(rows),
        "cargo_dropped": max(row["cargo_dropped"] for row in rows),
        "cargo_tipped": max(row["cargo_tipped"] for row in rows),
    })
    saturated_samples = 0
    for lift_index in range(1, 4):
        heights = [row[f"lift{lift_index}_height"] for row in rows]
        flags = [row[f"lift{lift_index}_saturated"] for row in rows]
        result[f"lift{lift_index}_min_mm"] = 1000.0 * min(heights)
        result[f"lift{lift_index}_max_mm"] = 1000.0 * max(heights)
        result[f"lift{lift_index}_saturation_s"] = step_dt * sum(flags)
        result[f"lift{lift_index}_saturation_fraction"] = sum(flags) / len(flags)
        saturated_samples += sum(flags)
    result["lift_saturation_fraction"] = saturated_samples / (3.0 * len(rows))
    return result


def _print_summary(mode: str, summary: dict[str, float]) -> None:
    print(
        f"[RESULT] {mode}: roll RMS={summary['roll_rms_deg']:.4f} deg, "
        f"pitch RMS={summary['pitch_rms_deg']:.4f} deg, "
        f"max |roll|={summary['max_abs_roll_deg']:.4f} deg, "
        f"max |pitch|={summary['max_abs_pitch_deg']:.4f} deg, "
        f"support RMS={summary['support_height_rms_mm']:.3f} mm"
    )
    print(
        f"[RESULT] {mode}: Board roll/pitch angular-speed RMS="
        f"{summary['board_rp_angular_speed_rms']:.5f} rad/s, "
        f"vertical-acceleration RMS={summary['board_vertical_accel_rms']:.5f} m/s^2, "
        f"Lift velocity RMS={summary['lift_velocity_rms_mm_s']:.3f} mm/s, "
        f"feedback-height RMS={summary['feedback_height_rms_mm']:.3f} mm"
    )
    print(
        f"[RESULT] {mode}: Cargo relative XY RMS={summary['cargo_relative_xy_rms_mm']:.3f} mm, "
        f"max={summary['cargo_max_relative_xy_mm']:.3f} mm, "
        f"cumulative slip={summary['cargo_cumulative_slip_mm']:.3f} mm, "
        f"roll/pitch RMS={summary['cargo_roll_rms_deg']:.4f}/{summary['cargo_pitch_rms_deg']:.4f} deg, "
        f"angular-speed RMS={summary['cargo_ang_speed_rms']:.4f} rad/s, "
        f"contact={100.0 * summary['cargo_contact_fraction']:.2f}%, "
        f"dropped={bool(summary['cargo_dropped'])}, tipped={bool(summary['cargo_tipped'])}"
    )
    for lift_index in range(1, 4):
        print(
            f"[RESULT] {mode}: lift{lift_index}="
            f"[{summary[f'lift{lift_index}_min_mm']:.2f}, {summary[f'lift{lift_index}_max_mm']:.2f}] mm, "
            f"saturated={summary[f'lift{lift_index}_saturation_s']:.3f} s "
            f"({100.0 * summary[f'lift{lift_index}_saturation_fraction']:.2f}%)"
        )


def _make_environment():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    overrides = (
        ("feedback_roll_kp", "leveling_feedback_roll_kp", 1.0),
        ("feedback_pitch_kp", "leveling_feedback_pitch_kp", 1.0),
        ("feedback_roll_kd", "leveling_feedback_roll_kd", 1.0),
        ("feedback_pitch_kd", "leveling_feedback_pitch_kd", 1.0),
        ("feedback_max_correction_mm", "leveling_feedback_max_correction", 0.001),
        ("feedback_filter_alpha", "leveling_feedback_filter_alpha", 1.0),
    )
    for arg_name, cfg_name, scale in overrides:
        value = getattr(args_cli, arg_name)
        if value is not None:
            setattr(env_cfg, cfg_name, float(value) * scale)
    scenario_overrides = (
        ("bump_amplitude", "bump_amplitude"),
        ("bump_phase_x", "bump_phase_x"),
        ("bump_phase_y", "bump_phase_y"),
    )
    for arg_name, cfg_name in scenario_overrides:
        value = getattr(args_cli, arg_name)
        if value is not None:
            setattr(env_cfg, cfg_name, float(value))
    if args_cli.cargo_mass is not None:
        cargo_mass = float(args_cli.cargo_mass)
        if cargo_mass <= 0.0:
            raise ValueError(f"Cargo mass must be positive, got {cargo_mass}")
        env_cfg.cargo_mass = cargo_mass
        env_cfg.cargo_cfg.spawn.mass_props.mass = cargo_mass
    if float(env_cfg.bump_amplitude) <= 0.0:
        raise ValueError(f"Terrain bump amplitude must be positive, got {env_cfg.bump_amplitude}")
    gains = (
        float(env_cfg.leveling_feedback_roll_kp),
        float(env_cfg.leveling_feedback_pitch_kp),
        float(env_cfg.leveling_feedback_roll_kd),
        float(env_cfg.leveling_feedback_pitch_kd),
    )
    if any(value < 0.0 for value in gains):
        raise ValueError(f"Feedback gains must be non-negative, got {gains}")
    if float(env_cfg.leveling_feedback_max_correction) <= 0.0:
        raise ValueError("Feedback maximum height correction must be positive")
    alpha = float(env_cfg.leveling_feedback_filter_alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"Feedback filter alpha must be in (0, 1], got {alpha}")
    env_cfg.enable_front_position_guard = False
    env_cfg.enable_rear_lateral_guard = False
    env_cfg.enable_bumpy_support = True
    env_cfg.enable_cargo = True
    # The AGVs/lifts are kinematic proxies.  Reuse the existing V6 no-slip
    # coupling so the dynamic board follows the translating supports; without
    # it the board stays behind and the comparison terminates on support loss.
    env_cfg.enable_virtual_friction_carry = True
    # Each run selects the concrete mode immediately before reset. External is
    # the compatibility path used by the original V7.3/V7.4 benchmark logic.
    env_cfg.leveling_controller_mode = "external"
    env_cfg.seed = 0
    env_cfg.episode_length_s = max(float(args_cli.duration) + 2.0, float(env_cfg.episode_length_s))
    env = gym.make(args_cli.task, cfg=env_cfg)
    return env


def _configure_robustness_condition(
    raw_env,
    condition: dict[str, float | str],
    reference_cargo_masses: torch.Tensor,
    reference_cargo_inertias: torch.Tensor,
) -> None:
    """Apply one V7.4 condition without changing controller parameters."""
    raw_env.cfg.bump_amplitude = float(condition["bump_amplitude"])
    raw_env.cfg.bump_phase_x = float(condition["bump_phase_x"])
    raw_env.cfg.bump_phase_y = float(condition["bump_phase_y"])
    cargo_mass = float(condition["cargo_mass"])
    raw_env.cfg.cargo_mass = cargo_mass

    reference_mass = float(reference_cargo_masses[0, 0])
    mass_scale = cargo_mass / reference_mass
    masses = torch.full_like(reference_cargo_masses, cargo_mass)
    inertias = reference_cargo_inertias * mass_scale
    indices = torch.arange(raw_env.num_envs, dtype=torch.int32, device="cpu")
    raw_env.cargo.root_physx_view.set_masses(masses, indices)
    raw_env.cargo.root_physx_view.set_inertias(inertias, indices)

    # Keep the visible terrain consistent with the analytical support surface
    # after amplitude or phase changes. This mesh remains visual-only.
    raw_env._spawn_visual_bumpy_terrain()
    print(
        f"[CONDITION] {condition['name']}: speed={float(condition['target_speed']):.3f} m/s, "
        f"amplitude={1000.0 * float(condition['bump_amplitude']):.1f} mm, "
        f"phase=({float(condition['bump_phase_x']):.2f}, "
        f"{float(condition['bump_phase_y']):.2f}), Cargo mass={cargo_mass:.1f} kg"
    )


def run_mode(
    env, mode: str, log_dir: Path, target_speed_override: float | None = None
) -> dict[str, float]:
    raw_env = env.unwrapped
    environment_modes = {
        "no_leveling": "neutral",
        "geometric": "geometric",
        "feedback": "geometric_feedback",
    }
    raw_env.cfg.leveling_controller_mode = (
        environment_modes[mode] if args_cli.controller_source == "environment" else "external"
    )
    with torch.inference_mode():
        env.reset(seed=0)
    if raw_env.cargo is None:
        raise RuntimeError("V7.2 Cargo was not created")
    raw_env.base_z_disturbance.zero_()

    requested_speed = (
        float(args_cli.target_speed)
        if target_speed_override is None
        else float(target_speed_override)
    )
    actions, target_speed, linear_action = _make_actions(raw_env, requested_speed)
    neutral = float(raw_env.cfg.lift_neutral_height)
    lift_min = float(raw_env.cfg.lift_min_height)
    lift_max = float(raw_env.cfg.lift_max_height)
    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    rows: list[dict[str, float]] = []
    elapsed = 0.0
    initial_xy = torch.stack([agv.data.root_pos_w[0, :2] for agv in raw_env.agvs]).clone()
    initial_cargo_rel, _, _, initial_surface_gap = _cargo_board_state(raw_env)
    initial_cargo_rel = initial_cargo_rel.clone()
    previous_cargo_rel_xy = initial_cargo_rel[:2].clone()
    cumulative_slip = 0.0
    max_relative_slip = 0.0
    filtered_feedback_height = torch.zeros(3, device=raw_env.device)
    feedback_roll_cmd = torch.zeros((), device=raw_env.device)
    feedback_pitch_cmd = torch.zeros((), device=raw_env.device)
    previous_board_vz = raw_env.payload.data.root_lin_vel_w[0, 2].clone()
    board_half_xy = 0.5 * torch.tensor(raw_env.cfg.payload_size[:2], device=raw_env.device)
    cargo_half_xy = 0.5 * torch.tensor(raw_env.cfg.cargo_size[:2], device=raw_env.device)
    initial_center_over_board = bool(torch.all(torch.abs(initial_cargo_rel[:2]) <= board_half_xy))
    if not initial_center_over_board:
        raise RuntimeError(f"Cargo initial XY is outside Board: relative={initial_cargo_rel.tolist()} m")
    if abs(float(initial_surface_gap)) > 1.0e-4:
        raise RuntimeError(
            f"Cargo initial face gap must be zero (no penetration/suspension), got "
            f"{1000.0 * float(initial_surface_gap):.4f} mm"
        )
    print(
        f"[CHECK] {mode}: Cargo size={tuple(float(v) for v in raw_env.cfg.cargo_size)} m, "
        f"mass={float(raw_env.cfg.cargo_mass):.2f} kg, "
        f"friction={float(raw_env.cfg.cargo_static_friction):.2f}/"
        f"{float(raw_env.cfg.cargo_dynamic_friction):.2f} static/dynamic, "
        f"initial Board-frame xyz={initial_cargo_rel.tolist()} m, "
        f"face gap={1000.0 * float(initial_surface_gap):.4f} mm"
    )
    print(
        f"[CHECK] {mode}: Cargo support/contact are analytical geometry proxies; "
        "no PhysX contact sensor is used."
    )
    print(
        f"[INFO] {mode}: target speed={target_speed:.4f} m/s, action={linear_action:.6f}, "
        f"duration={args_cli.duration:.2f} s, controller source={args_cli.controller_source}"
    )
    if mode == "feedback":
        print(
            f"[CHECK] feedback gains: roll Kp/Kd="
            f"{float(raw_env.cfg.leveling_feedback_roll_kp):.3f}/"
            f"{float(raw_env.cfg.leveling_feedback_roll_kd):.3f} s, "
            f"pitch Kp/Kd={float(raw_env.cfg.leveling_feedback_pitch_kp):.3f}/"
            f"{float(raw_env.cfg.leveling_feedback_pitch_kd):.3f} s, "
            f"max correction={1000.0 * float(raw_env.cfg.leveling_feedback_max_correction):.2f} mm, "
            f"filter alpha={float(raw_env.cfg.leveling_feedback_filter_alpha):.3f}"
        )

    while simulation_app.is_running() and elapsed < float(args_cli.duration):
        if args_cli.controller_source == "script":
            if mode == "feedback":
                (
                    raw_target,
                    filtered_feedback_height,
                    feedback_roll_cmd,
                    feedback_pitch_cmd,
                ) = _feedback_lift_target(raw_env, filtered_feedback_height)
            elif mode == "geometric":
                raw_target, _ = _geometric_lift_target(raw_env)
                filtered_feedback_height.zero_()
                feedback_roll_cmd.zero_()
                feedback_pitch_cmd.zero_()
            else:
                raw_target = torch.full_like(raw_env.lift_height[0], neutral)
                filtered_feedback_height.zero_()
                feedback_roll_cmd.zero_()
                feedback_pitch_cmd.zero_()
            saturated = (raw_target < lift_min) | (raw_target > lift_max)
            raw_env.lift_target_height[0] = torch.clamp(raw_target, min=lift_min, max=lift_max)
        raw_env.base_z_disturbance[0] = 0.0

        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        if args_cli.controller_source == "environment":
            filtered_feedback_height = raw_env.leveling_feedback_height[0].clone()
            feedback_roll_cmd = raw_env.last_leveling_roll_cmd[0].clone()
            feedback_pitch_cmd = raw_env.last_leveling_pitch_cmd[0].clone()
            saturated = raw_env.last_leveling_saturated[0].clone()
        elapsed += step_dt
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError(
                f"Unexpected reset at t={elapsed:.3f} s "
                f"(terminated={bool(terminated[0])}, truncated={bool(truncated[0])})"
            )

        terrain_z = raw_env.agv_terrain_samples[0].mean(dim=1)
        agv_z = torch.stack([agv.data.root_pos_w[0, 2] for agv in raw_env.agvs])
        lift_top = _lift_top_centers(raw_env)
        lift_top_z = lift_top[:, 2]
        support_error = torch.sqrt(torch.mean(torch.square(lift_top_z - lift_top_z.mean())))
        roll, pitch, board_yaw = raw_env._get_payload_rpy()
        board_pos = raw_env.payload.data.root_pos_w[0]
        board_ang_vel = raw_env.payload.data.root_ang_vel_w[0]
        board_ang_vel_local = _quat_rotate(
            _quat_conjugate(raw_env.payload.data.root_quat_w[0]), board_ang_vel
        )
        board_rp_angular_speed = torch.linalg.norm(board_ang_vel_local[0:2])
        board_vz = raw_env.payload.data.root_lin_vel_w[0, 2]
        board_vertical_accel = (board_vz - previous_board_vz) / step_dt
        previous_board_vz = board_vz.clone()
        cargo_rel, cargo_rel_quat, cargo_pos, cargo_surface_gap = _cargo_board_state(raw_env)
        cargo_rel_delta = cargo_rel - initial_cargo_rel
        relative_xy_displacement = torch.linalg.norm(cargo_rel_delta[:2])
        slip_step = torch.linalg.norm(cargo_rel[:2] - previous_cargo_rel_xy)
        cumulative_slip += float(slip_step)
        max_relative_slip = max(max_relative_slip, float(relative_xy_displacement))
        previous_cargo_rel_xy = cargo_rel[:2].clone()
        cargo_rpy = _quat_to_rpy(raw_env.cargo.data.root_quat_w[0])
        cargo_rel_rpy = _quat_to_rpy(cargo_rel_quat)
        cargo_ang_vel = raw_env.cargo.data.root_ang_vel_w[0]
        cargo_ang_speed = torch.linalg.norm(cargo_ang_vel)
        center_over_board = torch.all(torch.abs(cargo_rel[:2]) <= board_half_xy)
        fully_supported = torch.all(torch.abs(cargo_rel[:2]) <= board_half_xy - cargo_half_xy)
        cargo_contact = center_over_board & (
            torch.abs(cargo_surface_gap) <= float(raw_env.cfg.cargo_contact_tolerance)
        )
        cargo_dropped = (~center_over_board) | (
            cargo_pos[2] < board_pos[2] - 0.5 * float(raw_env.cfg.payload_size[2])
        )
        cargo_tipped = torch.maximum(torch.abs(cargo_rpy[0]), torch.abs(cargo_rpy[1])) > float(
            raw_env.cfg.cargo_tip_threshold
        )

        values = torch.cat((
            agv_z, terrain_z,
            torch.rad2deg(raw_env.agv_terrain_roll[0]),
            torch.rad2deg(raw_env.agv_terrain_pitch[0]),
            raw_env.agv_terrain_samples[0].reshape(-1),
            raw_env.lift_height[0], raw_env.lift_velocity[0], lift_top.reshape(-1),
            support_error.reshape(1), board_pos[2].reshape(1),
            torch.rad2deg(roll[0]).reshape(1), torch.rad2deg(pitch[0]).reshape(1),
            board_ang_vel[0:2], board_rp_angular_speed.reshape(1),
            board_vertical_accel.reshape(1),
            feedback_roll_cmd.reshape(1), feedback_pitch_cmd.reshape(1),
            filtered_feedback_height, saturated.float(),
            torch.rad2deg(board_yaw[0]).reshape(1),
            cargo_pos, cargo_rel, cargo_rel_delta,
            relative_xy_displacement.reshape(1), slip_step.reshape(1),
            torch.tensor((cumulative_slip, max_relative_slip), device=raw_env.device),
            torch.rad2deg(cargo_rpy), torch.rad2deg(cargo_rel_rpy),
            cargo_ang_vel, cargo_ang_speed.reshape(1), cargo_surface_gap.reshape(1),
            center_over_board.float().reshape(1), fully_supported.float().reshape(1),
            cargo_contact.float().reshape(1), cargo_dropped.float().reshape(1),
            cargo_tipped.float().reshape(1),
        ))
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"NaN/Inf detected at t={elapsed:.3f} s")
        if bool(torch.any(raw_env.lift_height[0] < lift_min - 1.0e-6)) or bool(
            torch.any(raw_env.lift_height[0] > lift_max + 1.0e-6)
        ):
            raise RuntimeError(f"Lift travel limit exceeded at t={elapsed:.3f} s")
        rows.append(dict(zip(CSV_FIELDS, (elapsed, *values.tolist()), strict=True)))
    final_xy = torch.stack([agv.data.root_pos_w[0, :2] for agv in raw_env.agvs]).clone()

    if not rows:
        raise RuntimeError("Simulation produced no samples")
    displacement = torch.linalg.norm(final_xy - initial_xy, dim=1)
    measured_speed = displacement / elapsed
    lateral_drift = torch.abs(final_xy[:, 1] - initial_xy[:, 1])
    terrain_span = max(row[f"terrain_z{i}"] for row in rows for i in range(1, 4)) - min(
        row[f"terrain_z{i}"] for row in rows for i in range(1, 4)
    )
    if bool(torch.any(torch.abs(measured_speed - target_speed) > max(0.002, 0.02 * target_speed))):
        raise RuntimeError(f"AGV speed check failed: measured={measured_speed.tolist()} m/s")
    if bool(torch.any(lateral_drift > 1.0e-3)):
        raise RuntimeError(f"Straight-line check failed: lateral drift={lateral_drift.tolist()} m")
    if terrain_span < 1.0e-3:
        raise RuntimeError(f"Terrain excitation too small: span={1000.0 * terrain_span:.3f} mm")

    csv_name = {
        "no_leveling": "cargo_no_leveling.csv",
        "geometric": "cargo_geometric_leveling.csv",
        "feedback": "cargo_feedback_leveling.csv",
    }[mode]
    csv_path = log_dir / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, step_dt)
    summary["terrain_span_mm"] = 1000.0 * terrain_span
    summary["mean_measured_speed"] = float(measured_speed.mean())
    _print_summary(mode, summary)
    print(
        f"[CHECK] {mode}: speed={summary['mean_measured_speed']:.4f} m/s, "
        f"terrain span={summary['terrain_span_mm']:.2f} mm, CSV={csv_path}"
    )
    return summary


def _print_comparison(
    title: str, baseline: dict[str, float], candidate: dict[str, float]
) -> None:
    print(f"[COMPARISON] {title}:")
    for key, label in (
        ("roll_rms_deg", "Board roll RMS"),
        ("pitch_rms_deg", "Board pitch RMS"),
        ("max_abs_roll_deg", "max |roll|"),
        ("max_abs_pitch_deg", "max |pitch|"),
        ("board_rp_angular_speed_rms", "Board roll/pitch angular-speed RMS"),
        ("board_vertical_accel_rms", "Board vertical-acceleration RMS"),
        ("support_height_rms_mm", "support height RMS"),
        ("lift_velocity_rms_mm_s", "Lift velocity RMS"),
        ("cargo_relative_xy_rms_mm", "Cargo relative XY RMS"),
        ("cargo_max_relative_xy_mm", "Cargo max relative XY"),
        ("cargo_cumulative_slip_mm", "Cargo cumulative slip"),
        ("cargo_roll_rms_deg", "Cargo roll RMS"),
        ("cargo_pitch_rms_deg", "Cargo pitch RMS"),
        ("cargo_ang_speed_rms", "Cargo angular-speed RMS"),
    ):
        absolute = baseline[key] - candidate[key]
        if abs(baseline[key]) < 1.0e-9:
            print(
                f"[COMPARISON] {label}: {baseline[key]:.6g} -> "
                f"{candidate[key]:.6g}, percentage=n/a"
            )
        else:
            improvement = 100.0 * absolute / abs(baseline[key])
            print(
                f"[COMPARISON] {label}: {absolute:+.6g} absolute, "
                f"{improvement:+.2f}% improvement"
            )


def _run_single_suite(env, log_dir: Path) -> None:
    if args_cli.mode == "both":
        modes = ("no_leveling", "geometric")
    elif args_cli.mode == "all":
        modes = ("no_leveling", "geometric", "feedback")
    else:
        modes = (args_cli.mode,)
    summaries = {mode: run_mode(env, mode, log_dir) for mode in modes}
    if args_cli.mode in ("both", "all"):
        _print_comparison(
            "C -> D, geometric contribution",
            summaries["no_leveling"],
            summaries["geometric"],
        )
    if args_cli.mode == "all":
        _print_comparison(
            "D -> E, Board-attitude feedback contribution",
            summaries["geometric"],
            summaries["feedback"],
        )


def _run_robustness_suite(env, log_dir: Path) -> None:
    raw_env = env.unwrapped
    if raw_env.cargo is None:
        raise RuntimeError("V7.4 robustness suite requires Cargo")
    reference_masses = raw_env.cargo.root_physx_view.get_masses().clone()
    reference_inertias = raw_env.cargo.root_physx_view.get_inertias().clone()
    result_rows: list[dict[str, float | str]] = []

    for condition in ROBUSTNESS_CONDITIONS:
        _configure_robustness_condition(
            raw_env, condition, reference_masses, reference_inertias
        )
        condition_dir = log_dir / str(condition["name"])
        condition_dir.mkdir(parents=True, exist_ok=True)
        summaries = {
            mode: run_mode(
                env,
                mode,
                condition_dir,
                target_speed_override=float(condition["target_speed"]),
            )
            for mode in ("geometric", "feedback")
        }
        geometric = summaries["geometric"]
        feedback = summaries["feedback"]
        _print_comparison(
            f"{condition['name']}: D -> E feedback contribution",
            geometric,
            feedback,
        )

        row: dict[str, float | str] = dict(condition)
        for mode, summary in summaries.items():
            for metric in ROBUSTNESS_METRICS:
                row[f"{mode}_{metric}"] = summary[metric]
        for metric in (
            "roll_rms_deg",
            "pitch_rms_deg",
            "max_abs_roll_deg",
            "max_abs_pitch_deg",
            "board_rp_angular_speed_rms",
            "board_vertical_accel_rms",
        ):
            baseline = geometric[metric]
            candidate = feedback[metric]
            difference = baseline - candidate
            denominator = max(abs(baseline), 1.0e-12)
            row[f"{metric}_improvement_pct"] = 100.0 * difference / denominator
        result_rows.append(row)

    summary_path = log_dir / "robustness_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    roll_improvements = [
        float(row["roll_rms_deg_improvement_pct"]) for row in result_rows
    ]
    pitch_improvements = [
        float(row["pitch_rms_deg_improvement_pct"]) for row in result_rows
    ]
    improved_count = sum(
        roll > 0.0 and pitch > 0.0
        for roll, pitch in zip(roll_improvements, pitch_improvements, strict=True)
    )
    print(
        f"[ROBUSTNESS] feedback improved both roll and pitch RMS in "
        f"{improved_count}/{len(result_rows)} conditions"
    )
    print(
        f"[ROBUSTNESS] roll improvement mean/worst="
        f"{sum(roll_improvements) / len(roll_improvements):.2f}%/"
        f"{min(roll_improvements):.2f}%, pitch mean/worst="
        f"{sum(pitch_improvements) / len(pitch_improvements):.2f}%/"
        f"{min(pitch_improvements):.2f}%"
    )
    print(f"[ROBUSTNESS] summary CSV={summary_path}")


def main() -> None:
    log_dir = Path(args_cli.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _make_environment()
    try:
        if args_cli.suite == "robustness":
            _run_robustness_suite(env, log_dir)
        else:
            _run_single_suite(env, log_dir)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
