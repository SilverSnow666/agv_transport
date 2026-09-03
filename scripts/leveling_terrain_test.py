# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V7.2 free-Cargo stability comparison on the V7.1.1 terrain baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate terrain-following AGV attitude and geometric leveling.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Template-Agv-Level-Carry-Direct-v0")
parser.add_argument("--mode", choices=("no_leveling", "geometric", "both"), default="both")
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--target_speed", type=float, default=0.10)
parser.add_argument("--log_dir", type=str, default="logs/v7_2")
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


def _summarize(rows: list[dict[str, float]], step_dt: float) -> dict[str, float]:
    result = {
        "roll_rms_deg": _rms([row["board_roll_deg"] for row in rows]),
        "pitch_rms_deg": _rms([row["board_pitch_deg"] for row in rows]),
        "max_abs_roll_deg": max(abs(row["board_roll_deg"]) for row in rows),
        "max_abs_pitch_deg": max(abs(row["board_pitch_deg"]) for row in rows),
        "support_height_rms_mm": 1000.0 * _rms([row["support_height_error_rms"] for row in rows]),
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
    env_cfg.enable_front_position_guard = False
    env_cfg.enable_rear_lateral_guard = False
    env_cfg.enable_bumpy_support = True
    env_cfg.enable_cargo = True
    # The AGVs/lifts are kinematic proxies.  Reuse the existing V6 no-slip
    # coupling so the dynamic board follows the translating supports; without
    # it the board stays behind and the comparison terminates on support loss.
    env_cfg.enable_virtual_friction_carry = True
    env_cfg.episode_length_s = max(float(args_cli.duration) + 2.0, float(env_cfg.episode_length_s))
    env = gym.make(args_cli.task, cfg=env_cfg)
    return env


def run_mode(env, mode: str, log_dir: Path) -> dict[str, float]:
    raw_env = env.unwrapped
    with torch.inference_mode():
        env.reset(seed=0)
    if raw_env.cargo is None:
        raise RuntimeError("V7.2 Cargo was not created")
    raw_env.base_z_disturbance.zero_()

    actions, target_speed, linear_action = _make_actions(raw_env, args_cli.target_speed)
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
        f"[INFO] {mode}: target speed={target_speed:.4f} m/s, action={linear_action:.6f}, "
        f"duration={args_cli.duration:.2f} s"
    )

    while simulation_app.is_running() and elapsed < float(args_cli.duration):
        if mode == "geometric":
            # V7.1.1 controls the three actual top reference-center world
            # heights.  A local-axis displacement changes world z by the
            # corresponding AGV local +Z component.
            top_before = _lift_top_centers(raw_env)
            top_error = top_before[:, 2].mean() - top_before[:, 2]
            agv_quat = torch.stack([agv.data.root_quat_w[0] for agv in raw_env.agvs])
            axis_z = raw_env._quat_rotate_z(agv_quat)[:, 2].clamp_min(0.25)
            raw_target = raw_env.lift_height[0] + top_error / axis_z
        else:
            raw_target = torch.full_like(raw_env.lift_height[0], neutral)
        saturated = (raw_target < lift_min) | (raw_target > lift_max)
        raw_env.lift_target_height[0] = torch.clamp(raw_target, min=lift_min, max=lift_max)
        raw_env.base_z_disturbance[0] = 0.0

        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
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
            board_ang_vel[0:2], saturated.float(), torch.rad2deg(board_yaw[0]).reshape(1),
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

    csv_name = "cargo_geometric_leveling.csv" if mode == "geometric" else "cargo_no_leveling.csv"
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


def main() -> None:
    log_dir = Path(args_cli.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    modes = ("no_leveling", "geometric") if args_cli.mode == "both" else (args_cli.mode,)
    env = _make_environment()
    try:
        summaries = {mode: run_mode(env, mode, log_dir) for mode in modes}
    finally:
        env.close()
    if args_cli.mode == "both":
        baseline = summaries["no_leveling"]
        geometric = summaries["geometric"]
        print("[COMPARISON] geometric improvement relative to no_leveling:")
        for key, label in (
            ("roll_rms_deg", "board roll RMS"),
            ("pitch_rms_deg", "board pitch RMS"),
            ("max_abs_roll_deg", "max |roll|"),
            ("max_abs_pitch_deg", "max |pitch|"),
            ("support_height_rms_mm", "support height RMS"),
            ("cargo_relative_xy_rms_mm", "Cargo relative XY RMS"),
            ("cargo_max_relative_xy_mm", "Cargo max relative XY"),
            ("cargo_cumulative_slip_mm", "Cargo cumulative slip"),
            ("cargo_roll_rms_deg", "Cargo roll RMS"),
            ("cargo_pitch_rms_deg", "Cargo pitch RMS"),
            ("cargo_ang_speed_rms", "Cargo angular-speed RMS"),
        ):
            improvement = 100.0 * (baseline[key] - geometric[key]) / max(baseline[key], 1.0e-12)
            print(f"[COMPARISON] {label}: {improvement:+.2f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
