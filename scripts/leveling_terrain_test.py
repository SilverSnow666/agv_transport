# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V7.1 bumpy-terrain validation for geometric active leveling."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Compare fixed and geometric lift targets on bumpy terrain.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Template-Agv-Level-Carry-Direct-v0")
parser.add_argument("--mode", choices=("no_leveling", "geometric", "both"), default="both")
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--target_speed", type=float, default=0.10)
parser.add_argument("--log_dir", type=str, default="logs/v7_1")
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
    "lift1_height", "lift2_height", "lift3_height",
    "lift1_velocity", "lift2_velocity", "lift3_velocity",
    "lift_top_world_z1", "lift_top_world_z2", "lift_top_world_z3",
    "support_height_error_rms",
    "board_z", "board_roll_deg", "board_pitch_deg",
    "board_roll_rate", "board_pitch_rate",
    "lift1_saturated", "lift2_saturated", "lift3_saturated",
)


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


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


def _terrain_heights(raw_env) -> torch.Tensor:
    env_xy = raw_env.scene.env_origins[0, :2]
    agv_xy = torch.stack([agv.data.root_pos_w[0, :2] for agv in raw_env.agvs])
    return raw_env._terrain_height(agv_xy - env_xy)


def _summarize(rows: list[dict[str, float]], step_dt: float) -> dict[str, float]:
    result = {
        "roll_rms_deg": _rms([row["board_roll_deg"] for row in rows]),
        "pitch_rms_deg": _rms([row["board_pitch_deg"] for row in rows]),
        "max_abs_roll_deg": max(abs(row["board_roll_deg"]) for row in rows),
        "max_abs_pitch_deg": max(abs(row["board_pitch_deg"]) for row in rows),
        "support_height_rms_mm": 1000.0 * _rms([row["support_height_error_rms"] for row in rows]),
    }
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
    raw_env.base_z_disturbance.zero_()

    actions, target_speed, linear_action = _make_actions(raw_env, args_cli.target_speed)
    neutral = float(raw_env.cfg.lift_neutral_height)
    lift_min = float(raw_env.cfg.lift_min_height)
    lift_max = float(raw_env.cfg.lift_max_height)
    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    rows: list[dict[str, float]] = []
    elapsed = 0.0
    initial_xy = torch.stack([agv.data.root_pos_w[0, :2] for agv in raw_env.agvs]).clone()
    print(
        f"[INFO] {mode}: target speed={target_speed:.4f} m/s, action={linear_action:.6f}, "
        f"duration={args_cli.duration:.2f} s"
    )

    while simulation_app.is_running() and elapsed < float(args_cli.duration):
        terrain_before = _terrain_heights(raw_env)
        if mode == "geometric":
            raw_target = neutral - (terrain_before - terrain_before.mean())
        else:
            raw_target = torch.full_like(terrain_before, neutral)
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

        terrain_z = _terrain_heights(raw_env)
        agv_z = torch.stack([agv.data.root_pos_w[0, 2] for agv in raw_env.agvs])
        lift_top_z = torch.stack([lift.data.root_pos_w[0, 2] for lift in raw_env.lifts])
        lift_top_z += 0.5 * float(raw_env.cfg.lift_plate_size[2])
        support_error = torch.sqrt(torch.mean(torch.square(lift_top_z - lift_top_z.mean())))
        roll, pitch, _ = raw_env._get_payload_rpy()
        board_pos = raw_env.payload.data.root_pos_w[0]
        board_ang_vel = raw_env.payload.data.root_ang_vel_w[0]

        values = torch.cat((
            agv_z, terrain_z, raw_env.lift_height[0], raw_env.lift_velocity[0], lift_top_z,
            support_error.reshape(1), board_pos[2].reshape(1),
            torch.rad2deg(roll[0]).reshape(1), torch.rad2deg(pitch[0]).reshape(1),
            board_ang_vel[0:2], saturated.float(),
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

    csv_name = "geometric_leveling.csv" if mode == "geometric" else "no_leveling.csv"
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
