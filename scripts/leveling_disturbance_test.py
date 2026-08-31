# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V7.0-B single-environment active-leveling validation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate active lift leveling under vertical disturbances.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Template-Agv-Level-Carry-Direct-v0")
parser.add_argument("--csv_path", type=str, default="logs/leveling_disturbance_test.csv")
parser.add_argument("--duration", type=float, default=8.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac Sim launched; loading task modules...", flush=True)

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import agv_transport.tasks  # noqa: F401

print("[INFO] Task modules loaded.", flush=True)


CSV_FIELDS = (
    "time",
    "disturbance_1",
    "disturbance_2",
    "disturbance_3",
    "lift_height_1",
    "lift_height_2",
    "lift_height_3",
    "lift_top_world_z_1",
    "lift_top_world_z_2",
    "lift_top_world_z_3",
    "support_height_error_rms",
    "board_z",
    "board_roll_deg",
    "board_pitch_deg",
    "board_roll_rate",
    "board_pitch_rate",
)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    # The validation isolates vertical leveling from V6 planar motion guards
    # and procedural terrain excitation.
    env_cfg.enable_front_position_guard = False
    env_cfg.enable_rear_lateral_guard = False
    env_cfg.enable_bumpy_support = False
    env_cfg.episode_length_s = max(float(args_cli.duration) + 2.0, float(env_cfg.episode_length_s))

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()

    actions = torch.zeros(env.action_space.shape, device=raw_env.device)
    actions[:, 0::2] = -1.0
    # Keep the V7.0-B demonstration away from the exact lower travel stop now
    # that the configured transport working height is 30 mm.
    disturbance_final = torch.tensor((0.020, -0.010, 0.010), device=raw_env.device)
    neutral = float(raw_env.cfg.lift_neutral_height)
    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    csv_path = Path(args_cli.csv_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    elapsed = 0.0
    try:
        while simulation_app.is_running() and elapsed < float(args_cli.duration):
            if elapsed < 2.0:
                disturbance = torch.zeros_like(disturbance_final)
                lift_target = torch.full_like(disturbance_final, neutral)
            elif elapsed < 4.0:
                ramp = min(1.0, max(0.0, (elapsed - 2.0) / 2.0))
                disturbance = ramp * disturbance_final
                lift_target = torch.full_like(disturbance_final, neutral)
            else:
                disturbance = disturbance_final
                lift_target = neutral - disturbance_final

            raw_env.base_z_disturbance[0] = disturbance
            raw_env.lift_target_height[0] = lift_target

            with torch.inference_mode():
                env.step(actions)
                elapsed += step_dt

                lift_top_z = torch.stack(
                    [lift.data.root_pos_w[0, 2] for lift in raw_env.lifts]
                ) + 0.5 * float(raw_env.cfg.lift_plate_size[2])
                support_error = torch.sqrt(torch.mean(torch.square(lift_top_z - lift_top_z.mean())))
                roll, pitch, _ = raw_env._get_payload_rpy()
                board_pos = raw_env.payload.data.root_pos_w[0]
                board_ang_vel = raw_env.payload.data.root_ang_vel_w[0]

                values = torch.cat(
                    (
                        raw_env.base_z_disturbance[0],
                        raw_env.lift_height[0],
                        lift_top_z,
                        support_error.reshape(1),
                        board_pos[2].reshape(1),
                        torch.rad2deg(roll[0]).reshape(1),
                        torch.rad2deg(pitch[0]).reshape(1),
                        board_ang_vel[0:2],
                    )
                )
                if not bool(torch.isfinite(values).all()):
                    raise RuntimeError(f"NaN/Inf detected at t={elapsed:.3f} s")
                if bool(
                    torch.any(raw_env.lift_height[0] < float(raw_env.cfg.lift_min_height) - 1.0e-6)
                    or torch.any(raw_env.lift_height[0] > float(raw_env.cfg.lift_max_height) + 1.0e-6)
                ):
                    raise RuntimeError(f"Lift travel limit exceeded at t={elapsed:.3f} s")

                rows.append(dict(zip(CSV_FIELDS, (elapsed, *values.tolist()), strict=True)))
    finally:
        env.close()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    uncompensated = [row for row in rows if 3.75 <= row["time"] < 4.0]
    compensated = [row for row in rows if 7.5 <= row["time"] <= float(args_cli.duration) + step_dt]
    for label, phase_rows in (("uncompensated", uncompensated), ("compensated", compensated)):
        if not phase_rows:
            continue
        mean_support_mm = 1000.0 * sum(row["support_height_error_rms"] for row in phase_rows) / len(phase_rows)
        mean_tilt_deg = sum(
            math.hypot(row["board_roll_deg"], row["board_pitch_deg"]) for row in phase_rows
        ) / len(phase_rows)
        print(f"[RESULT] {label}: support RMS={mean_support_mm:.3f} mm, board tilt={mean_tilt_deg:.3f} deg")
    print(f"[INFO] CSV written to: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
