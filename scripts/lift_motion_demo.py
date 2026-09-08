# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Visual/physical Lift motion demo: 0 -> target -> 0 mm.

The three hidden physical Lift plates move kinematically and physically push the
free dynamic Board. The visual telescopic heads/guide posts follow the same
``lift_height`` values, so the rendered mechanism and the real support surface
stay synchronized.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demonstrate synchronized Lift/Board vertical motion.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Template-Agv-Level-Carry-Direct-v0")
parser.add_argument("--start_height_mm", type=float, default=0.0)
parser.add_argument("--target_height_mm", type=float, default=30.0)
parser.add_argument("--lift_speed_mm_s", type=float, default=10.0)
parser.add_argument("--settle_duration", type=float, default=1.0)
parser.add_argument("--hold_duration", type=float, default=1.5)
parser.add_argument("--log_path", type=str, default="logs/v7_lift_motion/lift_motion.csv")
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


FIELDS = (
    "time", "phase",
    "lift1_mm", "lift2_mm", "lift3_mm",
    "column1_mm", "column2_mm", "column3_mm",
    "plate_top1_z", "plate_top2_z", "plate_top3_z",
    "board_center_z", "board_bottom_z", "plate_board_gap_mm",
    "board_vz",
)


def stop_actions(raw):
    actions = torch.zeros(raw.action_space.shape, device=raw.device)
    if bool(getattr(raw.cfg, "forward_only_linear_speed", True)):
        actions[:, 0::2] = -1.0
    return actions


def lift_top_z(raw) -> torch.Tensor:
    values = []
    half = 0.5 * float(raw.cfg.lift_plate_size[2])
    for lift in raw.lifts:
        local_z = raw._quat_rotate_z(lift.data.root_quat_w)
        values.append(lift.data.root_pos_w[:, 2] + half * local_z[:, 2])
    return torch.stack(values, dim=1)


def append_row(raw, rows, elapsed, phase):
    heights = 1000.0 * raw.lift_height[0]
    columns = 1000.0 * raw.lift_visual_column_lengths()[0]
    tops = lift_top_z(raw)[0]
    board_z = float(raw.payload.data.root_pos_w[0, 2])
    board_bottom = board_z - 0.5 * float(raw.cfg.payload_size[2])
    gap_mm = 1000.0 * (board_bottom - float(tops.mean()))
    row = {
        "time": elapsed,
        "phase": phase,
        "lift1_mm": float(heights[0]),
        "lift2_mm": float(heights[1]),
        "lift3_mm": float(heights[2]),
        "column1_mm": float(columns[0]),
        "column2_mm": float(columns[1]),
        "column3_mm": float(columns[2]),
        "plate_top1_z": float(tops[0]),
        "plate_top2_z": float(tops[1]),
        "plate_top3_z": float(tops[2]),
        "board_center_z": board_z,
        "board_bottom_z": board_bottom,
        "plate_board_gap_mm": gap_mm,
        "board_vz": float(raw.payload.data.root_lin_vel_w[0, 2]),
    }
    values = torch.tensor([float(v) for key, v in row.items() if key != "phase"], device=raw.device)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"NaN/Inf detected during {phase} at t={elapsed:.3f}s")
    rows.append(row)


def run_for(env, actions, rows, elapsed, duration, phase):
    raw = env.unwrapped
    dt = float(raw.cfg.sim.dt) * int(raw.cfg.decimation)
    end = elapsed + max(float(duration), 0.0)
    next_print = elapsed
    while simulation_app.is_running() and elapsed < end:
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        elapsed += dt
        append_row(raw, rows, elapsed, phase)
        if elapsed >= next_print:
            last = rows[-1]
            print(
                f"[MOTION] {phase:>6} t={elapsed:6.3f}s "
                f"Lift={last['lift1_mm']:6.2f}/{last['lift2_mm']:6.2f}/{last['lift3_mm']:6.2f} mm, "
                f"Board Z={1000.0 * last['board_center_z']:7.2f} mm, "
                f"gap={last['plate_board_gap_mm']:6.2f} mm"
            )
            next_print += 0.25
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError(
                f"Unexpected environment reset during {phase} at t={elapsed:.3f}s "
                f"(terminated={bool(terminated[0])}, truncated={bool(truncated[0])})"
            )
    return elapsed


def run_until_target(env, actions, rows, elapsed, target, phase, timeout):
    raw = env.unwrapped
    dt = float(raw.cfg.sim.dt) * int(raw.cfg.decimation)
    end = elapsed + timeout
    next_print = elapsed
    tolerance = 0.0005
    while simulation_app.is_running() and elapsed < end:
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        elapsed += dt
        append_row(raw, rows, elapsed, phase)
        error = torch.max(torch.abs(raw.lift_height[0] - target))
        if elapsed >= next_print:
            last = rows[-1]
            print(
                f"[MOTION] {phase:>6} t={elapsed:6.3f}s "
                f"Lift={last['lift1_mm']:6.2f}/{last['lift2_mm']:6.2f}/{last['lift3_mm']:6.2f} mm, "
                f"Board Z={1000.0 * last['board_center_z']:7.2f} mm, "
                f"gap={last['plate_board_gap_mm']:6.2f} mm"
            )
            next_print += 0.25
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError(f"Unexpected reset during {phase} at t={elapsed:.3f}s")
        if float(error) <= tolerance:
            return elapsed
    raise RuntimeError(f"Lift failed to reach {1000.0 * target:.1f} mm during {phase}")


def main():
    start_h = 0.001 * float(args_cli.start_height_mm)
    target_h = 0.001 * float(args_cli.target_height_mm)
    speed = 0.001 * float(args_cli.lift_speed_mm_s)
    if speed <= 0.0:
        raise ValueError("--lift_speed_mm_s must be positive")

    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    cfg.enable_bumpy_support = False
    cfg.enable_cargo = False
    cfg.enable_virtual_friction_carry = False
    cfg.enable_front_position_guard = False
    cfg.enable_rear_lateral_guard = False
    cfg.critical_support_contacts = 0.0
    cfg.payload_min_z = -100.0
    cfg.tip_roll_pitch_threshold = 3.141592653589793
    cfg.episode_length_s = max(float(cfg.episode_length_s), 30.0)
    cfg.max_lift_speed = speed

    lo = float(cfg.lift_min_height)
    hi = float(cfg.lift_max_height)
    for name, value in (("start", start_h), ("target", target_h)):
        if not lo <= value <= hi:
            raise ValueError(
                f"{name} Lift height {1000.0 * value:.1f} mm is outside "
                f"[{1000.0 * lo:.1f}, {1000.0 * hi:.1f}] mm"
            )

    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    if not hasattr(raw, "set_lift_stack_initial_height"):
        raise RuntimeError("Task is not using the telescopic Lift visual environment")

    try:
        with torch.inference_mode():
            env.reset(seed=0)
            raw.set_lift_stack_initial_height(start_h)
        raw.sim.set_camera_view(eye=(1.25, -1.45, 0.32), target=(0.10, 0.0, 0.13))

        actions = stop_actions(raw)
        rows = []
        elapsed = 0.0
        append_row(raw, rows, elapsed, "start")

        print(
            f"[INFO] true Lift-motion demo: start={1000.0 * start_h:.1f} mm, "
            f"target={1000.0 * target_h:.1f} mm, speed={1000.0 * speed:.1f} mm/s"
        )
        print("[INFO] virtual carry/damping is OFF; Board vertical motion comes from PhysX contact + gravity.")

        elapsed = run_for(env, actions, rows, elapsed, args_cli.settle_duration, "settle")

        raw.lift_target_height.fill_(target_h)
        travel = abs(target_h - start_h)
        timeout = max(3.0, 2.5 * travel / speed + 2.0)
        elapsed = run_until_target(env, actions, rows, elapsed, target_h, "raise", timeout)
        elapsed = run_for(env, actions, rows, elapsed, args_cli.hold_duration, "hold")

        raw.lift_target_height.fill_(start_h)
        elapsed = run_until_target(env, actions, rows, elapsed, start_h, "lower", timeout)
        elapsed = run_for(env, actions, rows, elapsed, args_cli.hold_duration, "finish")

        output = Path(args_cli.log_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        start_board = rows[0]["board_center_z"]
        max_board = max(row["board_center_z"] for row in rows)
        board_rise_mm = 1000.0 * (max_board - start_board)
        max_gap = max(abs(row["plate_board_gap_mm"]) for row in rows)
        print(f"[RESULT] Board physical rise={board_rise_mm:.2f} mm")
        print(f"[RESULT] max |physical plate top -> Board bottom gap|={max_gap:.2f} mm")
        print(f"[RESULT] CSV={output}")
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
