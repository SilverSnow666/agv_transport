# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V7.6-A residual action-pipeline and zero-residual equivalence checks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate the 3D residual Lift task.")
parser.add_argument(
    "--task", type=str, default="Template-Agv-Level-Residual-Direct-v0"
)
parser.add_argument(
    "--case", choices=("zero", "mapping", "random", "all"), default="all"
)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--log_dir", type=str, default="logs/v7_6_a")
parser.add_argument(
    "--reference_csv",
    type=str,
    default="logs/v7_5/environment_12s/cargo_feedback_leveling.csv",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import agv_transport.tasks  # noqa: F401


ZERO_FIELDS = (
    "time",
    "board_z",
    "board_roll_deg",
    "board_pitch_deg",
    "lift1_height",
    "lift2_height",
    "lift3_height",
    "feedback_height1",
    "feedback_height2",
    "feedback_height3",
    "cargo_x",
    "cargo_y",
    "cargo_z",
    "base_target1",
    "base_target2",
    "base_target3",
    "residual1",
    "residual2",
    "residual3",
)

REFERENCE_FIELDS = (
    "board_z",
    "board_roll_deg",
    "board_pitch_deg",
    "lift1_height",
    "lift2_height",
    "lift3_height",
    "feedback_height1",
    "feedback_height2",
    "feedback_height3",
    "cargo_x",
    "cargo_y",
    "cargo_z",
)

REFERENCE_TOLERANCES = {
    "board_z": 5.0e-5,
    "board_roll_deg": 5.0e-4,
    "board_pitch_deg": 5.0e-4,
    "lift1_height": 5.0e-6,
    "lift2_height": 5.0e-6,
    "lift3_height": 5.0e-6,
    "feedback_height1": 2.0e-5,
    "feedback_height2": 2.0e-5,
    "feedback_height3": 2.0e-5,
    "cargo_x": 5.0e-5,
    "cargo_y": 5.0e-5,
    "cargo_z": 5.0e-5,
}


def _make_environment():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = 0
    env_cfg.residual_domain_randomization = False
    env_cfg.enable_visual_terrain_mesh = args_cli.num_envs == 1
    env_cfg.episode_length_s = max(
        float(args_cli.duration) + 2.0, float(env_cfg.episode_length_s)
    )
    return gym.make(args_cli.task, cfg=env_cfg)


def _zero_row(raw_env, elapsed: float) -> dict[str, float]:
    roll, pitch, _ = raw_env._get_payload_rpy()
    values = torch.cat(
        (
            raw_env.payload.data.root_pos_w[0, 2:3],
            torch.rad2deg(roll[0]).reshape(1),
            torch.rad2deg(pitch[0]).reshape(1),
            raw_env.lift_height[0],
            raw_env.leveling_feedback_height[0],
            raw_env.cargo.data.root_pos_w[0],
            raw_env.base_leveling_target_height[0],
            raw_env.last_residual_height[0],
        )
    )
    return dict(zip(ZERO_FIELDS, (elapsed, *values.tolist()), strict=True))


def _compare_reference(rows: list[dict[str, float]]) -> None:
    reference_path = Path(args_cli.reference_csv).expanduser().resolve()
    if not reference_path.is_file():
        print(f"[WARN] V7.5 reference CSV not found; target equivalence still passed: {reference_path}")
        return
    with reference_path.open(newline="", encoding="utf-8") as csv_file:
        reference_rows = list(csv.DictReader(csv_file))
    if len(reference_rows) != len(rows):
        print(
            f"[WARN] V7.5 reference has {len(reference_rows)} rows but residual run has "
            f"{len(rows)}; rerun with matching duration for trajectory comparison."
        )
        return

    differences = {
        field: max(
            abs(float(actual[field]) - float(reference[field]))
            for actual, reference in zip(rows, reference_rows, strict=True)
        )
        for field in REFERENCE_FIELDS
    }
    normalized = {
        field: differences[field] / REFERENCE_TOLERANCES[field]
        for field in REFERENCE_FIELDS
    }
    limiting_field = max(normalized, key=normalized.get)
    if normalized[limiting_field] > 1.0:
        raise RuntimeError(
            f"Zero-residual trajectory differs from V7.5 E: {limiting_field} "
            f"max abs diff={differences[limiting_field]:.9g}, "
            f"tolerance={REFERENCE_TOLERANCES[limiting_field]:.9g}"
        )
    largest_angle_difference = max(
        differences["board_roll_deg"], differences["board_pitch_deg"]
    )
    largest_position_difference = max(
        differences[field]
        for field in REFERENCE_FIELDS
        if field not in ("board_roll_deg", "board_pitch_deg")
    )
    print(
        "[RESULT] zero-residual V7.5 trajectory equivalence passed: "
        f"max angle diff={largest_angle_difference:.9g} deg, "
        f"max linear diff={1000.0 * largest_position_difference:.6f} mm, "
        f"limiting field={limiting_field} ({normalized[limiting_field]:.3f}x tolerance)"
    )


def _run_zero(env, log_dir: Path) -> None:
    raw_env = env.unwrapped
    raw_env.cfg.residual_domain_randomization = False
    with torch.inference_mode():
        env.reset(seed=0)

    actions = torch.zeros(env.action_space.shape, device=raw_env.device)
    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    elapsed = 0.0
    rows: list[dict[str, float]] = []
    while simulation_app.is_running() and elapsed < float(args_cli.duration):
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        elapsed += step_dt
        target_error = torch.max(
            torch.abs(raw_env.lift_target_height - raw_env.base_leveling_target_height)
        )
        residual_error = torch.max(torch.abs(raw_env.last_residual_height))
        if float(target_error) > 1.0e-8 or float(residual_error) > 1.0e-8:
            raise RuntimeError(
                f"Zero residual changed Lift target at t={elapsed:.3f}: "
                f"target error={float(target_error):.9g}, "
                f"residual={float(residual_error):.9g}"
            )
        if bool(torch.any(terminated)) or bool(torch.any(truncated)):
            raise RuntimeError(f"Zero-residual run terminated early at t={elapsed:.3f} s")
        if args_cli.num_envs == 1:
            rows.append(_zero_row(raw_env, elapsed))

    print(
        f"[RESULT] zero residual target equivalence passed for "
        f"{args_cli.num_envs} env(s), {elapsed:.3f} s"
    )
    if args_cli.num_envs == 1:
        output_path = log_dir / "zero_residual.csv"
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=ZERO_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        _compare_reference(rows)
        roll_rms = math.sqrt(
            sum(row["board_roll_deg"] ** 2 for row in rows) / len(rows)
        )
        pitch_rms = math.sqrt(
            sum(row["board_pitch_deg"] ** 2 for row in rows) / len(rows)
        )
        print(
            f"[RESULT] zero residual: roll RMS={roll_rms:.4f} deg, "
            f"pitch RMS={pitch_rms:.4f} deg, CSV={output_path}"
        )


def _run_mapping(env) -> None:
    raw_env = env.unwrapped
    raw_env.cfg.residual_domain_randomization = False
    limit = float(raw_env.cfg.residual_height_limit)
    for lift_index in range(3):
        with torch.inference_mode():
            env.reset(seed=0)
        actions = torch.zeros(env.action_space.shape, device=raw_env.device)
        actions[:, lift_index] = 1.0
        with torch.inference_mode():
            env.step(actions)
        expected = torch.zeros_like(raw_env.last_residual_height)
        expected[:, lift_index] = limit
        residual_error = torch.max(torch.abs(raw_env.last_residual_height - expected))
        applied_delta = raw_env.lift_target_height - raw_env.base_leveling_target_height
        target_error = torch.max(torch.abs(applied_delta - expected))
        if float(residual_error) > 1.0e-8 or float(target_error) > 1.0e-8:
            raise RuntimeError(
                f"Lift{lift_index + 1} residual mapping failed: "
                f"residual error={float(residual_error):.9g}, "
                f"target error={float(target_error):.9g}"
            )
        print(
            f"[RESULT] action channel {lift_index + 1}: only Lift{lift_index + 1} "
            f"received +{1000.0 * limit:.3f} mm"
        )


def _run_random(env) -> None:
    raw_env = env.unwrapped
    raw_env.cfg.residual_domain_randomization = True
    with torch.inference_mode():
        observation, _ = env.reset(seed=7)
    obs = observation["policy"]
    if obs.shape != (args_cli.num_envs, 33):
        raise RuntimeError(f"Unexpected observation shape: {tuple(obs.shape)}")

    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    elapsed = 0.0
    random_duration = min(float(args_cli.duration), 2.0)
    while simulation_app.is_running() and elapsed < random_duration:
        actions = torch.empty(env.action_space.shape, device=raw_env.device).uniform_(-0.5, 0.5)
        with torch.inference_mode():
            observation, reward, _, _, _ = env.step(actions)
        values = torch.cat((observation["policy"].reshape(-1), reward.reshape(-1)))
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"NaN/Inf in randomized smoke test at t={elapsed:.3f} s")
        elapsed += step_dt

    print(
        f"[RESULT] randomized smoke: envs={args_cli.num_envs}, duration={elapsed:.3f} s, "
        f"speed=[{float(raw_env.randomized_target_speed.min()):.3f}, "
        f"{float(raw_env.randomized_target_speed.max()):.3f}] m/s, "
        f"terrain amplitude=[{1000.0 * float(raw_env.randomized_terrain_amplitude.min()):.1f}, "
        f"{1000.0 * float(raw_env.randomized_terrain_amplitude.max()):.1f}] mm, "
        f"Cargo mass=[{float(raw_env.randomized_cargo_mass.min()):.2f}, "
        f"{float(raw_env.randomized_cargo_mass.max()):.2f}] kg"
    )


def main() -> None:
    if args_cli.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    log_dir = Path(args_cli.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _make_environment()
    try:
        if args_cli.case in ("zero", "all"):
            _run_zero(env, log_dir)
        if args_cli.case in ("mapping", "all"):
            _run_mapping(env)
        if args_cli.case in ("random", "all"):
            _run_random(env)
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
