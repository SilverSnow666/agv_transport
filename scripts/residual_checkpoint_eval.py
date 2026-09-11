# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare zero residual (E) with a deterministic PPO checkpoint (F)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate a residual PPO checkpoint against the V7.5 zero-residual baseline."
)
parser.add_argument(
    "--task", type=str, default="Template-Agv-Level-Residual-Direct-v0"
)
parser.add_argument(
    "--case",
    choices=("nominal", "fast", "rough", "heavy", "offset", "combined", "all"),
    default="all",
)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--seed", type=int, default=137)
parser.add_argument("--phase_x", type=float, default=1.17)
parser.add_argument("--phase_y", type=float, default=-2.03)
parser.add_argument("--log_dir", type=str, default="logs/v7_6_c")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Hydra must only see arguments that it owns.
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import agv_transport.tasks  # noqa: F401


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    speed: float
    terrain_amplitude: float
    cargo_mass: float
    cargo_offset_x: float = 0.0
    cargo_offset_y: float = 0.0


CASES = {
    "nominal": EvaluationCase("nominal", 0.10, 0.030, 4.0),
    "fast": EvaluationCase("fast", 0.18, 0.030, 4.0),
    "rough": EvaluationCase("rough", 0.10, 0.050, 4.0),
    "heavy": EvaluationCase("heavy", 0.10, 0.030, 8.0),
    "offset": EvaluationCase("offset", 0.10, 0.030, 6.0, 0.030, 0.0),
    "combined": EvaluationCase("combined", 0.18, 0.050, 8.0, 0.030, 0.0),
}

COMPARISON_METRICS = (
    "board_roll_rms_deg",
    "board_pitch_rms_deg",
    "board_max_abs_roll_deg",
    "board_max_abs_pitch_deg",
    "board_rp_angular_velocity_rms_rad_s",
    "board_vertical_acceleration_rms_m_s2",
    "board_z_range_mm",
    "cargo_relative_xy_rms_mm",
    "cargo_cumulative_slip_mm",
    "cargo_relative_velocity_rms_m_s",
    "cargo_angular_velocity_rms_rad_s",
    "lift_velocity_rms_m_s",
)

DISPLAY_METRICS = (
    "board_roll_rms_deg",
    "board_pitch_rms_deg",
    "board_rp_angular_velocity_rms_rad_s",
    "board_vertical_acceleration_rms_m_s2",
    "cargo_relative_xy_rms_mm",
    "cargo_cumulative_slip_mm",
    "lift_velocity_rms_m_s",
    "residual_rms_mm",
)


def _resolve_checkpoint() -> Path:
    if args_cli.checkpoint:
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    root = Path("logs/skrl/agv_level_residual_direct").resolve()
    candidates = list(root.glob("*/checkpoints/best_agent.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best_agent.pt found below {root}; pass --checkpoint explicitly"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _selected_cases() -> list[EvaluationCase]:
    if args_cli.case == "all":
        return list(CASES.values())
    return [CASES[args_cli.case]]


def _configure_case(raw_env, case: EvaluationCase) -> None:
    cfg = raw_env.cfg
    cfg.residual_domain_randomization = True
    cfg.residual_speed_range = (case.speed, case.speed)
    cfg.residual_terrain_amplitude_range = (
        case.terrain_amplitude,
        case.terrain_amplitude,
    )
    cfg.residual_terrain_phase_x_range = (args_cli.phase_x, args_cli.phase_x)
    cfg.residual_terrain_phase_y_range = (args_cli.phase_y, args_cli.phase_y)
    cfg.residual_cargo_mass_range = (case.cargo_mass, case.cargo_mass)
    cfg.residual_cargo_offset_x_range = (case.cargo_offset_x, case.cargo_offset_x)
    cfg.residual_cargo_offset_y_range = (case.cargo_offset_y, case.cargo_offset_y)


def _policy_actions(agent, observation: torch.Tensor) -> torch.Tensor:
    outputs = agent.act(observation, timestep=0, timesteps=0)
    actions = outputs[-1].get("mean_actions", outputs[0])
    return torch.clamp(actions, -1.0, 1.0)


def _initial_state_signature(raw_env, case: EvaluationCase) -> torch.Tensor:
    expected = (
        (raw_env.randomized_target_speed[0], case.speed, "speed"),
        (
            raw_env.randomized_terrain_amplitude[0],
            case.terrain_amplitude,
            "terrain amplitude",
        ),
        (raw_env.randomized_terrain_phase_x[0], args_cli.phase_x, "phase x"),
        (raw_env.randomized_terrain_phase_y[0], args_cli.phase_y, "phase y"),
        (raw_env.randomized_cargo_mass[0], case.cargo_mass, "Cargo mass"),
        (raw_env.randomized_cargo_offset_xy[0, 0], case.cargo_offset_x, "Cargo offset x"),
        (raw_env.randomized_cargo_offset_xy[0, 1], case.cargo_offset_y, "Cargo offset y"),
    )
    for actual, target, name in expected:
        if abs(float(actual) - float(target)) > 1.0e-6:
            raise RuntimeError(
                f"{case.name} reset {name} mismatch: actual={float(actual):.9g}, "
                f"expected={float(target):.9g}"
            )
    clearance_error = torch.max(
        torch.abs(
            raw_env.last_reset_support_gap[0]
            - float(raw_env.cfg.board_support_clearance)
        )
    )
    if float(clearance_error) > 1.0e-5:
        raise RuntimeError(
            f"{case.name} reset support-gap error="
            f"{1000.0 * float(clearance_error):.6f} mm"
        )

    states = [
        raw_env.payload.data.root_state_w[0],
        raw_env.cargo.data.root_state_w[0],
        raw_env.lift_height[0],
        raw_env.lift_velocity[0],
    ]
    states.extend(agv.data.root_state_w[0] for agv in raw_env.agvs)
    states.extend(lift.data.root_state_w[0] for lift in raw_env.lifts)
    return torch.cat(states).detach().clone()


def _state_row(
    raw_env,
    *,
    case: EvaluationCase,
    controller: str,
    elapsed: float,
    reward: float,
    vertical_acceleration: float,
    cargo_cumulative_slip: float,
    residual_rate: torch.Tensor,
) -> dict[str, float | int | str]:
    roll, pitch, yaw = raw_env._get_payload_rpy()
    board_quat_inverse = raw_env.payload.data.root_quat_w.clone()
    board_quat_inverse[:, 1:4] *= -1.0
    board_ang_vel_local = raw_env._quat_rotate_vector(
        board_quat_inverse, raw_env.payload.data.root_ang_vel_w
    )
    (
        cargo_position,
        cargo_velocity,
        cargo_roll,
        cargo_pitch,
        cargo_angular_speed,
    ) = raw_env._cargo_relative_state()
    cargo_slip = cargo_position[:, :2] - raw_env.cargo_initial_relative_xy
    (
        board_dropped,
        board_tipped,
        support_lost,
        _out_of_bounds,
        cargo_dropped,
        cargo_tipped,
        contact_count,
    ) = raw_env._failure_state()

    board_position = raw_env.payload.data.root_pos_w[0]
    board_velocity = raw_env.payload.data.root_lin_vel_w[0]
    board_angular_velocity = board_ang_vel_local[0]
    cargo_position = cargo_position[0]
    cargo_velocity = cargo_velocity[0]
    lift_height = raw_env.lift_height[0]
    lift_velocity = raw_env.lift_velocity[0]
    residual = raw_env.last_residual_height[0]

    return {
        "time_s": elapsed,
        "case": case.name,
        "controller": controller,
        "seed": args_cli.seed,
        "speed_m_s": case.speed,
        "terrain_amplitude_m": case.terrain_amplitude,
        "terrain_phase_x_rad": args_cli.phase_x,
        "terrain_phase_y_rad": args_cli.phase_y,
        "cargo_mass_kg": case.cargo_mass,
        "cargo_offset_x_m": case.cargo_offset_x,
        "cargo_offset_y_m": case.cargo_offset_y,
        "reward": reward,
        "board_x_m": float(board_position[0]),
        "board_y_m": float(board_position[1]),
        "board_z_m": float(board_position[2]),
        "board_roll_deg": math.degrees(float(roll[0])),
        "board_pitch_deg": math.degrees(float(pitch[0])),
        "board_yaw_deg": math.degrees(float(yaw[0])),
        "board_vx_m_s": float(board_velocity[0]),
        "board_vy_m_s": float(board_velocity[1]),
        "board_vz_m_s": float(board_velocity[2]),
        "board_roll_rate_rad_s": float(board_angular_velocity[0]),
        "board_pitch_rate_rad_s": float(board_angular_velocity[1]),
        "board_yaw_rate_rad_s": float(board_angular_velocity[2]),
        "board_vertical_acceleration_m_s2": vertical_acceleration,
        "cargo_relative_x_m": float(cargo_position[0]),
        "cargo_relative_y_m": float(cargo_position[1]),
        "cargo_relative_z_m": float(cargo_position[2]),
        "cargo_slip_m": float(torch.linalg.norm(cargo_slip[0])),
        "cargo_cumulative_slip_m": cargo_cumulative_slip,
        "cargo_relative_vx_m_s": float(cargo_velocity[0]),
        "cargo_relative_vy_m_s": float(cargo_velocity[1]),
        "cargo_relative_speed_m_s": float(torch.linalg.norm(cargo_velocity[:2])),
        "cargo_relative_roll_deg": math.degrees(float(cargo_roll[0])),
        "cargo_relative_pitch_deg": math.degrees(float(cargo_pitch[0])),
        "cargo_relative_angular_speed_rad_s": float(cargo_angular_speed[0]),
        "lift1_height_m": float(lift_height[0]),
        "lift2_height_m": float(lift_height[1]),
        "lift3_height_m": float(lift_height[2]),
        "lift1_velocity_m_s": float(lift_velocity[0]),
        "lift2_velocity_m_s": float(lift_velocity[1]),
        "lift3_velocity_m_s": float(lift_velocity[2]),
        "residual1_m": float(residual[0]),
        "residual2_m": float(residual[1]),
        "residual3_m": float(residual[2]),
        "residual_rate1_m_s": float(residual_rate[0]),
        "residual_rate2_m_s": float(residual_rate[1]),
        "residual_rate3_m_s": float(residual_rate[2]),
        "residual_at_limit_count": int(
            (
                torch.abs(residual)
                >= float(raw_env.cfg.residual_height_limit) - 1.0e-7
            ).sum()
        ),
        "residual_saturated_count": int(raw_env.last_residual_saturated[0].sum()),
        "support_count_proxy": float(contact_count[0]),
        "support_lost_proxy": int(support_lost[0]),
        "board_dropped": int(board_dropped[0]),
        "board_tipped": int(board_tipped[0]),
        "cargo_dropped": int(cargo_dropped[0]),
        "cargo_tipped": int(cargo_tipped[0]),
    }


def _rms(rows: list[dict], key: str) -> float:
    return math.sqrt(sum(float(row[key]) ** 2 for row in rows) / len(rows))


def _summarize(
    rows: list[dict],
    *,
    case: EvaluationCase,
    controller: str,
    requested_duration: float,
    early_termination: bool,
) -> dict[str, float | int | str]:
    if not rows:
        raise RuntimeError(f"No samples collected for {case.name}/{controller}")
    initial_x = float(rows[0]["board_x_m"])
    lift_ranges = [
        1000.0
        * (
            max(float(row[f"lift{index}_height_m"]) for row in rows)
            - min(float(row[f"lift{index}_height_m"]) for row in rows)
        )
        for index in range(1, 4)
    ]
    summary: dict[str, float | int | str] = {
        "case": case.name,
        "controller": controller,
        "seed": args_cli.seed,
        "speed_m_s": case.speed,
        "terrain_amplitude_m": case.terrain_amplitude,
        "terrain_phase_x_rad": args_cli.phase_x,
        "terrain_phase_y_rad": args_cli.phase_y,
        "cargo_mass_kg": case.cargo_mass,
        "cargo_offset_x_m": case.cargo_offset_x,
        "cargo_offset_y_m": case.cargo_offset_y,
        "requested_duration_s": requested_duration,
        "actual_duration_s": float(rows[-1]["time_s"]),
        "samples": len(rows),
        "early_termination": int(early_termination),
        "mean_reward": sum(float(row["reward"]) for row in rows) / len(rows),
        "board_roll_rms_deg": _rms(rows, "board_roll_deg"),
        "board_pitch_rms_deg": _rms(rows, "board_pitch_deg"),
        "board_max_abs_roll_deg": max(abs(float(row["board_roll_deg"])) for row in rows),
        "board_max_abs_pitch_deg": max(abs(float(row["board_pitch_deg"])) for row in rows),
        "board_rp_angular_velocity_rms_rad_s": math.sqrt(
            sum(
                float(row["board_roll_rate_rad_s"]) ** 2
                + float(row["board_pitch_rate_rad_s"]) ** 2
                for row in rows
            )
            / len(rows)
        ),
        "board_vertical_acceleration_rms_m_s2": _rms(
            rows, "board_vertical_acceleration_m_s2"
        ),
        "board_z_range_mm": 1000.0
        * (
            max(float(row["board_z_m"]) for row in rows)
            - min(float(row["board_z_m"]) for row in rows)
        ),
        "board_forward_displacement_m": float(rows[-1]["board_x_m"]) - initial_x,
        "cargo_relative_xy_rms_mm": 1000.0 * _rms(rows, "cargo_slip_m"),
        "cargo_cumulative_slip_mm": 1000.0
        * float(rows[-1]["cargo_cumulative_slip_m"]),
        "cargo_relative_velocity_rms_m_s": _rms(rows, "cargo_relative_speed_m_s"),
        "cargo_angular_velocity_rms_rad_s": _rms(
            rows, "cargo_relative_angular_speed_rad_s"
        ),
        "lift_velocity_rms_m_s": math.sqrt(
            sum(
                sum(float(row[f"lift{index}_velocity_m_s"]) ** 2 for index in range(1, 4))
                / 3.0
                for row in rows
            )
            / len(rows)
        ),
        "lift1_travel_range_mm": lift_ranges[0],
        "lift2_travel_range_mm": lift_ranges[1],
        "lift3_travel_range_mm": lift_ranges[2],
        "lift_max_travel_range_mm": max(lift_ranges),
        "residual_rms_mm": 1000.0
        * math.sqrt(
            sum(
                sum(float(row[f"residual{index}_m"]) ** 2 for index in range(1, 4))
                / 3.0
                for row in rows
            )
            / len(rows)
        ),
        "residual_max_abs_mm": 1000.0
        * max(
            abs(float(row[f"residual{index}_m"]))
            for row in rows
            for index in range(1, 4)
        ),
        "residual_rate_rms_mm_s": 1000.0
        * math.sqrt(
            sum(
                sum(float(row[f"residual_rate{index}_m_s"]) ** 2 for index in range(1, 4))
                / 3.0
                for row in rows
            )
            / len(rows)
        ),
        "residual_at_limit_fraction": sum(
            float(row["residual_at_limit_count"]) / 3.0 for row in rows
        )
        / len(rows),
        "residual_saturation_fraction": sum(
            float(row["residual_saturated_count"]) / 3.0 for row in rows
        )
        / len(rows),
        "mean_support_count_proxy": sum(float(row["support_count_proxy"]) for row in rows)
        / len(rows),
        "support_loss_fraction_proxy": sum(float(row["support_lost_proxy"]) for row in rows)
        / len(rows),
        "board_dropped": max(int(row["board_dropped"]) for row in rows),
        "board_tipped": max(int(row["board_tipped"]) for row in rows),
        "cargo_dropped": max(int(row["cargo_dropped"]) for row in rows),
        "cargo_tipped": max(int(row["cargo_tipped"]) for row in rows),
    }
    return summary


def _run_controller(
    gym_env,
    wrapped_env,
    raw_env,
    agent,
    case: EvaluationCase,
    controller: str,
) -> tuple[list[dict], dict, torch.Tensor]:
    _configure_case(raw_env, case)
    with torch.inference_mode():
        observations, _ = gym_env.reset(seed=args_cli.seed)
        observation = flatten_tensorized_space(
            tensorize_space(wrapped_env.observation_space, observations["policy"])
        )
    initial_signature = _initial_state_signature(raw_env, case)

    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    previous_vertical_velocity = float(raw_env.payload.data.root_lin_vel_w[0, 2])
    previous_cargo_xy = raw_env._cargo_relative_state()[0][0, :2].clone()
    previous_residual = torch.zeros(3, device=raw_env.device)
    elapsed = 0.0
    cumulative_slip = 0.0
    rows: list[dict] = []
    early_termination = False

    while simulation_app.is_running() and elapsed < float(args_cli.duration):
        with torch.inference_mode():
            if controller == "E_zero":
                actions = torch.zeros((1, 3), device=raw_env.device)
            else:
                actions = _policy_actions(agent, observation)
            observation, reward, terminated, truncated, _ = wrapped_env.step(actions)

        elapsed += step_dt
        current_vertical_velocity = float(raw_env.payload.data.root_lin_vel_w[0, 2])
        vertical_acceleration = (
            current_vertical_velocity - previous_vertical_velocity
        ) / step_dt
        previous_vertical_velocity = current_vertical_velocity

        current_cargo_xy = raw_env._cargo_relative_state()[0][0, :2]
        cumulative_slip += float(torch.linalg.norm(current_cargo_xy - previous_cargo_xy))
        previous_cargo_xy = current_cargo_xy.clone()

        current_residual = raw_env.last_residual_height[0].clone()
        residual_rate = (current_residual - previous_residual) / step_dt
        previous_residual = current_residual
        rows.append(
            _state_row(
                raw_env,
                case=case,
                controller=controller,
                elapsed=elapsed,
                reward=float(reward[0]),
                vertical_acceleration=vertical_acceleration,
                cargo_cumulative_slip=cumulative_slip,
                residual_rate=residual_rate,
            )
        )

        if bool(terminated[0]) or bool(truncated[0]):
            early_termination = elapsed + 0.5 * step_dt < float(args_cli.duration)
            break

    summary = _summarize(
        rows,
        case=case,
        controller=controller,
        requested_duration=float(args_cli.duration),
        early_termination=early_termination,
    )
    return rows, summary, initial_signature


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _comparisons(summaries: list[dict]) -> list[dict[str, float | str]]:
    indexed = {
        (str(summary["case"]), str(summary["controller"])): summary
        for summary in summaries
    }
    rows: list[dict[str, float | str]] = []
    for case in _selected_cases():
        baseline = indexed[(case.name, "E_zero")]
        policy = indexed[(case.name, "F_ppo")]
        for metric in COMPARISON_METRICS:
            baseline_value = float(baseline[metric])
            policy_value = float(policy[metric])
            absolute = baseline_value - policy_value
            percent = (
                100.0 * absolute / abs(baseline_value)
                if abs(baseline_value) > 1.0e-12
                else math.nan
            )
            rows.append(
                {
                    "case": case.name,
                    "seed": args_cli.seed,
                    "checkpoint": str(baseline["checkpoint"]),
                    "metric": metric,
                    "e_zero": baseline_value,
                    "f_ppo": policy_value,
                    "absolute_improvement": absolute,
                    "percent_improvement": percent,
                }
            )
    return rows


def _print_case_result(baseline: dict, policy: dict) -> None:
    print(f"[RESULT] {baseline['case']}: E zero residual vs F PPO")
    for metric in DISPLAY_METRICS:
        baseline_value = float(baseline[metric])
        policy_value = float(policy[metric])
        if abs(baseline_value) > 1.0e-12:
            improvement = 100.0 * (baseline_value - policy_value) / abs(baseline_value)
            suffix = f"{improvement:+.2f}%"
        else:
            suffix = "n/a"
        print(
            f"  {metric}: E={baseline_value:.6g}, F={policy_value:.6g}, "
            f"improvement={suffix}"
        )
    print(
        "  safety: "
        f"E early={baseline['early_termination']}, F early={policy['early_termination']}, "
        f"E/F support mean={float(baseline['mean_support_count_proxy']):.3f}/"
        f"{float(policy['mean_support_count_proxy']):.3f}, "
        f"F residual at limit={100.0 * float(policy['residual_at_limit_fraction']):.3f}%, "
        f"F residual saturation={100.0 * float(policy['residual_saturation_fraction']):.3f}%"
    )


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    experiment_cfg: dict,
) -> None:
    if args_cli.duration <= 0.0:
        raise ValueError("--duration must be positive")

    checkpoint = _resolve_checkpoint()
    output_dir = Path(args_cli.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.residual_domain_randomization = True
    env_cfg.enable_visual_terrain_mesh = False
    env_cfg.episode_length_s = max(float(args_cli.duration) + 2.0, 16.0)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = gym_env.unwrapped
    wrapped_env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")

    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped_env, experiment_cfg)
    runner.agent.load(str(checkpoint))
    runner.agent.set_running_mode("eval")
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(
        f"[INFO] Held-out phase=({args_cli.phase_x:.3f}, {args_cli.phase_y:.3f}), "
        f"seed={args_cli.seed}, duration={args_cli.duration:.3f} s"
    )

    trajectory_rows: list[dict] = []
    summaries: list[dict] = []
    try:
        for case in _selected_cases():
            case_summaries = []
            initial_signatures = []
            for controller in ("E_zero", "F_ppo"):
                rows, summary, initial_signature = _run_controller(
                    gym_env, wrapped_env, raw_env, runner.agent, case, controller
                )
                trajectory_rows.extend(rows)
                summaries.append(summary)
                case_summaries.append(summary)
                initial_signatures.append(initial_signature)
            initial_difference = torch.max(
                torch.abs(initial_signatures[0] - initial_signatures[1])
            )
            if float(initial_difference) > 1.0e-6:
                raise RuntimeError(
                    f"{case.name} E/F initial states differ: "
                    f"max abs difference={float(initial_difference):.9g}"
                )
            for summary in case_summaries:
                summary["ef_initial_state_max_abs_difference"] = float(
                    initial_difference
                )
                summary["checkpoint"] = str(checkpoint)
            print(
                f"[CHECK] {case.name}: E/F initial-state max difference="
                f"{float(initial_difference):.3e}"
            )
            _print_case_result(case_summaries[0], case_summaries[1])
    finally:
        wrapped_env.close()

    comparison_rows = _comparisons(summaries)
    trajectory_path = output_dir / "residual_checkpoint_trajectories.csv"
    summary_path = output_dir / "residual_checkpoint_summary.csv"
    comparison_path = output_dir / "residual_checkpoint_improvements.csv"
    _write_csv(trajectory_path, trajectory_rows)
    _write_csv(summary_path, summaries)
    _write_csv(comparison_path, comparison_rows)
    print(f"[RESULT] trajectory CSV: {trajectory_path}")
    print(f"[RESULT] summary CSV: {summary_path}")
    print(f"[RESULT] improvement CSV: {comparison_path}")
    print("[NOTE] Support/contact metrics are analytical proxies, not contact sensors.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
