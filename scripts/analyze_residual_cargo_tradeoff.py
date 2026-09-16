"""Audit V7.6-H Cargo reward scale and E1 stress-case trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "logs/v7_6_f/multiphase_stress",
    ROOT / "logs/v7_6_g/seed43_multiphase_stress",
    ROOT / "logs/v7_6_g/seed44_multiphase_stress",
)
REWARD_KEYS = (
    "board_angle",
    "board_angular_velocity",
    "board_vertical_velocity",
    "cargo_slip",
    "cargo_velocity",
    "cargo_tilt",
    "cargo_angular_velocity",
    "action",
    "action_rate",
    "lift_velocity",
)
OBSERVATION_LAYOUT = (
    "board_roll",
    "board_pitch",
    "board_roll_rate",
    "board_pitch_rate",
    "board_vertical_velocity",
    "lift1_height_error",
    "lift2_height_error",
    "lift3_height_error",
    "lift1_velocity",
    "lift2_velocity",
    "lift3_velocity",
    "base_target1_error",
    "base_target2_error",
    "base_target3_error",
    "feedback1_height",
    "feedback2_height",
    "feedback3_height",
    "support1_relative_height",
    "support2_relative_height",
    "support3_relative_height",
    "agv1_roll",
    "agv2_roll",
    "agv3_roll",
    "agv1_pitch",
    "agv2_pitch",
    "agv3_pitch",
    "cargo_relative_x",
    "cargo_relative_y",
    "cargo_relative_vx",
    "cargo_relative_vy",
    "cargo_relative_roll",
    "cargo_relative_pitch",
    "cargo_relative_angular_speed",
)


class _TupleSafeLoader(yaml.SafeLoader):
    """Load Isaac Lab's saved tuple values without enabling arbitrary YAML objects."""


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def _load_yaml(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_TupleSafeLoader)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_training_metadata(directory: Path) -> tuple[dict, Path, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if _sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    run_dir = checkpoint.parent.parent
    agent_cfg = _load_yaml(run_dir / "params/agent.yaml")
    env_cfg = _load_yaml(run_dir / "params/env.yaml")
    seed = agent_cfg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Invalid training seed for {checkpoint}: {seed!r}")
    manifest_seed = int(manifest.get("training_seed", seed))
    if manifest_seed != seed:
        raise RuntimeError(f"Training-seed mismatch in {directory}")
    return manifest, checkpoint, env_cfg


def _scale(env_cfg: dict, name: str) -> float:
    return float(env_cfg[f"residual_{name}_penalty_scale"])


def _reference(env_cfg: dict, name: str) -> float:
    return float(env_cfg[f"residual_{name}_reference"])


def _reward_terms(
    row: dict[str, str],
    previous_action: np.ndarray,
    env_cfg: dict,
) -> tuple[dict[str, float], np.ndarray]:
    roll = math.radians(float(row["board_roll_deg"]))
    pitch = math.radians(float(row["board_pitch_deg"]))
    board_rates = np.array(
        [float(row["board_roll_rate_rad_s"]), float(row["board_pitch_rate_rad_s"])]
    )
    cargo_tilt = np.radians(
        [float(row["cargo_relative_roll_deg"]), float(row["cargo_relative_pitch_deg"])]
    )
    height_limit = float(env_cfg["residual_height_limit"])
    action = np.array([float(row[f"residual{i}_m"]) for i in range(1, 4)]) / height_limit
    lift_velocity = np.array(
        [float(row[f"lift{i}_velocity_m_s"]) for i in range(1, 4)]
    )
    costs = {
        "board_angle": (roll**2 + pitch**2) / _reference(env_cfg, "board_angle") ** 2,
        "board_angular_velocity": float(
            np.sum(np.square(board_rates / _reference(env_cfg, "board_angular_velocity")))
        ),
        "board_vertical_velocity": (
            float(row["board_vz_m_s"]) / _reference(env_cfg, "board_vertical_velocity")
        ) ** 2,
        "cargo_slip": (
            float(row["cargo_slip_m"]) / _reference(env_cfg, "cargo_slip")
        ) ** 2,
        "cargo_velocity": (
            float(row["cargo_relative_speed_m_s"])
            / _reference(env_cfg, "cargo_velocity")
        ) ** 2,
        "cargo_tilt": float(np.sum(np.square(cargo_tilt / _reference(env_cfg, "cargo_tilt")))),
        "cargo_angular_velocity": (
            float(row["cargo_relative_angular_speed_rad_s"])
            / _reference(env_cfg, "cargo_angular_velocity")
        ) ** 2,
        "action": float(np.mean(np.square(action))),
        "action_rate": float(np.mean(np.square(action - previous_action))),
        "lift_velocity": float(
            np.mean(np.square(lift_velocity / _reference(env_cfg, "lift_velocity")))
        ),
    }
    terms = {key: -_scale(env_cfg, key) * costs[key] for key in REWARD_KEYS}
    return terms, action


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _rms(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


def _pearson(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) < 2 or float(a.std()) <= 1.0e-12 or float(b.std()) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _trajectory_summary(
    rows: list[dict[str, str]],
    *,
    training_seed: int,
    controller: str,
    env_cfg: dict,
    warmup_s: float,
) -> tuple[dict, list[dict], dict]:
    selected = [row for row in rows if row["controller"] == controller]
    if not selected:
        raise RuntimeError(f"Missing {controller} trajectory for seed {training_seed}")
    previous_action = np.zeros(3)
    reward_rows: list[dict] = []
    for row in selected:
        terms, previous_action = _reward_terms(row, previous_action, env_cfg)
        reconstructed = float(env_cfg["residual_alive_reward"]) + sum(terms.values())
        reward_rows.append({
            "training_seed": training_seed,
            "controller": controller,
            "time_s": float(row["time_s"]),
            **terms,
            "reward_reconstructed": reconstructed,
            "reward_logged": float(row["reward"]),
            "reward_error": reconstructed - float(row["reward"]),
        })
    post = [index for index, row in enumerate(selected) if float(row["time_s"]) >= warmup_s]
    first_cumulative = float(selected[0]["cargo_cumulative_slip_m"])
    warmup_cumulative = float(selected[post[0]]["cargo_cumulative_slip_m"])
    summary = {
        "training_seed": training_seed,
        "controller": controller,
        "samples": len(selected),
        "warmup_s": warmup_s,
        "initial_step_cumulative_slip_mm": 1000.0 * first_cumulative,
        "final_cumulative_slip_mm": 1000.0 * float(selected[-1]["cargo_cumulative_slip_m"]),
        "post_initial_cumulative_slip_mm": 1000.0
        * (float(selected[-1]["cargo_cumulative_slip_m"]) - first_cumulative),
        "cumulative_slip_at_warmup_mm": 1000.0 * warmup_cumulative,
        "post_warmup_cumulative_slip_mm": 1000.0
        * (float(selected[-1]["cargo_cumulative_slip_m"]) - warmup_cumulative),
        "cargo_displacement_rms_mm": 1000.0
        * _rms([float(selected[index]["cargo_slip_m"]) for index in post]),
        "cargo_relative_speed_rms_mm_s": 1000.0
        * _rms([float(selected[index]["cargo_relative_speed_m_s"]) for index in post]),
        "board_rp_speed_rms_rad_s": _rms([
            math.hypot(
                float(selected[index]["board_roll_rate_rad_s"]),
                float(selected[index]["board_pitch_rate_rad_s"]),
            )
            for index in post
        ]),
        "lift_speed_rms_mm_s": 1000.0
        * _rms([
            math.sqrt(
                sum(float(selected[index][f"lift{i}_velocity_m_s"]) ** 2 for i in range(1, 4))
                / 3.0
            )
            for index in post
        ]),
        "reward_reconstruction_max_abs_error": max(
            abs(float(row["reward_error"])) for row in reward_rows
        ),
    }
    for key in REWARD_KEYS:
        summary[f"reward_{key}_mean"] = _mean(
            [float(reward_rows[index][key]) for index in post]
        )

    increments = [
        1000.0
        * max(
            0.0,
            float(selected[index]["cargo_cumulative_slip_m"])
            - float(selected[index - 1]["cargo_cumulative_slip_m"]),
        )
        for index in post
        if index > 0
    ]
    driver_indices = [index for index in post if index > 0]
    drivers = {
        "board_rp_speed": [
            math.hypot(
                float(selected[index]["board_roll_rate_rad_s"]),
                float(selected[index]["board_pitch_rate_rad_s"]),
            )
            for index in driver_indices
        ],
        "board_vertical_acceleration": [
            abs(float(selected[index]["board_vertical_acceleration_m_s2"]))
            for index in driver_indices
        ],
        "lift_speed": [
            math.sqrt(
                sum(float(selected[index][f"lift{i}_velocity_m_s"]) ** 2 for i in range(1, 4))
                / 3.0
            )
            for index in driver_indices
        ],
        "residual_rate": [
            math.sqrt(
                sum(float(selected[index][f"residual_rate{i}_m_s"]) ** 2 for i in range(1, 4))
                / 3.0
            )
            for index in driver_indices
        ],
    }
    correlations = {
        key: _pearson(increments, values) for key, values in drivers.items()
    }
    return summary, reward_rows, correlations


def _observation_scalers(checkpoint: Path, training_seed: int) -> list[dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scaler = state["state_preprocessor"]
    mean = scaler["running_mean"].detach().cpu().numpy()
    variance = scaler["running_variance"].detach().cpu().numpy()
    if len(mean) != len(OBSERVATION_LAYOUT):
        raise RuntimeError(f"Unexpected observation size in {checkpoint}: {len(mean)}")
    return [
        {
            "training_seed": training_seed,
            "index": index,
            "observation": label,
            "running_mean": float(mean[index]),
            "running_std": float(math.sqrt(max(float(variance[index]), 0.0))),
        }
        for index, label in enumerate(OBSERVATION_LAYOUT)
    ]


def _plot(
    path: Path,
    trajectories: dict[int, list[dict[str, str]]],
    summaries: list[dict],
    correlations: list[dict],
    warmup_s: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted(trajectories)
    colors = dict(zip(seeds, ("#2474a6", "#d15a29", "#3b8f55"), strict=True))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    e_rows = [row for row in trajectories[seeds[0]] if row["controller"] == "E_zero"]
    axes[0, 0].plot(
        [float(row["time_s"]) for row in e_rows],
        [1000.0 * float(row["cargo_cumulative_slip_m"]) for row in e_rows],
        color="black",
        linestyle="--",
        label="E zero residual",
    )
    for seed in seeds:
        rows = [row for row in trajectories[seed] if row["controller"] == "F_ppo"]
        axes[0, 0].plot(
            [float(row["time_s"]) for row in rows],
            [1000.0 * float(row["cargo_cumulative_slip_m"]) for row in rows],
            color=colors[seed],
            label=f"E1 seed {seed}",
        )
    axes[0, 0].set_title("p3/combined Cargo cumulative slip")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Cumulative slip (mm)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    for seed in seeds:
        rows = [row for row in trajectories[seed] if row["controller"] == "F_ppo"]
        post = [row for row in rows if float(row["time_s"]) >= warmup_s]
        initial = float(post[0]["cargo_cumulative_slip_m"])
        axes[0, 1].plot(
            [float(row["time_s"]) for row in post],
            [1000.0 * (float(row["cargo_cumulative_slip_m"]) - initial) for row in post],
            color=colors[seed],
            label=f"seed {seed}",
        )
    post_e = [row for row in e_rows if float(row["time_s"]) >= warmup_s]
    initial_e = float(post_e[0]["cargo_cumulative_slip_m"])
    axes[0, 1].plot(
        [float(row["time_s"]) for row in post_e],
        [1000.0 * (float(row["cargo_cumulative_slip_m"]) - initial_e) for row in post_e],
        color="black",
        linestyle="--",
        label="E zero residual",
    )
    axes[0, 1].axvline(warmup_s, color="grey", lw=0.8, linestyle=":")
    axes[0, 1].set_title(f"Slip accumulated after {warmup_s:.1f} s settling window")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Post-initial cumulative slip (mm)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    penalty_names = ("board_angle", "cargo_slip", "cargo_velocity", "lift_velocity", "action")
    x = np.arange(len(penalty_names))
    width = 0.22
    for offset, seed in enumerate(seeds):
        summary = next(
            row for row in summaries
            if int(row["training_seed"]) == seed and row["controller"] == "F_ppo"
        )
        axes[1, 0].bar(
            x + (offset - 1) * width,
            [-float(summary[f"reward_{key}_mean"]) for key in penalty_names],
            width,
            color=colors[seed],
            label=f"seed {seed}",
        )
    axes[1, 0].set_xticks(x, [name.replace("_", "\n") for name in penalty_names])
    axes[1, 0].set_title(f"Mean E1 penalty magnitude after {warmup_s:.1f} s")
    axes[1, 0].set_ylabel("Penalty per control step")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    driver_names = (
        "board_rp_speed",
        "board_vertical_acceleration",
        "lift_speed",
        "residual_rate",
    )
    for offset, seed in enumerate(seeds):
        row = next(item for item in correlations if int(item["training_seed"]) == seed)
        axes[1, 1].bar(
            x[: len(driver_names)] + (offset - 1) * width,
            [float(row[name]) for name in driver_names],
            width,
            color=colors[seed],
            label=f"seed {seed}",
        )
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].set_xticks(
        x[: len(driver_names)], [name.replace("_", "\n") for name in driver_names]
    )
    axes[1, 1].set_title("Correlation with per-step Cargo path increment")
    axes[1, 1].set_ylabel("Pearson correlation")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("V7.6-H.1 Cargo objective and trajectory audit")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def analyze(inputs: list[Path], output: Path, warmup_s: float) -> None:
    summaries: list[dict] = []
    reward_rows: list[dict] = []
    correlation_rows: list[dict] = []
    scaler_rows: list[dict] = []
    trajectories: dict[int, list[dict[str, str]]] = {}
    sources: list[dict] = []
    for directory in inputs:
        manifest, checkpoint, env_cfg = _load_training_metadata(directory.resolve())
        seed = int(_load_yaml(checkpoint.parent.parent / "params/agent.yaml")["seed"])
        trajectory_path = directory / "p3/combined/residual_checkpoint_trajectories.csv"
        rows = _read_csv(trajectory_path)
        trajectories[seed] = rows
        for controller in ("E_zero", "F_ppo"):
            summary, terms, correlations = _trajectory_summary(
                rows,
                training_seed=seed,
                controller=controller,
                env_cfg=env_cfg,
                warmup_s=warmup_s,
            )
            summaries.append(summary)
            reward_rows.extend(terms)
            if controller == "F_ppo":
                correlation_rows.append({"training_seed": seed, **correlations})
        scaler_rows.extend(_observation_scalers(checkpoint, seed))
        sources.append({
            "training_seed": seed,
            "input_directory": str(directory.resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        })
    if len(trajectories) != len(inputs):
        raise RuntimeError("Training seeds are not unique")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "cargo_tradeoff_summary.csv", summaries)
    _write_csv(output / "reward_term_trajectories.csv", reward_rows)
    _write_csv(output / "cargo_slip_correlations.csv", correlation_rows)
    _write_csv(output / "observation_scalers.csv", scaler_rows)
    (output / "manifest.json").write_text(json.dumps({
        "sources": sorted(sources, key=lambda row: row["training_seed"]),
        "case": "p3/combined",
        "warmup_s": warmup_s,
        "observation_finding": (
            "The reward uses cargo_relative_xy - cargo_initial_relative_xy, but the policy "
            "observes only cargo_relative_xy and is not given cargo_initial_relative_xy or "
            "the slip-from-reset vector."
        ),
        "support_contact_note": "Support/contact fields are analytical proxies.",
    }, indent=2), encoding="utf-8")
    _plot(output / "cargo_tradeoff_audit.png", trajectories, summaries, correlation_rows, warmup_s)
    print(f"[RESULT] Training seeds: {sorted(trajectories)}", flush=True)
    print(f"[RESULT] Cargo audit outputs: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/v7_6_h/cargo_objective_audit"
    )
    parser.add_argument("--warmup", type=float, default=0.5)
    args = parser.parse_args()
    inputs = args.input or list(DEFAULT_INPUTS)
    if args.warmup < 0.0:
        parser.error("--warmup must be non-negative")
    analyze(inputs, args.output.resolve(), args.warmup)


if __name__ == "__main__":
    main()
