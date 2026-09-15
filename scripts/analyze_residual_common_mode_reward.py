"""Aggregate and plot the V7.6-E3 common-mode reward experiment.

The three inputs are existing paired E/F checkpoint evaluations. Older E1
summaries are enriched from their trajectory CSV so all strategies use the
same definitions for common/differential residual and Lift velocity.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ("nominal", "fast", "rough", "heavy", "offset", "combined")
STRATEGIES = ("E1 best", "E3 best", "E3 final")
IMPROVEMENT_METRICS = {
    "Roll RMS": "board_roll_rms_deg",
    "Pitch RMS": "board_pitch_rms_deg",
    "Board RP speed": "board_rp_angular_velocity_rms_rad_s",
    "Lift speed": "lift_velocity_rms_m_s",
    "Cargo slip": "cargo_cumulative_slip_mm",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rms(values) -> float:
    values = list(values)
    return math.sqrt(sum(value * value for value in values) / len(values))


def _derived(trajectory: list[dict[str, str]]) -> dict[str, float]:
    residual = [
        [float(row[f"residual{index}_m"]) for index in range(1, 4)]
        for row in trajectory
    ]
    velocity = [
        [float(row[f"lift{index}_velocity_m_s"]) for index in range(1, 4)]
        for row in trajectory
    ]
    residual_common = [sum(values) / 3.0 for values in residual]
    velocity_common = [sum(values) / 3.0 for values in velocity]
    residual_differential = [
        value - residual_common[row_index]
        for row_index, values in enumerate(residual)
        for value in values
    ]
    velocity_differential = [
        value - velocity_common[row_index]
        for row_index, values in enumerate(velocity)
        for value in values
    ]
    return {
        "residual_common_mode_rms_mm": 1000.0 * _rms(residual_common),
        "residual_differential_rms_mm": 1000.0 * _rms(residual_differential),
        "lift_common_mode_velocity_rms_m_s": _rms(velocity_common),
        "lift_differential_velocity_rms_m_s": _rms(velocity_differential),
    }


def _load_strategy(label: str, directory: Path) -> list[dict]:
    summaries = _read(directory / "residual_checkpoint_summary.csv")
    trajectories = _read(directory / "residual_checkpoint_trajectories.csv")
    if len(summaries) != 12 or len(trajectories) != 8640:
        raise RuntimeError(
            f"Incomplete {label}: {len(summaries)} summaries, "
            f"{len(trajectories)} trajectories"
        )
    for row in trajectories:
        for key, value in row.items():
            if key not in ("case", "controller") and not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite trajectory value in {label}: {key}={value}")
    if any(
        float(row["ef_initial_state_max_abs_difference"]) > 1.0e-6
        for row in summaries
    ):
        raise RuntimeError(f"Paired initial states differ in {label}")
    output = []
    for case in CASES:
        baseline = next(
            row for row in summaries
            if row["case"] == case and row["controller"] == "E_zero"
        )
        policy = dict(next(
            row for row in summaries
            if row["case"] == case and row["controller"] == "F_ppo"
        ))
        policy_rows = [
            row for row in trajectories
            if row["case"] == case and row["controller"] == "F_ppo"
        ]
        if len(policy_rows) != 720:
            raise RuntimeError(f"Expected 720 samples for {label}/{case}")
        policy.update(_derived(policy_rows))
        row: dict[str, str | float | int] = {
            "strategy": label,
            "case": case,
        }
        for metric in IMPROVEMENT_METRICS.values():
            baseline_value = float(baseline[metric])
            policy_value = float(policy[metric])
            row[f"{metric}_e"] = baseline_value
            row[f"{metric}_policy"] = policy_value
            row[f"{metric}_improvement_pct"] = (
                100.0 * (baseline_value - policy_value) / abs(baseline_value)
            )
        for metric in (
            "residual_rms_mm",
            "residual_common_mode_rms_mm",
            "residual_differential_rms_mm",
            "lift_common_mode_velocity_rms_m_s",
            "lift_differential_velocity_rms_m_s",
            "residual_at_limit_fraction",
            "residual_saturation_fraction",
            "mean_support_count_proxy",
            "support_loss_fraction_proxy",
            "early_termination",
            "board_dropped",
            "board_tipped",
            "cargo_dropped",
            "cargo_tipped",
        ):
            row[metric] = float(policy[metric])
        output.append(row)
    return output


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for strategy in STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy]
        result: dict[str, str | float | int] = {"strategy": strategy}
        for metric in IMPROVEMENT_METRICS.values():
            result[f"mean_{metric}_e"] = sum(
                float(row[f"{metric}_e"]) for row in selected
            ) / len(selected)
            result[f"mean_{metric}_policy"] = sum(
                float(row[f"{metric}_policy"]) for row in selected
            ) / len(selected)
            result[f"{metric}_improvement_pct"] = sum(
                float(row[f"{metric}_improvement_pct"]) for row in selected
            ) / len(selected)
        for metric in (
            "residual_rms_mm",
            "residual_common_mode_rms_mm",
            "residual_differential_rms_mm",
            "lift_common_mode_velocity_rms_m_s",
            "lift_differential_velocity_rms_m_s",
            "residual_at_limit_fraction",
        ):
            result[f"mean_{metric}"] = sum(float(row[metric]) for row in selected) / len(selected)
        result["worst_residual_at_limit_fraction"] = max(
            float(row["residual_at_limit_fraction"]) for row in selected
        )
        result["safety_event_count"] = int(sum(
            int(float(row[key]))
            for row in selected
            for key in (
                "early_termination", "board_dropped", "board_tipped",
                "cargo_dropped", "cargo_tipped",
            )
        ))
        result["support_loss_fraction_proxy"] = max(
            float(row["support_loss_fraction_proxy"]) for row in selected
        )
        result["residual_saturation_fraction"] = max(
            float(row["residual_saturation_fraction"]) for row in selected
        )
        output.append(result)
    return output


def _plot(path: Path, aggregates: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(STRATEGIES))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    width = 0.15
    for index, (display, metric) in enumerate(IMPROVEMENT_METRICS.items()):
        values = [row[f"{metric}_improvement_pct"] for row in aggregates]
        axes[0, 0].bar(x + (index - 2) * width, values, width, label=display)
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_ylabel("Mean improvement vs paired E (%)")
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].bar(x - 0.18, [row["mean_residual_common_mode_rms_mm"] for row in aggregates], 0.36, label="Common")
    axes[0, 1].bar(x + 0.18, [row["mean_residual_differential_rms_mm"] for row in aggregates], 0.36, label="Differential")
    axes[0, 1].set_ylabel("Residual RMS (mm)")
    axes[0, 1].legend()

    axes[1, 0].bar(x - 0.18, [1000 * row["mean_lift_common_mode_velocity_rms_m_s"] for row in aggregates], 0.36, label="Common")
    axes[1, 0].bar(x + 0.18, [1000 * row["mean_lift_differential_velocity_rms_m_s"] for row in aggregates], 0.36, label="Differential")
    axes[1, 0].set_ylabel("Lift velocity RMS (mm/s)")
    axes[1, 0].legend()

    axes[1, 1].bar(x - 0.18, [100 * row["mean_residual_at_limit_fraction"] for row in aggregates], 0.36, label="Six-case mean")
    axes[1, 1].bar(x + 0.18, [100 * row["worst_residual_at_limit_fraction"] for row in aggregates], 0.36, label="Worst case")
    axes[1, 1].set_ylabel("Residual at limit (%)")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(x, STRATEGIES)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("V7.6-E3 targeted common-mode reward: held-out six-condition comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1", type=Path, default=ROOT / "logs/v7_6_e1/checkpoint_best")
    parser.add_argument("--e3_best", type=Path, default=ROOT / "logs/v7_6_e3/checkpoint_best")
    parser.add_argument("--e3_final", type=Path, default=ROOT / "logs/v7_6_e3/checkpoint_300")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/v7_6_e3/comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, directory in zip(STRATEGIES, (args.e1, args.e3_best, args.e3_final), strict=True):
        rows.extend(_load_strategy(label, directory.resolve()))
    aggregates = _aggregate(rows)
    if any(
        int(row["safety_event_count"])
        or float(row["support_loss_fraction_proxy"])
        or float(row["residual_saturation_fraction"])
        for row in aggregates
    ):
        raise RuntimeError("Safety, support-proxy or final-target saturation event detected")
    _write(args.output / "per_case_comparison.csv", rows)
    _write(args.output / "aggregate_comparison.csv", aggregates)
    _plot(args.output / "common_mode_reward_comparison.png", aggregates)
    for row in aggregates:
        print(
            f"[RESULT] {row['strategy']}: "
            f"roll={row['board_roll_rms_deg_improvement_pct']:+.2f}%, "
            f"pitch={row['board_pitch_rms_deg_improvement_pct']:+.2f}%, "
            f"Board RP speed={row['board_rp_angular_velocity_rms_rad_s_improvement_pct']:+.2f}%, "
            f"Lift speed={row['lift_velocity_rms_m_s_improvement_pct']:+.2f}%, "
            f"Cargo slip={row['cargo_cumulative_slip_mm_improvement_pct']:+.2f}%, "
            f"common/differential residual="
            f"{row['mean_residual_common_mode_rms_mm']:.3f}/"
            f"{row['mean_residual_differential_rms_mm']:.3f} mm, "
            f"limit mean/worst={100 * row['mean_residual_at_limit_fraction']:.2f}%/"
            f"{100 * row['worst_residual_at_limit_fraction']:.2f}%"
        )
    print(f"[RESULT] comparison outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
