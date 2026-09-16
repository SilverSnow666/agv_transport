"""Compare the frozen E1 and H2 Cargo-observable residual policies.

Both inputs must be complete V7.6 multiphase stress evaluations using the
same phase/case matrix.  The script verifies the paired E baselines before
reporting each policy relative to E and H2 directly relative to E1.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KEYS = ("phase", "case")
METRICS = {
    "roll": ("roll_e", "roll_e1"),
    "pitch": ("pitch_e", "pitch_e1"),
    "board_rp_speed": ("board_rp_speed_e", "board_rp_speed_e1"),
    "lift_speed": ("lift_speed_e", "lift_speed_e1"),
    "cargo_slip": ("cargo_slip_e", "cargo_slip_e1"),
    "cargo_post_warmup_slip": (
        "cargo_post_warmup_slip_e",
        "cargo_post_warmup_slip_e1",
    ),
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


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed = {(row["phase"], row["case"]): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError("Duplicate phase/case row")
    return indexed


def _improvement(reference: float, candidate: float) -> float:
    if reference == 0.0:
        raise ZeroDivisionError("Improvement reference is zero")
    return 100.0 * (reference - candidate) / abs(reference)


def compare(e1_rows: list[dict[str, str]], h2_rows: list[dict[str, str]]) -> list[dict]:
    e1_index = _index(e1_rows)
    h2_index = _index(h2_rows)
    if set(e1_index) != set(h2_index):
        raise RuntimeError("E1 and H2 phase/case matrices differ")
    output: list[dict] = []
    for key in sorted(e1_index):
        e1 = e1_index[key]
        h2 = h2_index[key]
        row: dict[str, str | float] = {"phase": key[0], "case": key[1]}
        for label, (baseline_key, policy_key) in METRICS.items():
            e1_baseline = float(e1[baseline_key])
            h2_baseline = float(h2[baseline_key])
            if not math.isclose(e1_baseline, h2_baseline, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(
                    f"Paired E baseline mismatch for {key}/{label}: "
                    f"{e1_baseline} != {h2_baseline}"
                )
            e1_value = float(e1[policy_key])
            h2_value = float(h2[policy_key])
            row[f"{label}_e"] = e1_baseline
            row[f"{label}_e1"] = e1_value
            row[f"{label}_h2"] = h2_value
            row[f"{label}_e1_vs_e_pct"] = _improvement(e1_baseline, e1_value)
            row[f"{label}_h2_vs_e_pct"] = _improvement(e1_baseline, h2_value)
            row[f"{label}_h2_vs_e1_pct"] = _improvement(e1_value, h2_value)
        for metric in (
            "residual_rms_mm",
            "residual_common_mode_rms_mm",
            "residual_differential_rms_mm",
            "residual_at_limit_fraction",
            "residual_saturation_fraction",
            "support_loss_fraction_proxy",
            "early_termination",
            "board_dropped",
            "board_tipped",
            "cargo_dropped",
            "cargo_tipped",
        ):
            row[f"{metric}_e1"] = float(e1[metric])
            row[f"{metric}_h2"] = float(h2[metric])
        output.append(row)
    return output


def aggregate(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    scopes = sorted({str(row["case"]) for row in rows}) + ["all"]
    for scope in scopes:
        selected = rows if scope == "all" else [row for row in rows if row["case"] == scope]
        result: dict[str, str | float | int] = {"scope": scope, "samples": len(selected)}
        for label in METRICS:
            for comparison in ("e1_vs_e_pct", "h2_vs_e_pct", "h2_vs_e1_pct"):
                values = np.asarray(
                    [float(row[f"{label}_{comparison}"]) for row in selected], dtype=float
                )
                result[f"mean_{label}_{comparison}"] = float(values.mean())
                result[f"std_{label}_{comparison}"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
                result[f"positive_fraction_{label}_{comparison}"] = float(
                    np.mean(values > 0.0)
                )
        for strategy in ("e1", "h2"):
            result[f"mean_residual_at_limit_fraction_{strategy}"] = float(
                np.mean([float(row[f"residual_at_limit_fraction_{strategy}"]) for row in selected])
            )
            result[f"mean_residual_common_mode_rms_mm_{strategy}"] = float(
                np.mean([float(row[f"residual_common_mode_rms_mm_{strategy}"]) for row in selected])
            )
        output.append(result)
    return output


def _validate_safety(rows: list[dict]) -> None:
    event_metrics = (
        "residual_saturation_fraction",
        "support_loss_fraction_proxy",
        "early_termination",
        "board_dropped",
        "board_tipped",
        "cargo_dropped",
        "cargo_tipped",
    )
    if any(
        float(row[f"{metric}_{strategy}"])
        for row in rows
        for metric in event_metrics
        for strategy in ("e1", "h2")
    ):
        raise RuntimeError("Safety, support-proxy or final-target saturation event detected")


def _plot(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: (str(row["phase"]), str(row["case"])))
    labels = [f"{row['phase']}\n{row['case']}" for row in ordered]
    x = np.arange(len(ordered))
    panels = (
        ("roll", "Roll RMS improvement vs E (%)"),
        ("pitch", "Pitch RMS improvement vs E (%)"),
        ("board_rp_speed", "Board RP speed improvement vs E (%)"),
        ("lift_speed", "Lift speed improvement vs E (%)"),
        ("cargo_slip", "Full Cargo slip improvement vs E (%)"),
        ("cargo_post_warmup_slip", "Post-0.5 s Cargo slip improvement vs E (%)"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for axis, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        axis.plot(
            x,
            [float(row[f"{metric}_e1_vs_e_pct"]) for row in ordered],
            marker="o",
            label="E1",
        )
        axis.plot(
            x,
            [float(row[f"{metric}_h2_vs_e_pct"]) for row in ordered],
            marker="s",
            label="H2",
        )
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 0].legend()
    fig.suptitle("V7.6-H2 Cargo-target observation: frozen multi-phase comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e1",
        type=Path,
        default=ROOT / "logs/v7_6_f/multiphase_stress/multiphase_runs.csv",
    )
    parser.add_argument(
        "--h2",
        type=Path,
        default=ROOT / "logs/v7_6_h2/multiphase_stress/multiphase_runs.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/v7_6_h2/comparison"
    )
    args = parser.parse_args()
    rows = compare(_read(args.e1.resolve()), _read(args.h2.resolve()))
    _validate_safety(rows)
    aggregates = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "per_run_comparison.csv", rows)
    _write(args.output / "aggregate_comparison.csv", aggregates)
    _plot(args.output / "cargo_observable_comparison.png", rows)
    all_row = next(row for row in aggregates if row["scope"] == "all")
    print(
        "[RESULT] H2 vs E1 mean direct improvements: "
        f"roll={all_row['mean_roll_h2_vs_e1_pct']:+.2f}%, "
        f"pitch={all_row['mean_pitch_h2_vs_e1_pct']:+.2f}%, "
        f"Board RP speed={all_row['mean_board_rp_speed_h2_vs_e1_pct']:+.2f}%, "
        f"Lift speed={all_row['mean_lift_speed_h2_vs_e1_pct']:+.2f}%, "
        f"full Cargo slip={all_row['mean_cargo_slip_h2_vs_e1_pct']:+.2f}%, "
        "post-warmup Cargo slip="
        f"{all_row['mean_cargo_post_warmup_slip_h2_vs_e1_pct']:+.2f}%"
    )


if __name__ == "__main__":
    main()
