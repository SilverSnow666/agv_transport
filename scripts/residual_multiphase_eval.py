"""Run and analyze deterministic multi-phase E versus E1 evaluation.

The default stress suite uses four phase/seed pairs and the rough/combined
conditions. Each child process performs a strictly matched E-zero/F-PPO pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
E1_TASK = "Template-Agv-Level-Residual-Smooth-Direct-v0"
E1_CHECKPOINT = (
    ROOT
    / "logs/skrl/agv_level_residual_smooth_direct"
    / "2026-09-14_15-10-40_ppo_torch_v7_6_e1_residual_ppo_smooth_reward"
    / "checkpoints/best_agent.pt"
)
PHASES = (
    {"label": "p0", "seed": 137, "phase_x": 1.17, "phase_y": -2.03},
    {"label": "p1", "seed": 211, "phase_x": -1.31, "phase_y": 0.77},
    {"label": "p2", "seed": 353, "phase_x": 2.43, "phase_y": 1.88},
    {"label": "p3", "seed": 509, "phase_x": -2.68, "phase_y": -0.49},
)
ALL_CASES = ("nominal", "fast", "rough", "heavy", "offset", "combined")
METRICS = {
    "roll": "board_roll_rms_deg",
    "pitch": "board_pitch_rms_deg",
    "board_rp_speed": "board_rp_angular_velocity_rms_rad_s",
    "lift_speed": "lift_velocity_rms_m_s",
    "cargo_slip": "cargo_cumulative_slip_mm",
    "cargo_post_warmup_slip": "cargo_post_warmup_cumulative_slip_mm",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_training_seed(path: Path) -> int:
    agent_cfg = path.parent.parent / "params/agent.yaml"
    if not agent_cfg.is_file():
        raise FileNotFoundError(f"Checkpoint training config is missing: {agent_cfg}")
    config = yaml.safe_load(agent_cfg.read_text(encoding="utf-8"))
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Invalid training seed in {agent_cfg}: {seed!r}")
    return seed


def _validate_numeric_rows(rows: list[dict[str, str]], excluded: set[str]) -> None:
    for row in rows:
        for key, value in row.items():
            if key not in excluded and not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite {key}={value}")


def _post_warmup_cumulative_slip(
    trajectories: list[dict[str, str]], controller: str, warmup_s: float
) -> float:
    rows = [row for row in trajectories if row["controller"] == controller]
    post = [row for row in rows if float(row["time_s"]) >= warmup_s]
    if not rows or not post:
        raise RuntimeError(f"Missing {controller} samples after {warmup_s:.6g} s")
    return 1000.0 * (
        float(rows[-1]["cargo_cumulative_slip_m"])
        - float(post[0]["cargo_cumulative_slip_m"])
    )


def _analyze(output: Path, manifest: dict) -> None:
    runs: list[dict] = []
    for phase in manifest["phases"]:
        for case in manifest["cases"]:
            directory = output / phase["label"] / case
            summaries = _read(directory / "residual_checkpoint_summary.csv")
            trajectories = _read(directory / "residual_checkpoint_trajectories.csv")
            expected_samples = sum(int(float(row["samples"])) for row in summaries)
            if len(summaries) != 2 or len(trajectories) != expected_samples:
                raise RuntimeError(
                    f"Incomplete {phase['label']}/{case}: "
                    f"{len(summaries)} summaries, {len(trajectories)} samples"
                )
            _validate_numeric_rows(trajectories, {"case", "controller"})
            baseline = next(row for row in summaries if row["controller"] == "E_zero")
            policy = next(row for row in summaries if row["controller"] == "F_ppo")
            if float(policy["ef_initial_state_max_abs_difference"]) > 1.0e-6:
                raise RuntimeError(f"Initial-state mismatch: {phase['label']}/{case}")
            result: dict[str, str | float | int] = {
                "phase": phase["label"],
                "seed": phase["seed"],
                "phase_x_rad": phase["phase_x"],
                "phase_y_rad": phase["phase_y"],
                "case": case,
            }
            for label, metric in METRICS.items():
                if metric in baseline and metric in policy:
                    e_value = float(baseline[metric])
                    f_value = float(policy[metric])
                elif metric == "cargo_post_warmup_cumulative_slip_mm":
                    warmup = float(manifest.get("slip_warmup_s", 0.5))
                    e_value = _post_warmup_cumulative_slip(
                        trajectories, "E_zero", warmup
                    )
                    f_value = _post_warmup_cumulative_slip(
                        trajectories, "F_ppo", warmup
                    )
                else:
                    raise KeyError(metric)
                result[f"{label}_e"] = e_value
                result[f"{label}_e1"] = f_value
                result[f"{label}_improvement_pct"] = (
                    100.0 * (e_value - f_value) / abs(e_value)
                )
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
                result[metric] = float(policy[metric])
            runs.append(result)
            print(
                f"[CHECK] {phase['label']}/{case}: matched, finite, "
                f"roll={result['roll_improvement_pct']:+.2f}%, "
                f"pitch={result['pitch_improvement_pct']:+.2f}%",
                flush=True,
            )

    aggregates: list[dict] = []
    scopes = (*manifest["cases"], "all")
    for scope in scopes:
        selected = runs if scope == "all" else [row for row in runs if row["case"] == scope]
        for label in METRICS:
            values = np.array(
                [float(row[f"{label}_improvement_pct"]) for row in selected]
            )
            aggregates.append(
                {
                    "scope": scope,
                    "metric": label,
                    "samples": len(values),
                    "mean_improvement_pct": float(values.mean()),
                    "sample_std_pct": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min_improvement_pct": float(values.min()),
                    "max_improvement_pct": float(values.max()),
                    "positive_fraction": float(np.mean(values > 0.0)),
                }
            )

    safety_keys = (
        "early_termination", "board_dropped", "board_tipped",
        "cargo_dropped", "cargo_tipped",
    )
    if any(
        any(float(row[key]) for key in safety_keys)
        or float(row["support_loss_fraction_proxy"])
        or float(row["residual_saturation_fraction"])
        for row in runs
    ):
        raise RuntimeError("Safety, support-proxy or target-saturation event detected")

    _write(output / "multiphase_runs.csv", runs)
    _write(output / "multiphase_aggregate.csv", aggregates)
    _plot(output / "multiphase_stress_comparison.png", runs, manifest)
    print(f"[RESULT] Multi-phase outputs: {output}", flush=True)


def _plot(path: Path, runs: list[dict], manifest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("roll_improvement_pct", "Roll RMS improvement (%)"),
        ("pitch_improvement_pct", "Pitch RMS improvement (%)"),
        ("board_rp_speed_improvement_pct", "Board RP speed improvement (%)"),
        ("lift_speed_improvement_pct", "Lift speed improvement (%)"),
        ("cargo_slip_improvement_pct", "Cargo slip improvement (%)"),
        ("residual_at_limit_fraction", "Residual at limit (%)"),
    )
    labels = [phase["label"] for phase in manifest["phases"]]
    x = np.arange(len(labels))
    colors = ("#2474a6", "#d15a29", "#17834b", "#6b4fa1", "#8c6d31", "#777777")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for axis, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        for color, case in zip(colors, manifest["cases"], strict=False):
            case_rows = [row for row in runs if row["case"] == case]
            values = [float(row[metric]) for row in case_rows]
            if metric == "residual_at_limit_fraction":
                values = [100.0 * value for value in values]
            axis.plot(x, values, marker="o", color=color, label=case)
        if metric != "residual_at_limit_fraction":
            axis.axhline(0.0, color="black", lw=0.8)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_xticks(x, labels)
    axes[0, 0].legend()
    for axis in axes[-1]:
        axis.set_xlabel("Held-out phase/seed pair")
    fig.suptitle("V7.6-F E1 best: matched multi-phase stress evaluation")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("stress", "all"), default="stress")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--slip_warmup", type=float, default=0.5)
    parser.add_argument("--task", type=str, default=E1_TASK)
    parser.add_argument("--checkpoint", type=Path, default=E1_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/v7_6_f/multiphase_stress")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.analyze_only:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        _analyze(output, manifest)
        return
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be positive and finite")
    if not math.isfinite(args.slip_warmup) or not 0.0 <= args.slip_warmup < args.duration:
        parser.error("--slip_warmup must be finite and in [0, duration)")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty; use --analyze_only to reanalyze")
    cases = ALL_CASES if args.suite == "all" else ("rough", "combined")
    manifest = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_seed": _checkpoint_training_seed(checkpoint),
        "duration_s": args.duration,
        "slip_warmup_s": args.slip_warmup,
        "suite": args.suite,
        "cases": cases,
        "phases": PHASES,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "note": (
            "Phase pairs are deterministic, spread across quadrants and fixed before "
            "execution. Seeds exercise reset reproducibility; with pinned case ranges "
            "they are not independent training seeds. Support/contact is analytical."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for phase in PHASES:
        for case in cases:
            directory = output / phase["label"] / case
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/residual_checkpoint_eval.py"),
                "--task", args.task,
                "--checkpoint", str(checkpoint),
                "--case", case,
                "--duration", str(args.duration),
                "--slip_warmup", str(args.slip_warmup),
                "--seed", str(phase["seed"]),
                "--phase_x", str(phase["phase_x"]),
                "--phase_y", str(phase["phase_y"]),
                "--log_dir", str(directory),
            ]
            if args.headless:
                command.append("--headless")
            print(
                f"[RUN] {phase['label']}/{case}: seed={phase['seed']}, "
                f"phase=({phase['phase_x']},{phase['phase_y']})",
                flush=True,
            )
            subprocess.run(command, cwd=ROOT, check=True)
    if _sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    _analyze(output, manifest)


if __name__ == "__main__":
    main()
