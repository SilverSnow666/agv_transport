"""V7.6-E2.2: E, unchanged E1 and inference-only bounded zero-mean E1.

No new task or training configuration. The mapping removes common residual and
uniformly rescales only where centering would exceed the existing +/-3 mm cap.
This is a projected-policy diagnostic, not a retrained-policy performance claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from residual_control_decomposition import E1, ROOT, TASKS, read_csv, write_csv, sha256
from residual_control_metrics import analyze_group, rms


LABELS = {"E_zero": "E", "F_ppo": "E1", "F_ppo_zm": "E1-ZM"}
METRICS = (
    "board_roll_rms_deg", "board_pitch_rms_deg", "board_rp_angular_velocity_rms_rad_s",
    "board_vertical_velocity_rms_m_s", "board_vertical_acceleration_rms_m_s2",
    "cargo_cumulative_slip_mm", "reported_lift_velocity_rms_mm_s",
    "actual_interval_velocity_rms_mm_s", "actual_common_velocity_rms_mm_s",
    "mean_actual_height_range_mm", "residual_common_rms_mm", "residual_differential_rms_mm",
)


def summarize(rows, warmup):
    selected = [r for r in rows if float(r["command_time_s"]) >= warmup - 1e-9]
    start = len(rows) - len(selected)
    def col(key):
        return np.array([float(r[key]) for r in selected])
    residual = np.array([[float(r[f"residual{j}_m"]) for j in (1, 2, 3)] for r in selected])
    policy = np.array([[float(r[f"policy_action{j}_before_projection"]) for j in (1, 2, 3)] for r in selected])
    result = analyze_group(rows, warmup)
    result.update({
        "board_roll_rms_deg": rms(col("board_roll_deg")),
        "board_pitch_rms_deg": rms(col("board_pitch_deg")),
        "board_rp_angular_velocity_rms_rad_s": float(np.sqrt(np.mean(col("board_roll_rate_rad_s")**2 + col("board_pitch_rate_rad_s")**2))),
        "board_vertical_velocity_rms_m_s": rms(col("board_vz_m_s")),
        "board_vertical_acceleration_rms_m_s2": rms(col("board_vertical_acceleration_m_s2")),
        "cargo_cumulative_slip_mm": 1000 * (float(selected[-1]["cargo_cumulative_slip_m"]) - (float(rows[start-1]["cargo_cumulative_slip_m"]) if start else 0)),
        "policy_common_residual_rms_mm": 3 * rms(policy.mean(axis=1)),
        "projection_rescaled_fraction": float(np.mean(col("projection_scale") < 1 - 1e-6)),
        "projection_min_scale": float(col("projection_scale").min()),
        "projection_mean_scale": float(col("projection_scale").mean()),
        "applied_common_max_abs_mm": 1000 * float(np.abs(residual.mean(axis=1)).max()),
        "applied_residual_max_abs_mm": 1000 * float(np.abs(residual).max()),
        "support_loss_fraction_proxy": float(col("support_lost_proxy").mean()),
        "final_target_saturation_fraction": float(col("residual_saturated_count").mean() / 3),
    })
    for key in ("board_dropped", "board_tipped", "cargo_dropped", "cargo_tipped"):
        result[key] = int(col(key).max())
    return result


def plot(output, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"E": "#777777", "E1": "#1976b9", "E1-ZM": "#17834b"}
    fig, axes = plt.subplots(4, len(trajectories), figsize=(6 * len(trajectories), 10), squeeze=False)
    for col, (case, arms) in enumerate(trajectories.items()):
        for name, rows in arms.items():
            t = np.array([float(r["time_s"]) for r in rows])
            series = [np.array([float(r[key]) for r in rows]) for key in ("board_roll_deg", "board_pitch_deg")]
            series += [1000 * np.array([[float(r[f"lift{j}_{suffix}"]) for j in (1, 2, 3)] for r in rows]).mean(axis=1)
                       for suffix in ("height_m", "actual_interval_velocity_m_s")]
            for row, values in enumerate(series):
                axes[row, col].plot(t, values, color=colors[name], lw=1, label=name)
                axes[row, col].grid(alpha=.2)
        axes[0, col].set_title(case)
        axes[0, col].legend(ncol=3)
        axes[-1, col].set_xlabel("Time (s)")
    for row, label in enumerate(("Board roll (deg)", "Board pitch (deg)", "Mean Lift height (mm)", "Mean Lift velocity (mm/s)")):
        axes[row, 0].set_ylabel(label)
    fig.suptitle("E2.2 | E1 checkpoint unchanged | E1-ZM = centered, uniformly bounded residual")
    fig.tight_layout()
    fig.savefig(output / "common_mode_ablation.png", dpi=150)
    fig.savefig(output / "common_mode_ablation.pdf")
    plt.close(fig)


def analyze(output, manifest):
    summaries, comparisons, trajectories = [], [], {}
    for case in manifest["cases"]:
        rows = read_csv(output / case / "residual_checkpoint_trajectories.csv")
        physical = read_csv(output / case / "residual_checkpoint_summary.csv")
        if len(physical) != 3 or any(float(r["ef_initial_state_max_abs_difference"]) > 1e-6 or int(r["early_termination"]) for r in physical):
            raise RuntimeError("Incomplete or unmatched three-arm evaluation")
        trajectories[case] = {}
        for controller, label in LABELS.items():
            arm = [r for r in rows if r["controller"] == controller]
            expected = int(np.ceil(manifest["duration_s"] / float(arm[0]["control_dt_s"]) - 1e-9))
            if len(arm) != expected:
                raise RuntimeError(f"Missing samples: {case}/{controller}")
            for r in arm:
                if any(not np.isfinite(float(value)) for key, value in r.items() if key not in ("case", "controller")):
                    raise RuntimeError("Nonfinite trajectory")
            trajectories[case][label] = arm
            for warmup in sorted({0., manifest["warmup_s"]}):
                if manifest["duration_s"] - warmup < 2 * float(arm[0]["control_dt_s"]):
                    continue
                item = {"case": case, "policy": label, **summarize(arm, warmup)}
                if label == "E1-ZM" and (item["applied_common_max_abs_mm"] > 1e-5 or item["applied_residual_max_abs_mm"] > 3.00001):
                    raise RuntimeError("Applied zero-mean/bound invariant failed")
                summaries.append(item)
        for warmup in sorted({r["warmup_s"] for r in summaries if r["case"] == case}):
            indexed = {r["policy"]: r for r in summaries if r["case"] == case and r["warmup_s"] == warmup}
            for before, after in (("E", "E1"), ("E", "E1-ZM"), ("E1", "E1-ZM")):
                for metric in METRICS:
                    a, b = indexed[before][metric], indexed[after][metric]
                    comparisons.append({"case": case, "warmup_s": warmup, "before": before, "after": after,
                                        "metric": metric, "before_value": a, "after_value": b,
                                        "absolute_improvement": a - b,
                                        "percent_improvement": 100 * (a - b) / abs(a) if abs(a) > 1e-12 else None})
        print(f"[CHECK] {case}: three arms complete, initial state matched, finite logs, ZM/bound invariant passed", flush=True)
    write_csv(output / "ablation_summary.csv", summaries)
    write_csv(output / "ablation_comparisons.csv", comparisons)
    plot(output, trajectories)
    (output / "analysis_source_sha256.json").write_text(json.dumps({p.name: sha256(p) for p in (Path(__file__), ROOT / "scripts/residual_control_metrics.py")}, indent=2), encoding="utf-8")
    print(f"[RESULT] {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("stress", "rough", "combined", "nominal", "fast", "heavy", "offset", "all"), default="stress")
    parser.add_argument("--duration", type=float, default=12.)
    parser.add_argument("--warmup", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--phase_x", type=float, default=1.17)
    parser.add_argument("--phase_y", type=float, default=-2.03)
    parser.add_argument("--checkpoint", type=Path, default=E1)
    parser.add_argument("--log_dir", type=Path, default=ROOT / "logs/v7_6_e2_2/stress_12s")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()
    output = args.log_dir.resolve()
    if args.analyze_only:
        analyze(output, json.loads((output / "manifest.json").read_text(encoding="utf-8")))
        return
    if not np.isfinite([args.duration, args.warmup, args.phase_x, args.phase_y]).all() or args.duration <= 0 or args.warmup < 0:
        parser.error("Invalid duration/warmup/phase")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty; choose another directory or --analyze_only")
    cases = ["rough", "combined"] if args.case == "stress" else ["nominal", "fast", "rough", "heavy", "offset", "combined"] if args.case == "all" else [args.case]
    manifest = {
        "cases": cases, "duration_s": args.duration, "warmup_s": args.warmup,
        "seed": args.seed, "phase_x": args.phase_x, "phase_y": args.phase_y,
        "task": TASKS["E1"], "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "mapping": "Clip policy mean to [-1,1] as before; ZM: center, then divide by max(1,maxabs(centered)). Environment unchanged. Applied residual limit +/-3mm. Scaling attenuates differential magnitude and is explicitly logged.",
        "windows": "0s includes startup; the second window begins at warmup_s. Command rates exclude first transition inside each window. Cargo slip is incremental within the window. Contact is an analytical proxy.",
        "source_sha256": {p.name: sha256(p) for p in (Path(__file__), ROOT / "scripts/residual_common_mode.py", ROOT / "scripts/residual_checkpoint_eval.py", ROOT / "scripts/residual_control_metrics.py")},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for case in cases:
        command = [sys.executable, "-u", str(ROOT / "scripts/residual_checkpoint_eval.py"),
                   "--task", TASKS["E1"], "--checkpoint", str(checkpoint), "--case", case,
                   "--duration", str(args.duration), "--seed", str(args.seed),
                   "--phase_x", str(args.phase_x), "--phase_y", str(args.phase_y),
                   "--log_dir", str(output / case), "--common_mode_ablation"]
        if args.headless:
            command.append("--headless")
        print(f"[RUN] {case}: E / E1 / E1-ZM", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    analyze(output, manifest)


if __name__ == "__main__":
    main()
