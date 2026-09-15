"""Run paired E/E1 and E/E2 control diagnostics, then summarize and plot.

Each child launches its own Isaac Sim process. Only the evaluator's optional
read-only capture is enabled; rewards, policy actions and dynamics are intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from residual_control_metrics import analyze_channel, analyze_group


ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "logs/skrl/agv_level_residual_smooth_direct/2026-09-14_15-10-40_ppo_torch_v7_6_e1_residual_ppo_smooth_reward/checkpoints/best_agent.pt"
E2 = ROOT / "logs/skrl/agv_level_residual_action_penalty_direct/2026-09-14_16-21-35_ppo_torch_v7_6_e2_residual_ppo_action_penalty/checkpoints/best_agent.pt"
TASKS = {
    "E1": "Template-Agv-Level-Residual-Smooth-Direct-v0",
    "E2": "Template-Agv-Level-Residual-ActionPenalty-Direct-v0",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_case(output, case, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"E_zero": "#666666", "E1": "#1976b9", "E2": "#d15a29"}
    fig, axes = plt.subplots(5, 3, figsize=(15, 13), sharex=True)
    for index in range(3):
        ch = index + 1
        for label, rows in trajectories.items():
            t = np.array([float(r["command_time_s"]) for r in rows])
            def series(key):
                return np.array([float(r[key]) for r in rows])
            color = colors[label]
            axes[0, index].plot(t, series(f"residual{ch}_m") * 1000, color=color, label=label, lw=1)
            axes[1, index].plot(t, series(f"lift{ch}_base_target_m") * 1000, color=color, lw=1)
            axes[2, index].plot(t, series(f"lift{ch}_feedback_axis_height_m") * 1000, color=color, lw=1)
            axes[3, index].plot(t, series(f"lift{ch}_final_target_m") * 1000, color=color, lw=1)
            axes[3, index].plot(t, series(f"lift{ch}_actual_height_pre_m") * 1000, color=color, ls="--", lw=0.8, alpha=0.6)
            axes[4, index].plot(t + series("control_dt_s"), series(f"lift{ch}_actual_interval_velocity_m_s") * 1000, color=color, lw=0.8)
        axes[0, index].set_title(f"Lift {ch}")
        axes[0, index].axhline(3, color="black", ls=":", lw=0.7)
        axes[0, index].axhline(-3, color="black", ls=":", lw=0.7)
        axes[-1, index].set_xlabel("Time (s)")
    for row, label in enumerate(("Residual (mm)", "Base target (mm)", "Feedback / axis-z (mm)", "Final / actual pre (mm)", "Interval velocity (mm/s)")):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="best", ncol=3)
    fig.suptitle(f"{case}: control decomposition | E shown from E1 pair\nCommands at interval start; dashed actual height at start; velocity at interval end")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / f"{case}_control_decomposition.png", dpi=160)
    fig.savefig(output / f"{case}_control_decomposition.pdf")
    plt.close(fig)


def analyze(output, manifest):
    channels, events, summaries, groups = [], [], [], []
    common_mode_plots = {}
    for case in manifest["cases"]:
        plotted = {}
        signatures = []
        for variant in ("E1", "E2"):
            directory = output / f"{variant.lower()}_{case}"
            rows = read_csv(directory / "residual_checkpoint_trajectories.csv")
            summary = read_csv(directory / "residual_checkpoint_summary.csv")
            for item in summary:
                if float(item["ef_initial_state_max_abs_difference"]) > 1e-6:
                    raise RuntimeError("Unmatched E/F initial state")
                summaries.append({"variant": variant, **item})
            for controller in ("E_zero", "F_ppo"):
                group = [row for row in rows if row["controller"] == controller]
                expected = int(np.ceil(manifest["duration_s"] / float(group[0]["control_dt_s"]) - 1e-9))
                if len(group) != expected:
                    raise RuntimeError(f"{variant}/{case}/{controller}: incomplete run")
                for row in group:
                    for key, value in row.items():
                        if key in ("case", "controller"):
                            continue
                        if not np.isfinite(float(value)):
                            raise RuntimeError(f"Nonfinite {key}")
                if controller == "E_zero":
                    signatures.append([float(group[0][k]) for k in ("board_roll_pre_rad", "board_pitch_pre_rad", "lift1_actual_height_pre_m", "lift2_actual_height_pre_m", "lift3_actual_height_pre_m")])
                label = variant if controller == "F_ppo" else f"E_zero_{variant}_pair"
                for warmup in sorted({0.0, manifest["warmup_s"]}):
                    if warmup >= manifest["duration_s"] - 2 * float(group[0]["control_dt_s"]):
                        continue
                    groups.append({"case": case, "policy": label, **analyze_group(group, warmup, manifest["deadband_m"])})
                    for ch in (1, 2, 3):
                        result, runs = analyze_channel(group, ch, warmup, manifest["deadband_m"])
                        channels.append({"case": case, "policy": label, **result})
                        events.extend({"case": case, "policy": label, "channel": ch, "warmup_s": warmup, **run} for run in runs)
                if controller == "F_ppo":
                    plotted[variant] = group
                elif variant == "E1":
                    plotted["E_zero"] = group
        # Full signatures are checked inside each pair. Cross-pair selected
        # start states are also reported without asserting identical trajectories.
        print(f"[CHECK] {case} cross-pair initial selected state max difference={np.max(np.abs(np.array(signatures[0])-np.array(signatures[1]))):.3e}", flush=True)
        plot_case(output, case, plotted)
        common_mode_plots[case] = plotted
    plot_common_mode(output, common_mode_plots)
    write_csv(output / "control_channel_summary.csv", channels)
    write_csv(output / "control_group_summary.csv", groups)
    if events:
        write_csv(output / "limit_events.csv", events)
    write_csv(output / "physical_summary.csv", summaries)
    (output / "analysis_source_sha256.json").write_text(json.dumps({
        p.name: sha256(p) for p in (Path(__file__), ROOT / "scripts/residual_control_metrics.py")
    }, indent=2), encoding="utf-8")
    (output / "analysis_definitions.json").write_text(json.dumps({
        "sampling": "Commands at command_time_s; pre-state before controller update; physical state at time_s after physics step.",
        "rates": "Differences of consecutive commands inside each analysis window; first/reset transition excluded. Raw action_rate columns include first transition.",
        "limit": "Same-polarity contiguous command intervals at abs(residual)>=0.003-1e-7 m; left/right window boundary censoring retained.",
        "switches": "Transitions per second (not cycles), hysteresis +/-deadband_m; sign carries across deadband.",
        "correlations": "Pearson within each Lift channel; undefined constant-series correlations are blank in CSV. Correlation does not establish causation.",
        "rate_identity": "Without final clamping, mean(final_rate^2)=mean(base_rate^2)+mean(residual_rate^2)+2mean(base_rate*residual_rate).",
        "group": "RMS pooled over time and three channels, not arithmetic mean of channel RMS. Common residual is the three-channel mean; differential residual subtracts that mean. Height change is last post-state minus first pre-state within the window.",
        "contact": "Analytical proxies, not contact sensors.",
    }, indent=2), encoding="utf-8")
    print(f"[RESULT] {len(channels)} channel/window rows, {len(events)} limit events; outputs: {output}", flush=True)


def plot_common_mode(output, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"E_zero": "#666666", "E1": "#1976b9", "E2": "#d15a29"}
    fig, axes = plt.subplots(2, len(cases), figsize=(7 * len(cases), 6), squeeze=False)
    for col, (case, trajectories) in enumerate(cases.items()):
        for label, rows in trajectories.items():
            times = np.array([float(r["command_time_s"]) for r in rows])
            heights = np.array([[float(r[f"lift{j}_actual_height_pre_m"]) for j in (1, 2, 3)] for r in rows])
            residuals = np.array([[float(r[f"residual{j}_m"]) for j in (1, 2, 3)] for r in rows])
            axes[0, col].plot(times, heights.mean(axis=1) * 1000, label=label, color=colors[label])
            axes[1, col].plot(times, residuals.mean(axis=1) * 1000, color=colors[label])
        axes[0, col].set_title(case)
        axes[0, col].legend(ncol=3)
        axes[1, col].set_xlabel("Command time (s)")
        for row in range(2):
            axes[row, col].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean actual Lift height (mm)")
    axes[1, 0].set_ylabel("Mean residual (mm)")
    fig.suptitle("Common-mode motion | mean of three local Lift-axis heights, not Board world Z")
    fig.tight_layout()
    fig.savefig(output / "common_mode_motion.png", dpi=160)
    fig.savefig(output / "common_mode_motion.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("rough", "combined", "all"), default="all")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--phase_x", type=float, default=1.17)
    parser.add_argument("--phase_y", type=float, default=-2.03)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--deadband_mm", type=float, default=0.05)
    parser.add_argument("--e1_checkpoint", type=Path, default=E1)
    parser.add_argument("--e2_checkpoint", type=Path, default=E2)
    parser.add_argument("--log_dir", type=Path, default=ROOT / "logs/v7_6_e2_1/decomposition_12s")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--analyze_only", action="store_true", help="Regenerate summaries/plots using the saved manifest and trajectories.")
    args = parser.parse_args()
    output = args.log_dir.resolve()
    if args.analyze_only:
        analyze(output, json.loads((output / "manifest.json").read_text(encoding="utf-8")))
        return
    if not np.isfinite([args.duration, args.phase_x, args.phase_y, args.warmup, args.deadband_mm]).all() or args.duration <= 0 or args.warmup < 0 or args.deadband_mm < 0:
        parser.error("Invalid duration, phase, warmup or deadband")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; use a new --log_dir or --analyze_only")
    checkpoints = {"E1": args.e1_checkpoint.resolve(), "E2": args.e2_checkpoint.resolve()}
    for path in checkpoints.values():
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cases": ["rough", "combined"] if args.case == "all" else [args.case],
        "duration_s": args.duration, "seed": args.seed, "phase_x": args.phase_x,
        "phase_y": args.phase_y, "warmup_s": args.warmup, "deadband_m": args.deadband_mm / 1000,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoints": {k: {"path": str(p), "sha256": sha256(p), "task": TASKS[k]} for k, p in checkpoints.items()},
        "source_sha256": {p.name: sha256(p) for p in (Path(__file__), ROOT / "scripts/residual_control_metrics.py", ROOT / "scripts/residual_checkpoint_eval.py")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for case in manifest["cases"]:
        for variant, checkpoint in checkpoints.items():
            command = [sys.executable, "-u", str(ROOT / "scripts/residual_checkpoint_eval.py"),
                       "--task", TASKS[variant], "--checkpoint", str(checkpoint),
                       "--case", case, "--duration", str(args.duration), "--seed", str(args.seed),
                       "--phase_x", str(args.phase_x), "--phase_y", str(args.phase_y),
                       "--log_dir", str(output / f"{variant.lower()}_{case}"), "--control_decomposition"]
            if args.headless:
                command.append("--headless")
            print(f"[RUN] {variant}/{case}: E zero and PPO, duration={args.duration}s", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    analyze(output, manifest)


if __name__ == "__main__":
    main()
