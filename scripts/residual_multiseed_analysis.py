"""Aggregate matched V7.6-G multi-phase evaluations across training seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("roll", "pitch", "board_rp_speed", "lift_speed", "cargo_slip")
SAFETY_KEYS = (
    "early_termination",
    "board_dropped",
    "board_tipped",
    "cargo_dropped",
    "cargo_tipped",
    "support_loss_fraction_proxy",
    "residual_saturation_fraction",
)


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


def _saved_seed(checkpoint: Path) -> int:
    agent_cfg = checkpoint.parent.parent / "params/agent.yaml"
    config = yaml.safe_load(agent_cfg.read_text(encoding="utf-8"))
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Invalid seed in {agent_cfg}: {seed!r}")
    return seed


def _load_run(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if _sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    saved_seed = _saved_seed(checkpoint)
    manifest_seed = manifest.get("training_seed", saved_seed)
    if int(manifest_seed) != saved_seed:
        raise RuntimeError(f"Training-seed mismatch in {directory}")
    rows = _read_csv(directory / "multiphase_runs.csv")
    expected = len(manifest["phases"]) * len(manifest["cases"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows in {directory}, found {len(rows)}")
    for row in rows:
        row["training_seed"] = saved_seed
        row["source_directory"] = str(directory)
        for key, value in row.items():
            if key not in {"phase", "case", "source_directory"}:
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    raise RuntimeError(f"Non-finite {key}={value!r} in {directory}")
    return manifest, rows


def _aggregate(rows: list[dict]) -> list[dict]:
    aggregates: list[dict] = []
    seeds = sorted({int(row["training_seed"]) for row in rows})
    cases = sorted({row["case"] for row in rows})
    scopes = [
        ("all_seeds", "all", rows),
        *(("all_seeds", case, [row for row in rows if row["case"] == case]) for case in cases),
    ]
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["training_seed"]) == seed]
        scopes.append((str(seed), "all", seed_rows))
        for case in cases:
            scopes.append((str(seed), case, [row for row in seed_rows if row["case"] == case]))
    for seed_scope, case_scope, selected in scopes:
        for metric in METRICS:
            values = np.array(
                [float(row[f"{metric}_improvement_pct"]) for row in selected], dtype=float
            )
            aggregates.append({
                "training_seed": seed_scope,
                "case": case_scope,
                "metric": metric,
                "samples": len(values),
                "mean_improvement_pct": float(values.mean()),
                "sample_std_pct": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min_improvement_pct": float(values.min()),
                "max_improvement_pct": float(values.max()),
                "positive_fraction": float(np.mean(values > 0.0)),
            })
    return aggregates


def _plot(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({int(row["training_seed"]) for row in rows})
    titles = {
        "roll": "Roll RMS improvement",
        "pitch": "Pitch RMS improvement",
        "board_rp_speed": "Board RP speed improvement",
        "lift_speed": "Lift speed improvement",
        "cargo_slip": "Cargo slip improvement",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, metric in zip(axes.flat, METRICS, strict=False):
        grouped = [
            [float(row[f"{metric}_improvement_pct"]) for row in rows
             if int(row["training_seed"]) == seed]
            for seed in seeds
        ]
        means = [float(np.mean(values)) for values in grouped]
        stds = [float(np.std(values, ddof=1)) for values in grouped]
        axis.errorbar(seeds, means, yerr=stds, marker="o", capsize=4, color="#2474a6")
        for seed, values in zip(seeds, grouped, strict=True):
            axis.scatter([seed] * len(values), values, alpha=0.45, color="#d15a29", s=22)
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_title(titles[metric])
        axis.set_xlabel("Training seed")
        axis.set_ylabel("Improvement over matched E (%)")
        axis.set_xticks(seeds)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("V7.6-G E1 reproducibility across independent training seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(inputs: list[Path], output: Path) -> None:
    manifests: list[dict] = []
    rows: list[dict] = []
    for directory in inputs:
        manifest, run_rows = _load_run(directory.resolve())
        manifests.append(manifest)
        rows.extend(run_rows)
    seeds = [int(manifest.get("training_seed", _saved_seed(Path(manifest["checkpoint"]))))
             for manifest in manifests]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"Training seeds are not unique: {seeds}")
    reference_design = (
        manifests[0]["duration_s"], manifests[0]["cases"], manifests[0]["phases"]
    )
    for manifest in manifests[1:]:
        design = (manifest["duration_s"], manifest["cases"], manifest["phases"])
        if design != reference_design:
            raise RuntimeError("Evaluation designs differ across training seeds")
    if any(any(float(row[key]) != 0.0 for key in SAFETY_KEYS) for row in rows):
        raise RuntimeError("Safety, support-proxy, or target-saturation event detected")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "multiseed_runs.csv", rows)
    _write_csv(output / "multiseed_aggregate.csv", _aggregate(rows))
    (output / "manifest.json").write_text(json.dumps({
        "training_seeds": sorted(seeds),
        "input_directories": [str(path.resolve()) for path in inputs],
        "evaluation_duration_s": reference_design[0],
        "cases": reference_design[1],
        "phases": reference_design[2],
        "note": (
            "Training seeds are read from each saved agent.yaml. Evaluation reset seeds "
            "and terrain phases are matched across policies. Support/contact is analytical."
        ),
    }, indent=2), encoding="utf-8")
    _plot(output / "multiseed_reproducibility.png", rows)
    print(f"[RESULT] Training seeds: {sorted(seeds)}", flush=True)
    print(f"[RESULT] Paired comparisons: {len(rows)}", flush=True)
    print(f"[RESULT] Multi-seed outputs: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/v7_6_g/multiseed_aggregate")
    args = parser.parse_args()
    if len(args.input) < 2:
        parser.error("At least two --input evaluation directories are required")
    analyze(args.input, args.output.resolve())


if __name__ == "__main__":
    main()
