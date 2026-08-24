"""Compare two episode_summary.csv files for payload CoM validation.

Example:

    python compare_com_eval_metrics.py \
      --baseline logs/eval/eval_com0/episode_summary.csv \
      --offset logs/eval/eval_com010/episode_summary.csv

Interpretation:
- If success/steps/yaw/reward/contact/distance statistics are nearly identical
  even under a very large CoM offset, the MassAPI CoM may not be affecting the
  PhysX dynamics.
- This script is a diagnostic helper, not a proof by itself.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

FIELDS = [
    "episode_steps",
    "final_goal_dist",
    "final_payload_yaw_abs",
    "path_lateral_error_mean",
    "path_lateral_error_max",
    "final_path_progress_ratio",
    "two_pusher_gate_mean",
    "agv1_contact_ratio",
    "agv2_contact_ratio",
    "agv3_contact_ratio",
    "agv1_payload_dist_max",
    "agv2_payload_dist_max",
    "agv3_payload_dist_max",
    "payload_wall_clearance_min",
    "agv1_wall_clearance_min",
    "agv2_wall_clearance_min",
    "agv3_wall_clearance_min",
    "episode_reward",
]
BOOL_FIELDS = ["success", "last_out_of_bounds", "last_bad_rear_push", "last_agv_escaped"]


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(row: dict[str, str], field: str) -> float | None:
    try:
        return float(row[field])
    except Exception:
        return None


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        vals = [safe_float(r, field) for r in rows]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if vals:
            out[field] = {
                "mean": mean(vals),
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }
    return out


def count_flags(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for field in BOOL_FIELDS:
        if field in rows[0]:
            out[field] = sum(parse_bool(r[field]) for r in rows)
    if "episode_steps" in rows[0]:
        out["steps_gt_1000"] = sum((safe_float(r, "episode_steps") or 0.0) > 1000.0 for r in rows)
    if "escaped_agv_idx" in rows[0]:
        out["escaped_agv_idx_not_minus1"] = sum(int(float(r["escaped_agv_idx"])) != -1 for r in rows)
    return out


def fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="episode_summary.csv with zero / smaller CoM offset")
    parser.add_argument("--offset", required=True, help="episode_summary.csv with larger CoM offset")
    args = parser.parse_args()

    base_rows = load_rows(Path(args.baseline))
    off_rows = load_rows(Path(args.offset))
    if not base_rows or not off_rows:
        raise RuntimeError("Both CSV files must contain at least one row")

    base = summarize(base_rows)
    off = summarize(off_rows)
    base_flags = count_flags(base_rows)
    off_flags = count_flags(off_rows)

    print("# CoM offset rollout comparison")
    print(f"baseline episodes: {len(base_rows)}")
    print(f"offset episodes:   {len(off_rows)}")
    print()
    print("## Flag counts")
    keys = sorted(set(base_flags) | set(off_flags))
    print("| flag | baseline | offset | delta |")
    print("|---|---:|---:|---:|")
    for k in keys:
        b = base_flags.get(k, 0)
        o = off_flags.get(k, 0)
        print(f"| {k} | {b} | {o} | {o-b:+d} |")

    print("\n## Numeric metrics")
    print("| metric | baseline mean | offset mean | delta mean | baseline max | offset max |")
    print("|---|---:|---:|---:|---:|---:|")
    for field in FIELDS:
        if field not in base or field not in off:
            continue
        b = base[field]
        o = off[field]
        print(
            f"| {field} | {fmt(b['mean'])} | {fmt(o['mean'])} | {fmt(o['mean']-b['mean'])} | "
            f"{fmt(b['max'])} | {fmt(o['max'])} |"
        )

    # Conservative heuristic.  It only warns that the rollout response is too similar.
    yaw_delta = abs((off.get("final_payload_yaw_abs", {}).get("mean", 0.0)) - (base.get("final_payload_yaw_abs", {}).get("mean", 0.0)))
    reward_delta = abs((off.get("episode_reward", {}).get("mean", 0.0)) - (base.get("episode_reward", {}).get("mean", 0.0)))
    steps_delta = abs((off.get("episode_steps", {}).get("mean", 0.0)) - (base.get("episode_steps", {}).get("mean", 0.0)))
    dist_delta = abs((off.get("agv1_payload_dist_max", {}).get("mean", 0.0)) - (base.get("agv1_payload_dist_max", {}).get("mean", 0.0)))

    print("\n## Diagnostic conclusion")
    if yaw_delta < 0.005 and reward_delta < 10.0 and steps_delta < 2.0 and dist_delta < 0.02:
        print(
            "The two rollouts are extremely similar. If the offset test used a large CoM "
            "such as (0.10, 0.10, 0.0) or (0.25, 0.25, 0.0), this suggests that the "
            "MassAPI CoM setting may not be producing a meaningful PhysX dynamics change."
        )
    else:
        print(
            "The larger CoM offset changed the rollout metrics. This suggests the CoM setting "
            "is likely affecting the dynamics, although visual checks and repeated runs are still recommended."
        )


if __name__ == "__main__":
    main()
