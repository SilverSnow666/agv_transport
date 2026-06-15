"""Plot V5 trajectory and compact role/wall diagnostics.

The script reads the CSVs produced by eval_three_agv_ppo_v5.py. It remains
compatible with older trajectory.csv files by plotting only columns that exist.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATH_PRESETS: dict[str, list[tuple[float, float]]] = {
    "v501": [
        (0.00, 0.00),
        (0.80, 0.00),
        (1.60, 0.00),
        (2.40, -0.40),
        (3.00, -0.80),
        (3.60, -0.80),
        (4.20, -0.40),
        (4.80, 0.00),
        (5.60, 0.00),
    ],
}


def parse_waypoints(text: str | None, preset: str) -> list[tuple[float, float]]:
    """Return local path points including the start point."""
    if text is None:
        if preset not in PATH_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Options: {sorted(PATH_PRESETS)}")
        return PATH_PRESETS[preset]

    parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("--waypoints must be a list/tuple of (x, y) pairs")

    points: list[tuple[float, float]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each waypoint must be a pair: (x, y)")
        points.append((float(item[0]), float(item[1])))

    if not points:
        raise ValueError("At least one waypoint is required")
    if points[0] != (0.0, 0.0):
        points.insert(0, (0.0, 0.0))
    return points


def load_xy_csv(path: Path | None) -> list[tuple[float, float]] | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"CSV must contain x/y columns: {path}")
    return [(float(row.x), float(row.y)) for row in df.itertuples(index=False)]


def load_reference_path(path: str | None, trajectory_csv: Path) -> list[tuple[float, float]] | None:
    ref_path = Path(path) if path else trajectory_csv.parent / "reference_path_local.csv"
    return load_xy_csv(ref_path)


def load_boundary_paths(trajectory_csv: Path, left_path: str | None, right_path: str | None):
    left = load_xy_csv(Path(left_path) if left_path else trajectory_csv.parent / "boundary_left_local.csv")
    right = load_xy_csv(Path(right_path) if right_path else trajectory_csv.parent / "boundary_right_local.csv")
    return left, right


def get_episode_df(df: pd.DataFrame, episode: int) -> pd.DataFrame:
    if "episode" not in df.columns:
        if episode != 0:
            raise ValueError("CSV has no episode column; only --episode 0 is valid")
        ep = df.copy()
    else:
        ep = df[df["episode"] == episode].copy()
    if ep.empty:
        available = sorted(df["episode"].unique().tolist()) if "episode" in df.columns else [0]
        raise ValueError(f"Episode {episode} not found. Available examples: {available[:10]}")
    return ep.reset_index(drop=True)


def trim_post_reset_rows(ep: pd.DataFrame) -> pd.DataFrame:
    """Remove reset-tail rows when older eval logs contain them."""
    if "path_progress" not in ep.columns or len(ep) < 3:
        return ep
    progress = ep["path_progress"].to_numpy(dtype=np.float64)
    max_progress = float(np.nanmax(progress))
    if max_progress <= 1e-6:
        return ep
    max_idx = int(np.nanargmax(progress))
    for i in range(max_idx + 1, len(progress)):
        if progress[i] < 0.15 * max_progress:
            return ep.iloc[:i].reset_index(drop=True)
    return ep


def infer_path_origin(ep: pd.DataFrame) -> tuple[float, float]:
    if {"env_origin_x", "env_origin_y"}.issubset(ep.columns):
        return float(ep["env_origin_x"].iloc[0]), float(ep["env_origin_y"].iloc[0])
    return float(ep["payload_x"].iloc[0]), float(ep["payload_y"].iloc[0])


def local_to_world(points_local: list[tuple[float, float]], origin_x: float, origin_y: float) -> np.ndarray:
    pts = np.asarray(points_local, dtype=np.float64)
    pts[:, 0] += origin_x
    pts[:, 1] += origin_y
    return pts


def point_to_polyline_metrics(points: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lateral distance and projected progress of points on a polyline."""
    if path.shape[0] < 2:
        raise ValueError("Path must contain at least two points")
    seg_start = path[:-1]
    seg_end = path[1:]
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len_safe = np.maximum(seg_len, 1e-9)
    seg_len_sq = np.maximum(seg_len_safe**2, 1e-12)
    cumulative_start = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])

    lateral = np.empty(points.shape[0], dtype=np.float64)
    progress = np.empty(points.shape[0], dtype=np.float64)
    for i, p in enumerate(points):
        rel = p[None, :] - seg_start
        t = np.sum(rel * seg_vec, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj = seg_start + t[:, None] * seg_vec
        d = np.linalg.norm(p[None, :] - proj, axis=1)
        best = int(np.argmin(d))
        lateral[i] = d[best]
        progress[i] = cumulative_start[best] + t[best] * seg_len_safe[best]
    return lateral, progress


def plot_episode(
    ep: pd.DataFrame,
    path_world: np.ndarray,
    left_boundary_world: np.ndarray | None,
    right_boundary_world: np.ndarray | None,
    out_path: Path,
    episode: int,
    show_agv: bool,
    show_target: bool,
) -> dict[str, float]:
    required = {"payload_x", "payload_y"}
    missing = required - set(ep.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    payload_points = ep[["payload_x", "payload_y"]].to_numpy(dtype=np.float64)
    lateral_error, progress = point_to_polyline_metrics(payload_points, path_world)
    total_path_len = float(np.sum(np.linalg.norm(np.diff(path_world, axis=0), axis=1)))
    final_dist = float(np.linalg.norm(payload_points[-1] - path_world[-1]))

    metrics = {
        "episode": float(episode),
        "num_steps": float(len(ep)),
        "path_length": total_path_len,
        "progress_final": float(progress[-1]),
        "progress_ratio_final": float(progress[-1] / max(total_path_len, 1e-9)),
        "lateral_error_mean": float(np.mean(lateral_error)),
        "lateral_error_max": float(np.max(lateral_error)),
        "final_dist_to_goal": final_dist,
    }
    for col in ("payload_wall_clearance", "agv1_wall_clearance", "agv2_wall_clearance", "agv3_wall_clearance"):
        if col in ep.columns:
            metrics[f"{col}_min"] = float(ep[col].astype(float).min())

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    if left_boundary_world is not None:
        ax.plot(left_boundary_world[:, 0], left_boundary_world[:, 1], linewidth=2.0, label="left boundary")
    if right_boundary_world is not None:
        ax.plot(right_boundary_world[:, 0], right_boundary_world[:, 1], linewidth=2.0, label="right boundary")
    ax.plot(path_world[:, 0], path_world[:, 1], "-.", marker="o", linewidth=2, label="planned path")
    ax.plot(ep["payload_x"], ep["payload_y"], linewidth=2.5, label="payload trajectory")

    if show_target and {"target_x", "target_y"}.issubset(ep.columns):
        ax.plot(ep["target_x"], ep["target_y"], "--", linewidth=1.5, label="target")

    if show_agv:
        for idx in (1, 2, 3):
            x_col, y_col = f"agv{idx}_x", f"agv{idx}_y"
            if {x_col, y_col}.issubset(ep.columns):
                ax.plot(ep[x_col], ep[y_col], linewidth=1.1, alpha=0.7, label=f"AGV{idx}")

    ax.scatter(payload_points[0, 0], payload_points[0, 1], s=70, label="start")
    ax.scatter(payload_points[-1, 0], payload_points[-1, 1], s=70, label="end")
    ax.scatter(path_world[-1, 0], path_world[-1, 1], s=90, marker="*", label="goal")
    ax.set_title(
        f"Payload trajectory, episode {episode}\n"
        f"mean lateral={metrics['lateral_error_mean']:.3f} m, "
        f"max lateral={metrics['lateral_error_max']:.3f} m, "
        f"final dist={metrics['final_dist_to_goal']:.3f} m"
    )
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return metrics


def _plot_available_series(ax, ep: pd.DataFrame, x: np.ndarray, columns: list[str], title: str) -> bool:
    plotted = False
    for col in columns:
        if col in ep.columns:
            ax.plot(x, ep[col].to_numpy(dtype=np.float64), linewidth=1.5, label=col)
            plotted = True
    ax.set_title(title)
    ax.grid(True)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    return plotted


def plot_dashboard(ep: pd.DataFrame, out_path: Path, episode: int, wall_near_margin: float) -> dict[str, float]:
    if "step_in_episode" in ep.columns:
        x = ep["step_in_episode"].to_numpy(dtype=np.float64)
    else:
        x = np.arange(len(ep), dtype=np.float64)

    metrics: dict[str, float] = {"episode": float(episode), "num_steps": float(len(ep))}
    if "active_goal_idx" in ep.columns and "agv1_contact" in ep.columns:
        after_wp2 = ep["active_goal_idx"].to_numpy(dtype=np.float64) >= 2.0
        if np.any(after_wp2):
            metrics["agv1_contact_after_wp2_ratio"] = float(ep.loc[after_wp2, "agv1_contact"].astype(float).mean())
    for col in ("payload_wall_clearance", "agv1_wall_clearance", "agv2_wall_clearance", "agv3_wall_clearance"):
        if col in ep.columns:
            metrics[f"{col}_min"] = float(ep[col].astype(float).min())
            metrics[f"{col}_near_ratio"] = float((ep[col].astype(float) < wall_near_margin).mean())

    fig, axes = plt.subplots(5, 1, figsize=(12.0, 13.0), sharex=True)
    _plot_available_series(
        axes[0],
        ep,
        x,
        ["path_progress_ratio", "path_lateral_error", "active_goal_idx", "turn_state"],
        "Path progress / lateral error / active goal / turn state",
    )
    _plot_available_series(
        axes[1],
        ep,
        x,
        ["agv1_push_utility", "agv2_push_utility", "agv3_push_utility", "two_pusher_gate"],
        "Push utility and two-pusher gate",
    )
    _plot_available_series(
        axes[2], ep, x, ["agv1_contact", "agv2_contact", "agv3_contact"], "Contact flags"
    )
    _plot_available_series(
        axes[3],
        ep,
        x,
        ["action_v1", "action_v2", "action_v3", "action_w1", "action_w2", "action_w3"],
        "Actions",
    )
    _plot_available_series(
        axes[4],
        ep,
        x,
        ["payload_wall_clearance", "agv1_wall_clearance", "agv2_wall_clearance", "agv3_wall_clearance"],
        "Approximate wall clearances",
    )
    axes[4].axhline(0.0, linewidth=1.0, linestyle="--", label="zero clearance")
    axes[4].axhline(wall_near_margin, linewidth=1.0, linestyle=":", label="near-wall threshold")
    axes[4].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("step in episode")
    fig.suptitle(f"V5.2 diagnostics, episode {episode}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V5 trajectory and physical-boundary diagnostics.")
    parser.add_argument("--csv", type=str, required=True, help="Path to trajectory.csv")
    parser.add_argument("--episode", type=int, default=0, help="Episode id to plot")
    parser.add_argument("--preset", type=str, default="v501", choices=sorted(PATH_PRESETS))
    parser.add_argument("--waypoints", type=str, default=None, help="Optional Python literal path points")
    parser.add_argument("--reference-path-csv", type=str, default=None, help="Optional reference_path_local.csv")
    parser.add_argument("--boundary-left-csv", type=str, default=None, help="Optional boundary_left_local.csv")
    parser.add_argument("--boundary-right-csv", type=str, default=None, help="Optional boundary_right_local.csv")
    parser.add_argument("--no-trim-reset", action="store_true", help="Do not trim trailing post-reset rows")
    parser.add_argument("--out", type=str, default=None, help="Output trajectory PNG path")
    parser.add_argument("--show-agv", action="store_true", help="Overlay AGV trajectories")
    parser.add_argument("--no-target", action="store_true", help="Do not plot target trajectory")
    parser.add_argument("--dashboard", action="store_true", help="Save compact diagnostic dashboard")
    parser.add_argument("--dashboard-out", type=str, default=None, help="Output dashboard PNG path")
    parser.add_argument("--metrics-out", type=str, default=None, help="Optional one-row metrics CSV")
    parser.add_argument("--wall-near-margin", type=float, default=0.10, help="Near-wall clearance threshold")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    ep = get_episode_df(df, args.episode)
    if not args.no_trim_reset:
        ep = trim_post_reset_rows(ep)

    reference_points = load_reference_path(args.reference_path_csv, csv_path)
    waypoints_local = reference_points if reference_points is not None and args.waypoints is None else parse_waypoints(args.waypoints, args.preset)
    left_local, right_local = load_boundary_paths(csv_path, args.boundary_left_csv, args.boundary_right_csv)

    origin_x, origin_y = infer_path_origin(ep)
    path_world = local_to_world(waypoints_local, origin_x, origin_y)
    left_world = local_to_world(left_local, origin_x, origin_y) if left_local is not None else None
    right_world = local_to_world(right_local, origin_x, origin_y) if right_local is not None else None

    out_path = Path(args.out) if args.out else csv_path.with_name(f"{csv_path.stem}_episode_{args.episode}_trajectory.png")
    metrics = plot_episode(
        ep=ep,
        path_world=path_world,
        left_boundary_world=left_world,
        right_boundary_world=right_world,
        out_path=out_path,
        episode=args.episode,
        show_agv=args.show_agv,
        show_target=not args.no_target,
    )
    print(f"Saved plot: {out_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if key != "episode" else f"{key}: {int(value)}")

    if args.dashboard:
        dashboard_path = Path(args.dashboard_out) if args.dashboard_out else csv_path.with_name(
            f"{csv_path.stem}_episode_{args.episode}_v52_dashboard.png"
        )
        dashboard_metrics = plot_dashboard(ep, dashboard_path, args.episode, args.wall_near_margin)
        metrics.update(dashboard_metrics)
        print(f"Saved dashboard: {dashboard_path}")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
