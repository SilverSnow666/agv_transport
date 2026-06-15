from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATH_PRESETS: dict[str, list[tuple[float, float]]] = {
    # V4.1 short path, kept for backward compatibility.
    "v41": [
        (0.00, 0.00),
        (0.85, 0.00),
        (1.20, 0.12),
        (1.55, 0.20),
        (1.80, 0.08),
        (1.95, 0.00),
    ],
    # Current D0A0d / D0A0e-style path from the latest cfg.
    "d0a0e": [
        (0.00, 0.00),
        (0.45, 0.00),
        (0.95, 0.28),
        (1.45, 0.45),
        (2.05, 0.20),
        (2.65, -0.25),
        (3.10, 0.00),
    ],
    # V5.0.x long S-like path used by the current symmetric turn-recruitment task.
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
    """Return local waypoints including the start point (0, 0)."""
    if text is None:
        if preset not in PATH_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Options: {sorted(PATH_PRESETS)}")
        return PATH_PRESETS[preset]

    parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("--waypoints must be a list/tuple of (x, y) pairs")

    waypoints: list[tuple[float, float]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each waypoint must be a pair: (x, y)")
        waypoints.append((float(item[0]), float(item[1])))

    if not waypoints:
        raise ValueError("At least one waypoint is required")
    if waypoints[0] != (0.0, 0.0):
        waypoints.insert(0, (0.0, 0.0))
    return waypoints




def load_reference_path_csv(path: str | None, trajectory_csv: Path | None = None) -> list[tuple[float, float]] | None:
    """Load local path points from reference_path_local.csv.

    If path is None, try <trajectory_csv parent>/reference_path_local.csv.
    Returns None when no file is available.
    """
    ref_path: Path | None = None
    if path:
        ref_path = Path(path)
    elif trajectory_csv is not None:
        candidate = trajectory_csv.parent / "reference_path_local.csv"
        if candidate.exists():
            ref_path = candidate
    if ref_path is None or not ref_path.exists():
        return None
    ref_df = pd.read_csv(ref_path)
    if not {"x", "y"}.issubset(ref_df.columns):
        raise ValueError(f"Reference path CSV must contain x/y columns: {ref_path}")
    return [(float(row.x), float(row.y)) for row in ref_df.itertuples(index=False)]


def trim_post_reset_rows(ep: pd.DataFrame) -> pd.DataFrame:
    """Remove trailing rows produced after automatic env reset.

    Older evaluation logs read base_env state after env.step(). When a success happens,
    Isaac Lab may already have reset the environment, so the final trajectory row jumps
    back to the start and progress becomes ~0. This function removes that trailing reset
    tail when path_progress/path_progress_ratio columns are present.
    """
    progress_col = "path_progress"
    if progress_col not in ep.columns or len(ep) < 3:
        return ep
    progress = ep[progress_col].to_numpy(dtype=np.float64)
    max_progress = float(np.nanmax(progress))
    if max_progress <= 1e-6:
        return ep
    max_idx = int(np.nanargmax(progress))
    # Find the first strong progress drop after meaningful progress has been reached.
    for i in range(max_idx + 1, len(progress)):
        if progress[i] < 0.15 * max_progress:
            return ep.iloc[:i].reset_index(drop=True)
    return ep


def get_episode_df(df: pd.DataFrame, episode: int) -> pd.DataFrame:
    if "episode" not in df.columns:
        if episode != 0:
            raise ValueError("CSV has no 'episode' column, so only --episode 0 is valid")
        ep = df.copy()
    else:
        ep = df[df["episode"] == episode].copy()
    if ep.empty:
        available = sorted(df["episode"].unique().tolist()) if "episode" in df.columns else [0]
        raise ValueError(f"Episode {episode} not found. Available examples: {available[:10]}")
    return ep.reset_index(drop=True)


def infer_path_origin(ep: pd.DataFrame) -> tuple[float, float]:
    """Infer world-frame origin for local planned path.

    The environment uses payload_init_pos[:2] = (0, 0), so the payload position at
    the first recorded step is normally the world-frame path origin. If your CSV
    includes env_origin_x/env_origin_y, those are preferred.
    """
    if {"env_origin_x", "env_origin_y"}.issubset(ep.columns):
        return float(ep["env_origin_x"].iloc[0]), float(ep["env_origin_y"].iloc[0])
    return float(ep["payload_x"].iloc[0]), float(ep["payload_y"].iloc[0])


def local_to_world_path(
    waypoints_local: list[tuple[float, float]], origin_x: float, origin_y: float
) -> np.ndarray:
    pts = np.asarray(waypoints_local, dtype=np.float64)
    pts[:, 0] += origin_x
    pts[:, 1] += origin_y
    return pts


def point_to_polyline_metrics(points: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute lateral distance and projected progress for each point.

    Args:
        points: [N, 2] world-frame payload points.
        path: [M, 2] world-frame planned polyline.

    Returns:
        lateral_error: [N] shortest distance to the polyline.
        progress: [N] arc-length coordinate of nearest projection.
    """
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

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ax.plot(
        path_world[:, 0],
        path_world[:, 1],
        "-.",
        marker="o",
        linewidth=2,
        label="planned path",
    )
    ax.plot(
        ep["payload_x"],
        ep["payload_y"],
        linewidth=2,
        label="payload trajectory",
    )

    if show_target and {"target_x", "target_y"}.issubset(ep.columns):
        ax.plot(
            ep["target_x"],
            ep["target_y"],
            "--",
            linewidth=1.5,
            label="lookahead target",
        )

    if show_agv:
        for idx in (1, 2, 3):
            x_col, y_col = f"agv{idx}_x", f"agv{idx}_y"
            if {x_col, y_col}.issubset(ep.columns):
                ax.plot(
                    ep[x_col],
                    ep[y_col],
                    linewidth=1.0,
                    alpha=0.65,
                    label=f"AGV{idx}",
                )

    ax.scatter(payload_points[0, 0], payload_points[0, 1], s=70, label="start")
    ax.scatter(payload_points[-1, 0], payload_points[-1, 1], s=70, label="end")
    ax.scatter(path_world[-1, 0], path_world[-1, 1], s=90, marker="*", label="goal")

    title = (
        f"Payload trajectory, episode {episode}\n"
        f"mean lateral={metrics['lateral_error_mean']:.3f} m, "
        f"max lateral={metrics['lateral_error_max']:.3f} m, "
        f"final dist={metrics['final_dist_to_goal']:.3f} m"
    )
    ax.set_title(title)
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
    """Plot columns that exist in the trajectory DataFrame."""
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


def plot_role_dashboard(ep: pd.DataFrame, out_path: Path, episode: int) -> dict[str, float]:
    """Plot role-switching diagnostics for one episode.

    This dashboard is designed for V5.0.x symmetric turn-recruitment evaluation.
    It uses only columns that exist, so it remains compatible with older CSVs.
    """
    if "step_in_episode" in ep.columns:
        x = ep["step_in_episode"].to_numpy(dtype=np.float64)
    else:
        x = np.arange(len(ep), dtype=np.float64)

    metrics: dict[str, float] = {"episode": float(episode), "num_steps": float(len(ep))}

    if "active_goal_idx" in ep.columns:
        after_wp2 = ep["active_goal_idx"].to_numpy(dtype=np.float64) >= 2.0
        if np.any(after_wp2) and "agv1_contact" in ep.columns:
            metrics["agv1_contact_after_wp2_ratio"] = float(ep.loc[after_wp2, "agv1_contact"].astype(float).mean())
        if np.any(after_wp2) and "agv1_push_utility" in ep.columns:
            metrics["agv1_push_after_wp2_mean"] = float(ep.loc[after_wp2, "agv1_push_utility"].astype(float).mean())

    for idx in (1, 2, 3):
        col = f"agv{idx}_push_utility"
        if col in ep.columns:
            metrics[f"agv{idx}_push_utility_mean"] = float(ep[col].astype(float).mean())
        col = f"agv{idx}_contact"
        if col in ep.columns:
            metrics[f"agv{idx}_contact_ratio"] = float(ep[col].astype(float).mean())
        col = f"agv{idx}_payload_dist"
        if col in ep.columns:
            metrics[f"agv{idx}_payload_dist_max"] = float(ep[col].astype(float).max())

    if {"turn_state", "agv2_push_utility", "agv3_push_utility"}.issubset(ep.columns):
        right = ep["turn_state"].astype(int) < 0
        left = ep["turn_state"].astype(int) > 0
        if right.any():
            metrics["right_turn_agv2_minus_agv3_push_mean"] = float(
                (ep.loc[right, "agv2_push_utility"].astype(float) - ep.loc[right, "agv3_push_utility"].astype(float)).mean()
            )
        if left.any():
            metrics["left_turn_agv3_minus_agv2_push_mean"] = float(
                (ep.loc[left, "agv3_push_utility"].astype(float) - ep.loc[left, "agv2_push_utility"].astype(float)).mean()
            )

    fig, axes = plt.subplots(6, 1, figsize=(12.0, 15.0), sharex=True)

    _plot_available_series(
        axes[0], ep, x,
        ["path_progress_ratio", "path_lateral_error", "active_goal_idx", "turn_state"],
        "Path progress / lateral error / active goal / turn state",
    )
    _plot_available_series(
        axes[1], ep, x,
        ["agv1_push_utility", "agv2_push_utility", "agv3_push_utility", "two_pusher_gate"],
        "Push utility and two-pusher gate",
    )
    _plot_available_series(
        axes[2], ep, x,
        ["agv1_contact", "agv2_contact", "agv3_contact"],
        "Contact flags",
    )
    _plot_available_series(
        axes[3], ep, x,
        ["action_v1", "action_v2", "action_v3", "action_w1", "action_w2", "action_w3"],
        "Actions",
    )
    _plot_available_series(
        axes[4], ep, x,
        ["agv1_payload_dist", "agv2_payload_dist", "agv3_payload_dist"],
        "AGV-payload distances",
    )
    _plot_available_series(
        axes[5], ep, x,
        ["agv1_heading_parallel", "agv2_heading_parallel", "agv3_heading_parallel", "agv1_heading_center", "agv2_heading_center", "agv3_heading_center"],
        "Heading diagnostics",
    )

    axes[-1].set_xlabel("step in episode")
    fig.suptitle(f"Role-switching diagnostics, episode {episode}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot payload trajectory against the planned path from trajectory.csv."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to trajectory.csv")
    parser.add_argument("--episode", type=int, default=0, help="Episode id to plot")
    parser.add_argument(
        "--preset",
        type=str,
        default="d0a0e",
        choices=sorted(PATH_PRESETS),
        help="Built-in local waypoint preset",
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help="Optional Python literal list of local waypoints, e.g. '[(0,0),(0.45,0),(3.10,0)]'",
    )
    parser.add_argument(
        "--reference-path-csv",
        type=str,
        default=None,
        help="Optional reference_path_local.csv from the eval script. If omitted, auto-detect beside trajectory.csv.",
    )
    parser.add_argument(
        "--no-trim-reset",
        action="store_true",
        help="Do not trim trailing post-reset rows from old evaluation logs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Default: <csv stem>_episode_<id>_trajectory.png",
    )
    parser.add_argument("--show-agv", action="store_true", help="Overlay AGV trajectories if columns exist")
    parser.add_argument("--no-target", action="store_true", help="Do not plot lookahead target trajectory")
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Optional path to save one-row metrics CSV",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Also save a role-switching diagnostic dashboard for the selected episode.",
    )
    parser.add_argument(
        "--dashboard-out",
        type=str,
        default=None,
        help="Output PNG path for the role-switching dashboard.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    ep = get_episode_df(df, args.episode)
    if not args.no_trim_reset:
        ep = trim_post_reset_rows(ep)

    reference_waypoints = load_reference_path_csv(args.reference_path_csv, csv_path)
    if reference_waypoints is not None and args.waypoints is None:
        waypoints_local = reference_waypoints
    else:
        waypoints_local = parse_waypoints(args.waypoints, args.preset)
    origin_x, origin_y = infer_path_origin(ep)
    path_world = local_to_world_path(waypoints_local, origin_x, origin_y)

    out_path = Path(args.out) if args.out else csv_path.with_name(
        f"{csv_path.stem}_episode_{args.episode}_trajectory.png"
    )

    metrics = plot_episode(
        ep=ep,
        path_world=path_world,
        out_path=out_path,
        episode=args.episode,
        show_agv=args.show_agv,
        show_target=not args.no_target,
    )

    print(f"Saved plot: {out_path}")
    for key, value in metrics.items():
        if key == "episode":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")

    if args.dashboard:
        dashboard_out = Path(args.dashboard_out) if args.dashboard_out else csv_path.with_name(
            f"{csv_path.stem}_episode_{args.episode}_role_dashboard.png"
        )
        dashboard_metrics = plot_role_dashboard(ep=ep, out_path=dashboard_out, episode=args.episode)
        metrics.update(dashboard_metrics)
        print(f"Saved dashboard: {dashboard_out}")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
