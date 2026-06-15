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
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    ep = get_episode_df(df, args.episode)

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

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
