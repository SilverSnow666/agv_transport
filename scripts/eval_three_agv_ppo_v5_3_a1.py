"""Evaluate the current three-AGV policy for V5.2/V5.3 physical-boundary and irregular-payload tests.

This script is intentionally narrower than the older v4 evaluator:
- keep the metrics needed for V5.1C/V5.2-A decisions;
- remove low-value diagnostics such as formation error, heading-center averages,
  action-rate averages, and low-utility action summaries;
- add physical-boundary clearance metrics for the manual U-shaped corridor.

Outputs:
1. episode_summary.csv
2. trajectory.csv
3. reference_path_local.csv
4. boundary_left_local.csv / boundary_right_local.csv, when boundary points exist
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate three-AGV centralized PPO policy for V5.2/V5.3 validation."
)
parser.add_argument("--task", type=str, default="Template-Agv-Transport-Direct-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint .pt file.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=5000, help="Maximum total simulation steps.")
parser.add_argument("--output_dir", type=str, default="logs/eval_three_agv_ppo", help="Directory for CSV outputs.")
parser.add_argument(
    "--eval_contact_threshold",
    type=float,
    default=1.20,
    help="Fallback center-distance threshold for contact statistics when env contact flags are unavailable.",
)
parser.add_argument(
    "--wall_near_margin",
    type=float,
    default=0.10,
    help="Clearance threshold, in meters, used to count near-wall steps.",
)
parser.add_argument("--save_reference_path", action="store_true", default=True, help="Save local planned path CSV.")
parser.add_argument(
    "--no_save_reference_path",
    action="store_false",
    dest="save_reference_path",
    help="Disable writing reference_path_local.csv.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Remove custom argparse arguments before Hydra parses sys.argv.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import agv_transport.tasks  # noqa: F401, E402


def quat_to_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion in wxyz format to yaw."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def get_payload_quat(base_env) -> torch.Tensor:
    """Return payload root quaternion in wxyz format."""
    payload_data = base_env.payload.data
    if hasattr(payload_data, "root_quat_w"):
        return payload_data.root_quat_w
    return payload_data.root_state_w[:, 3:7]


def get_path_metrics(base_env):
    """Return target, lateral error, progress and segment index."""
    if hasattr(base_env, "_compute_path_tracking_quantities"):
        return base_env._compute_path_tracking_quantities()
    target_xy = base_env._get_target_xy()
    zeros = torch.zeros(base_env.num_envs, device=base_env.device)
    segment_idx = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    return target_xy, zeros, zeros, segment_idx


def get_final_target_xy(base_env) -> torch.Tensor:
    """Return final goal xy in world frame."""
    if hasattr(base_env, "_get_final_target_xy"):
        return base_env._get_final_target_xy()
    return base_env._get_target_xy()


def compute_total_path_length(base_env) -> torch.Tensor:
    """Compute the planned path length per environment."""
    if hasattr(base_env, "_get_path_points_xy"):
        path_points = base_env._get_path_points_xy()
        segment_vec = path_points[:, 1:, :] - path_points[:, :-1, :]
        return torch.linalg.norm(segment_vec, dim=2).sum(dim=1).clamp_min(1e-6)
    final_target_xy = get_final_target_xy(base_env)
    payload_init = torch.tensor(base_env.cfg.payload_init_pos[:2], device=base_env.device).view(1, 2)
    start_xy = base_env.scene.env_origins[:, :2] + payload_init
    return torch.linalg.norm(final_target_xy - start_xy, dim=1).clamp_min(1e-6)


def compute_contact_flags(base_env, payload_xy: torch.Tensor, threshold: float) -> torch.Tensor:
    """Compute AGV-payload contact flags, preferring the environment implementation."""
    if hasattr(base_env, "_compute_contact_flags"):
        return base_env._compute_contact_flags().bool()
    contact_flags = []
    for agv in base_env.agvs:
        agv_xy = agv.data.root_pos_w[:, :2]
        contact_flags.append(torch.linalg.norm(agv_xy - payload_xy, dim=1) < threshold)
    return torch.stack(contact_flags, dim=1)


def get_contact_geometry_for_eval(base_env, payload_xy: torch.Tensor):
    """Return center-heading, push-heading, front/rear distances, and v actions.

    Current V5 environments return five values, where the second value is the
    parallel heading used by the push reward. Older environments return four
    values; in that case center heading is also used as the push heading.
    """
    if hasattr(base_env, "_compute_contact_geometry"):
        geom = base_env._compute_contact_geometry(payload_xy)
        if len(geom) == 5:
            return geom
        if len(geom) == 4:
            center_heading, front_dists, rear_dists, v_actions = geom
            return center_heading, center_heading, front_dists, rear_dists, v_actions
        raise RuntimeError(f"Unexpected _compute_contact_geometry return length: {len(geom)}")

    payload_yaw = quat_to_yaw_wxyz(get_payload_quat(base_env))
    payload_heading_xy = torch.stack((torch.cos(payload_yaw), torch.sin(payload_yaw)), dim=1)
    half_length = 0.5 * float(getattr(base_env.cfg, "agv_size", (0.70, 0.45, 0.06))[0])

    center_heading_list = []
    parallel_heading_list = []
    front_dist_list = []
    rear_dist_list = []
    v_action_list = []
    for i, agv in enumerate(base_env.agvs):
        agv_xy = agv.data.root_pos_w[:, :2]
        agv_heading_xy = torch.stack(
            (torch.cos(base_env.agv_yaw[:, i]), torch.sin(base_env.agv_yaw[:, i])), dim=1
        )
        to_payload = payload_xy - agv_xy
        dir_to_payload = to_payload / torch.linalg.norm(to_payload, dim=1, keepdim=True).clamp_min(1e-6)
        center_heading = torch.sum(agv_heading_xy * dir_to_payload, dim=1)
        parallel_heading = torch.sum(agv_heading_xy * payload_heading_xy, dim=1)
        front_xy = agv_xy + agv_heading_xy * half_length
        rear_xy = agv_xy - agv_heading_xy * half_length
        center_heading_list.append(center_heading)
        parallel_heading_list.append(parallel_heading)
        front_dist_list.append(torch.linalg.norm(front_xy - payload_xy, dim=1))
        rear_dist_list.append(torch.linalg.norm(rear_xy - payload_xy, dim=1))
        v_action_list.append(base_env.actions[:, 2 * i])

    return (
        torch.stack(center_heading_list, dim=1),
        torch.stack(parallel_heading_list, dim=1),
        torch.stack(front_dist_list, dim=1),
        torch.stack(rear_dist_list, dim=1),
        torch.stack(v_action_list, dim=1),
    )


def compute_push_metrics(base_env, payload_xy: torch.Tensor, contact_flags: torch.Tensor, actions: torch.Tensor):
    """Compute V5-compatible push utility and two-pusher gate."""
    _, heading_for_push, _, _, _ = get_contact_geometry_for_eval(base_env, payload_xy)
    front_heading_min = float(getattr(base_env.cfg, "front_contact_heading_min", 0.20))
    front_facing_score = torch.clamp((heading_for_push - front_heading_min) / (1.0 - front_heading_min), 0.0, 1.0)
    v_actions = torch.stack((actions[:, 0], actions[:, 2], actions[:, 4]), dim=1)
    push_utility = contact_flags.float() * front_facing_score * torch.clamp(v_actions, min=0.0)
    sorted_push_utility, _ = torch.sort(push_utility, dim=1, descending=True)
    second_push_utility = sorted_push_utility[:, 1]
    top2_push_utility = sorted_push_utility[:, 0] + sorted_push_utility[:, 1]
    threshold = float(getattr(base_env.cfg, "two_pusher_gate_threshold", 0.20))
    two_pusher_gate = torch.clamp(second_push_utility / threshold, 0.0, 1.0)
    return push_utility, second_push_utility, top2_push_utility, two_pusher_gate


def compute_action_norm(actions: torch.Tensor) -> torch.Tensor:
    """Compute per-AGV normalized action magnitudes."""
    return torch.stack(
        (
            torch.linalg.norm(actions[:, 0:2], dim=1),
            torch.linalg.norm(actions[:, 2:4], dim=1),
            torch.linalg.norm(actions[:, 4:6], dim=1),
        ),
        dim=1,
    )


def get_active_goal_idx(base_env) -> torch.Tensor:
    """Return the current active waypoint/subgoal index."""
    if hasattr(base_env, "active_goal_idx"):
        return base_env.active_goal_idx.detach()
    if hasattr(base_env, "current_waypoint_idx"):
        return base_env.current_waypoint_idx.detach()
    return torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)


def get_active_segment_dir(base_env, path_segment_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Return active path segment direction for turn-role diagnostics."""
    if hasattr(base_env, "_get_active_segment_info"):
        try:
            _, _, active_segment_dir, _ = base_env._get_active_segment_info()
            return active_segment_dir.detach()
        except Exception:
            pass
    if hasattr(base_env, "_get_path_points_xy") and path_segment_idx is not None:
        path_points = base_env._get_path_points_xy()
        env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
        max_seg = path_points.shape[1] - 2
        seg_idx = torch.clamp(path_segment_idx.to(base_env.device, dtype=torch.long), 0, max_seg)
        seg_vec = path_points[env_ids, seg_idx + 1] - path_points[env_ids, seg_idx]
        return seg_vec / torch.linalg.norm(seg_vec, dim=1, keepdim=True).clamp_min(1e-6)
    out = torch.zeros((base_env.num_envs, 2), device=base_env.device)
    out[:, 0] = 1.0
    return out


def get_bool_buffer(base_env, name: str) -> torch.Tensor:
    """Read a boolean diagnostic buffer when available."""
    if hasattr(base_env, name):
        return getattr(base_env, name).detach().bool()
    return torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)


def _local_points_tensor(points: Iterable[Iterable[float]], device: torch.device) -> torch.Tensor:
    return torch.tensor([(float(x), float(y)) for x, y in points], device=device, dtype=torch.float32)


def get_boundary_local_points(base_env):
    """Return manual wall-centerline points from cfg, if present."""
    cfg = base_env.cfg
    if hasattr(cfg, "path_boundary_left_points") and hasattr(cfg, "path_boundary_right_points"):
        left = _local_points_tensor(cfg.path_boundary_left_points, base_env.device)
        right = _local_points_tensor(cfg.path_boundary_right_points, base_env.device)
        return left, right
    return None, None


def point_to_polyline_distance(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """Minimum 2D distance from each point [N,2] to a polyline [N,M,2]."""
    if polyline is None or polyline.shape[1] < 2:
        return torch.full((points.shape[0],), float("nan"), device=points.device)

    p0 = polyline[:, :-1, :]
    p1 = polyline[:, 1:, :]
    seg = p1 - p0
    seg_len_sq = torch.sum(seg * seg, dim=2).clamp_min(1e-12)
    rel = points[:, None, :] - p0
    t = torch.sum(rel * seg, dim=2) / seg_len_sq
    t = torch.clamp(t, 0.0, 1.0)
    proj = p0 + t[:, :, None] * seg
    return torch.linalg.norm(points[:, None, :] - proj, dim=2).min(dim=1).values


def compute_wall_clearances(base_env, payload_xy: torch.Tensor, agv_xy: torch.Tensor):
    """Approximate object clearances to manual physical boundary centerlines.

    The clearance is geometric and conservative enough for diagnostics:
    distance_to_wall_centerline - wall_thickness/2 - object_half_width.
    It is not used for rewards.
    """
    left_local, right_local = get_boundary_local_points(base_env)
    if left_local is None or right_local is None:
        nan_payload = torch.full((base_env.num_envs,), float("nan"), device=base_env.device)
        nan_agv = torch.full((base_env.num_envs, 3), float("nan"), device=base_env.device)
        return nan_payload, nan_agv

    origins = base_env.scene.env_origins[:, None, :2]
    left_world = origins + left_local[None, :, :]
    right_world = origins + right_local[None, :, :]
    wall_thickness = float(getattr(base_env.cfg, "path_boundary_wall_thickness", 0.08))
    payload_half_width = float(
        getattr(
            base_env.cfg,
            "payload_clearance_half_width_y",
            0.5 * float(getattr(base_env.cfg, "payload_size", (0.90, 1.20, 0.30))[1]),
        )
    )
    agv_half_width = 0.5 * float(getattr(base_env.cfg, "agv_size", (0.70, 0.45, 0.06))[1])

    payload_dist = torch.minimum(
        point_to_polyline_distance(payload_xy, left_world),
        point_to_polyline_distance(payload_xy, right_world),
    )
    payload_clearance = payload_dist - 0.5 * wall_thickness - payload_half_width

    agv_clearances = []
    for idx in range(3):
        agv_dist = torch.minimum(
            point_to_polyline_distance(agv_xy[:, idx, :], left_world),
            point_to_polyline_distance(agv_xy[:, idx, :], right_world),
        )
        agv_clearances.append(agv_dist - 0.5 * wall_thickness - agv_half_width)
    return payload_clearance, torch.stack(agv_clearances, dim=1)


def write_reference_path(output_dir: Path, base_env) -> None:
    """Write local reference path points for plotting."""
    reference_csv = output_dir / "reference_path_local.csv"
    start = tuple(getattr(base_env.cfg, "payload_init_pos", (0.0, 0.0, 0.0))[:2])
    waypoints = list(getattr(base_env.cfg, "waypoints", ()))
    with open(reference_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point_idx", "x", "y"])
        for idx, (x, y) in enumerate([start] + waypoints):
            writer.writerow([idx, float(x), float(y)])


def write_boundary_paths(output_dir: Path, base_env) -> None:
    """Write manual boundary centerlines for plotting, when available."""
    for name, points in (
        ("boundary_left_local.csv", getattr(base_env.cfg, "path_boundary_left_points", None)),
        ("boundary_right_local.csv", getattr(base_env.cfg, "path_boundary_right_points", None)),
    ):
        if points is None:
            continue
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["point_idx", "x", "y"])
            for idx, (x, y) in enumerate(points):
                writer.writerow([idx, float(x), float(y)])


def get_runner_action(runner: Runner, obs: torch.Tensor) -> torch.Tensor:
    """Get deterministic/evaluation action from skrl runner agent."""
    with torch.inference_mode():
        return runner.agent.act(obs, timestep=0, timesteps=0)[0]


def safe_mean(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0)


def scalar(value: torch.Tensor | float | bool | int) -> float | bool | int:
    if isinstance(value, torch.Tensor):
        value = value.item()
    return value


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Run evaluation and write compact V5.2-oriented CSV logs."""
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "episode_summary.csv"
    trajectory_csv = output_dir / "trajectory.csv"

    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if "experiment" in agent_cfg.get("agent", {}):
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = raw_env.unwrapped
    print(f"[INFO] Eval target_radius: {base_env.cfg.target_radius}")
    print(f"[INFO] Eval target_yaw_radius: {base_env.cfg.target_yaw_radius}")
    print(f"[INFO] Eval waypoints: {getattr(base_env.cfg, 'waypoints', None)}")
    print(f"[INFO] Physical boundaries: {getattr(base_env.cfg, 'enable_physical_path_boundaries', False)}")

    if args_cli.save_reference_path:
        write_reference_path(output_dir, base_env)
        write_boundary_paths(output_dir, base_env)

    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    print(f"[INFO] Loading PPO checkpoint from: {checkpoint_path}")
    runner.agent.load(str(checkpoint_path))
    if hasattr(runner, "set_running_mode"):
        runner.set_running_mode("eval")
    elif hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        print("[WARN] set_running_mode is not available. Continuing with inference_mode.")

    obs, _ = env.reset()

    num_envs = args_cli.num_envs
    device = base_env.device
    action_dim = int(getattr(base_env.cfg, "action_space", 6))
    step_dt = float(getattr(base_env, "step_dt", base_env.cfg.sim.dt * base_env.cfg.decimation))
    target_radius = float(base_env.cfg.target_radius)
    target_yaw_radius = float(base_env.cfg.target_yaw_radius)
    path_total_length = compute_total_path_length(base_env)

    active_episode_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    next_episode_id = 0
    for env_id in range(num_envs):
        if next_episode_id < args_cli.num_episodes:
            active_episode_id[env_id] = next_episode_id
            next_episode_id += 1

    episode_reward = torch.zeros(num_envs, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    path_lateral_error_sum = torch.zeros(num_envs, device=device)
    path_lateral_error_max = torch.zeros(num_envs, device=device)
    path_progress_max = torch.zeros(num_envs, device=device)
    path_progress_ratio_last = torch.zeros(num_envs, device=device)

    contact_steps = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
    contact_count_sum = torch.zeros(num_envs, device=device)
    two_pusher_gate_sum = torch.zeros(num_envs, device=device)
    two_pusher_gate_active_steps = torch.zeros(num_envs, device=device)
    push_utility_sum = torch.zeros((num_envs, 3), device=device)
    action_norm_sum = torch.zeros((num_envs, 3), device=device)

    right_turn_steps = torch.zeros(num_envs, device=device)
    left_turn_steps = torch.zeros(num_envs, device=device)
    right_turn_agv2_push_sum = torch.zeros(num_envs, device=device)
    right_turn_agv3_push_sum = torch.zeros(num_envs, device=device)
    left_turn_agv2_push_sum = torch.zeros(num_envs, device=device)
    left_turn_agv3_push_sum = torch.zeros(num_envs, device=device)

    after_wp2_steps = torch.zeros(num_envs, device=device)
    agv1_contact_after_wp2_steps = torch.zeros(num_envs, device=device)

    payload_wall_clearance_min = torch.full((num_envs,), float("inf"), device=device)
    agv_wall_clearance_min = torch.full((num_envs, 3), float("inf"), device=device)
    agv_payload_dist_max = torch.zeros((num_envs, 3), device=device)
    payload_near_wall_steps = torch.zeros(num_envs, device=device)
    agv_near_wall_steps = torch.zeros((num_envs, 3), device=device)

    prev_eval_actions = torch.zeros((num_envs, action_dim), device=device)
    completed_episodes = 0
    global_step = 0

    summary_header = [
        "episode",
        "env_id",
        "success",
        "episode_steps",
        "episode_time_s",
        "final_goal_dist",
        "final_payload_yaw_abs",
        "path_lateral_error_mean",
        "path_lateral_error_max",
        "final_path_progress_ratio",
        "max_path_progress_ratio",
        "two_pusher_gate_mean",
        "two_pusher_gate_active_ratio",
        "agv1_contact_ratio",
        "agv2_contact_ratio",
        "agv3_contact_ratio",
        "mean_contact_count",
        "agv1_contact_after_wp2_ratio",
        "right_turn_agv2_push_mean",
        "right_turn_agv3_push_mean",
        "left_turn_agv2_push_mean",
        "left_turn_agv3_push_mean",
        "agv1_push_utility_mean",
        "agv2_push_utility_mean",
        "agv3_push_utility_mean",
        "agv1_action_norm_mean",
        "agv2_action_norm_mean",
        "agv3_action_norm_mean",
        "payload_wall_clearance_min",
        "agv1_wall_clearance_min",
        "agv2_wall_clearance_min",
        "agv3_wall_clearance_min",
        "agv1_payload_dist_max",
        "agv2_payload_dist_max",
        "agv3_payload_dist_max",
        "escaped_agv_idx",
        "payload_near_wall_ratio",
        "agv1_near_wall_ratio",
        "agv2_near_wall_ratio",
        "agv3_near_wall_ratio",
        "last_position_success",
        "last_yaw_success",
        "last_out_of_bounds",
        "last_bad_rear_push",
        "last_agv_escaped",
        "episode_reward",
        "terminated",
        "truncated",
    ]

    trajectory_header = [
        "global_step",
        "episode",
        "env_id",
        "step_in_episode",
        "env_origin_x",
        "env_origin_y",
        "payload_x",
        "payload_y",
        "target_x",
        "target_y",
        "final_target_x",
        "final_target_y",
        "path_segment_idx",
        "active_goal_idx",
        "turn_state",
        "path_lateral_error",
        "path_progress",
        "path_progress_ratio",
        "final_goal_dist",
        "payload_yaw",
        "payload_wall_clearance",
        "agv1_x",
        "agv1_y",
        "agv1_payload_dist",
        "agv1_wall_clearance",
        "agv1_contact",
        "agv1_push_utility",
        "action_v1",
        "action_w1",
        "agv2_x",
        "agv2_y",
        "agv2_payload_dist",
        "agv2_wall_clearance",
        "agv2_contact",
        "agv2_push_utility",
        "action_v2",
        "action_w2",
        "agv3_x",
        "agv3_y",
        "agv3_payload_dist",
        "agv3_wall_clearance",
        "agv3_contact",
        "agv3_push_utility",
        "action_v3",
        "action_w3",
        "two_pusher_gate",
        "reward",
        "terminated",
        "truncated",
        "success",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as summary_file, open(
        trajectory_csv, "w", newline="", encoding="utf-8"
    ) as traj_file:
        summary_writer = csv.writer(summary_file)
        traj_writer = csv.writer(traj_file)
        summary_writer.writerow(summary_header)
        traj_writer.writerow(trajectory_header)

        while simulation_app.is_running() and global_step < args_cli.max_steps:
            if completed_episodes >= args_cli.num_episodes:
                break

            with torch.inference_mode():
                actions = torch.clamp(get_runner_action(runner, obs).to(device), -1.0, 1.0)

                # Capture state before env.step() to avoid post-reset logging.
                payload_xy = base_env.payload.data.root_pos_w[:, :2].clone()
                agv_xy = torch.stack(
                    (
                        base_env.agv1.data.root_pos_w[:, :2].clone(),
                        base_env.agv2.data.root_pos_w[:, :2].clone(),
                        base_env.agv3.data.root_pos_w[:, :2].clone(),
                    ),
                    dim=1,
                )
                env_origin_xy = base_env.scene.env_origins[:, :2].clone()
                target_xy, path_lateral_error, path_progress, path_segment_idx = get_path_metrics(base_env)
                target_xy = target_xy.clone()
                path_lateral_error = path_lateral_error.clone()
                path_progress = path_progress.clone()
                path_segment_idx = path_segment_idx.clone()
                final_target_xy = get_final_target_xy(base_env).clone()
                path_progress_ratio = torch.clamp(path_progress / path_total_length, 0.0, 1.0)
                active_goal_idx = get_active_goal_idx(base_env).clone()
                active_segment_dir = get_active_segment_dir(base_env, path_segment_idx).clone()

                turn_threshold = float(getattr(base_env.cfg, "turn_role_y_threshold", 0.08))
                right_turn_gate = active_segment_dir[:, 1] < -turn_threshold
                left_turn_gate = active_segment_dir[:, 1] > turn_threshold
                turn_state = torch.zeros(num_envs, dtype=torch.long, device=device)
                turn_state[right_turn_gate] = -1
                turn_state[left_turn_gate] = 1

                final_goal_dist = torch.linalg.norm(payload_xy - final_target_xy, dim=1)
                payload_yaw = quat_to_yaw_wxyz(get_payload_quat(base_env).clone())
                payload_yaw_abs = torch.abs(payload_yaw)
                contact_flags = compute_contact_flags(base_env, payload_xy, args_cli.eval_contact_threshold)
                contact_count = contact_flags.float().sum(dim=1)
                agv_payload_dists = torch.linalg.norm(agv_xy - payload_xy[:, None, :], dim=2)
                push_utility, _, _, two_pusher_gate = compute_push_metrics(base_env, payload_xy, contact_flags, actions)
                action_norm = compute_action_norm(actions)
                payload_clearance, agv_clearance = compute_wall_clearances(base_env, payload_xy, agv_xy)

                position_success = final_goal_dist < target_radius
                yaw_success = payload_yaw_abs < target_yaw_radius
                success_state = position_success & yaw_success

                obs, rewards, terminated, truncated, _ = env.step(actions)
                rewards_1d = rewards.reshape(-1).to(device)
                terminated_1d = terminated.reshape(-1).bool().to(device)
                truncated_1d = truncated.reshape(-1).bool().to(device)
                done_1d = terminated_1d | truncated_1d

                valid = active_episode_id >= 0
                episode_reward[valid] += rewards_1d[valid]
                episode_steps[valid] += 1
                path_lateral_error_sum[valid] += path_lateral_error[valid]
                path_lateral_error_max[valid] = torch.maximum(path_lateral_error_max[valid], path_lateral_error[valid])
                path_progress_max[valid] = torch.maximum(path_progress_max[valid], path_progress[valid])
                path_progress_ratio_last[valid] = path_progress_ratio[valid]
                contact_steps[valid] += contact_flags[valid].long()
                contact_count_sum[valid] += contact_count[valid]
                two_pusher_gate_sum[valid] += two_pusher_gate[valid]
                two_pusher_gate_active_steps[valid] += (two_pusher_gate[valid] > 0.6).float()
                push_utility_sum[valid] += push_utility[valid]
                action_norm_sum[valid] += action_norm[valid]

                right_turn_steps[valid] += right_turn_gate[valid].float()
                left_turn_steps[valid] += left_turn_gate[valid].float()
                right_turn_agv2_push_sum[valid] += right_turn_gate[valid].float() * push_utility[valid, 1]
                right_turn_agv3_push_sum[valid] += right_turn_gate[valid].float() * push_utility[valid, 2]
                left_turn_agv2_push_sum[valid] += left_turn_gate[valid].float() * push_utility[valid, 1]
                left_turn_agv3_push_sum[valid] += left_turn_gate[valid].float() * push_utility[valid, 2]

                after_wp2 = active_goal_idx >= 2
                after_wp2_steps[valid] += (after_wp2 & valid).float()[valid]
                agv1_contact_after_wp2_steps[valid] += ((after_wp2 & contact_flags[:, 0]) & valid).float()[valid]

                payload_wall_clearance_min[valid] = torch.minimum(payload_wall_clearance_min[valid], payload_clearance[valid])
                agv_wall_clearance_min[valid] = torch.minimum(agv_wall_clearance_min[valid], agv_clearance[valid])
                agv_payload_dist_max[valid] = torch.maximum(agv_payload_dist_max[valid], agv_payload_dists[valid])
                payload_near_wall_steps[valid] += (payload_clearance[valid] < args_cli.wall_near_margin).float()
                agv_near_wall_steps[valid] += (agv_clearance[valid] < args_cli.wall_near_margin).float()

                actions_cpu = actions.detach().cpu()
                for env_id in range(num_envs):
                    ep_id = int(active_episode_id[env_id].item())
                    if ep_id < 0:
                        continue
                    traj_writer.writerow(
                        [
                            global_step,
                            ep_id,
                            env_id,
                            int(episode_steps[env_id].item()),
                            float(env_origin_xy[env_id, 0].item()),
                            float(env_origin_xy[env_id, 1].item()),
                            float(payload_xy[env_id, 0].item()),
                            float(payload_xy[env_id, 1].item()),
                            float(target_xy[env_id, 0].item()),
                            float(target_xy[env_id, 1].item()),
                            float(final_target_xy[env_id, 0].item()),
                            float(final_target_xy[env_id, 1].item()),
                            int(path_segment_idx[env_id].item()),
                            int(active_goal_idx[env_id].item()),
                            int(turn_state[env_id].item()),
                            float(path_lateral_error[env_id].item()),
                            float(path_progress[env_id].item()),
                            float(path_progress_ratio[env_id].item()),
                            float(final_goal_dist[env_id].item()),
                            float(payload_yaw[env_id].item()),
                            float(payload_clearance[env_id].item()),
                            float(agv_xy[env_id, 0, 0].item()),
                            float(agv_xy[env_id, 0, 1].item()),
                            float(agv_payload_dists[env_id, 0].item()),
                            float(agv_clearance[env_id, 0].item()),
                            bool(contact_flags[env_id, 0].item()),
                            float(push_utility[env_id, 0].item()),
                            float(actions_cpu[env_id, 0].item()),
                            float(actions_cpu[env_id, 1].item()),
                            float(agv_xy[env_id, 1, 0].item()),
                            float(agv_xy[env_id, 1, 1].item()),
                            float(agv_payload_dists[env_id, 1].item()),
                            float(agv_clearance[env_id, 1].item()),
                            bool(contact_flags[env_id, 1].item()),
                            float(push_utility[env_id, 1].item()),
                            float(actions_cpu[env_id, 2].item()),
                            float(actions_cpu[env_id, 3].item()),
                            float(agv_xy[env_id, 2, 0].item()),
                            float(agv_xy[env_id, 2, 1].item()),
                            float(agv_payload_dists[env_id, 2].item()),
                            float(agv_clearance[env_id, 2].item()),
                            bool(contact_flags[env_id, 2].item()),
                            float(push_utility[env_id, 2].item()),
                            float(actions_cpu[env_id, 4].item()),
                            float(actions_cpu[env_id, 5].item()),
                            float(two_pusher_gate[env_id].item()),
                            float(rewards_1d[env_id].item()),
                            bool(terminated_1d[env_id].item()),
                            bool(truncated_1d[env_id].item()),
                            bool(success_state[env_id].item()),
                        ]
                    )

                for env_id in range(num_envs):
                    if not bool(done_1d[env_id].item()):
                        continue
                    ep_id = int(active_episode_id[env_id].item())
                    if ep_id < 0:
                        continue

                    steps = int(episode_steps[env_id].item())
                    steps_t = torch.tensor(float(max(steps, 1)), device=device)
                    contact_ratios = contact_steps[env_id].float() / steps_t
                    mean_contact_count = contact_count_sum[env_id] / steps_t
                    path_lateral_mean = path_lateral_error_sum[env_id] / steps_t
                    two_pusher_gate_mean = two_pusher_gate_sum[env_id] / steps_t
                    two_pusher_active = two_pusher_gate_active_steps[env_id] / steps_t
                    push_utility_mean = push_utility_sum[env_id] / steps_t
                    action_norm_mean = action_norm_sum[env_id] / steps_t
                    agv1_after_wp2_ratio = safe_mean(agv1_contact_after_wp2_steps[env_id], after_wp2_steps[env_id])
                    payload_near_wall_ratio = payload_near_wall_steps[env_id] / steps_t
                    agv_near_wall_ratio = agv_near_wall_steps[env_id] / steps_t

                    if hasattr(base_env, "last_success"):
                        terminal_success = bool(base_env.last_success[env_id].item())
                        final_dist = float(base_env.last_payload_goal_dist[env_id].item())
                        final_yaw_abs = float(base_env.last_payload_yaw_abs[env_id].item())
                    else:
                        terminal_success = bool(success_state[env_id].item())
                        final_dist = float(final_goal_dist[env_id].item())
                        final_yaw_abs = float(payload_yaw_abs[env_id].item())

                    max_progress_ratio = float(
                        torch.clamp(path_progress_max[env_id] / path_total_length[env_id], 0.0, 1.0).item()
                    )
                    final_progress_ratio = float(path_progress_ratio_last[env_id].item())

                    row = [
                        ep_id,
                        env_id,
                        terminal_success,
                        steps,
                        steps * step_dt,
                        final_dist,
                        final_yaw_abs,
                        float(path_lateral_mean.item()),
                        float(path_lateral_error_max[env_id].item()),
                        final_progress_ratio,
                        max_progress_ratio,
                        float(two_pusher_gate_mean.item()),
                        float(two_pusher_active.item()),
                        float(contact_ratios[0].item()),
                        float(contact_ratios[1].item()),
                        float(contact_ratios[2].item()),
                        float(mean_contact_count.item()),
                        float(agv1_after_wp2_ratio.item()),
                        float(safe_mean(right_turn_agv2_push_sum[env_id], right_turn_steps[env_id]).item()),
                        float(safe_mean(right_turn_agv3_push_sum[env_id], right_turn_steps[env_id]).item()),
                        float(safe_mean(left_turn_agv2_push_sum[env_id], left_turn_steps[env_id]).item()),
                        float(safe_mean(left_turn_agv3_push_sum[env_id], left_turn_steps[env_id]).item()),
                        float(push_utility_mean[0].item()),
                        float(push_utility_mean[1].item()),
                        float(push_utility_mean[2].item()),
                        float(action_norm_mean[0].item()),
                        float(action_norm_mean[1].item()),
                        float(action_norm_mean[2].item()),
                        float(payload_wall_clearance_min[env_id].item()),
                        float(agv_wall_clearance_min[env_id, 0].item()),
                        float(agv_wall_clearance_min[env_id, 1].item()),
                        float(agv_wall_clearance_min[env_id, 2].item()),
                        float(agv_payload_dist_max[env_id, 0].item()),
                        float(agv_payload_dist_max[env_id, 1].item()),
                        float(agv_payload_dist_max[env_id, 2].item()),
                        int(torch.argmax(agv_payload_dist_max[env_id]).item()) if bool(get_bool_buffer(base_env, "last_agv_escaped")[env_id].item()) else -1,
                        float(payload_near_wall_ratio.item()),
                        float(agv_near_wall_ratio[0].item()),
                        float(agv_near_wall_ratio[1].item()),
                        float(agv_near_wall_ratio[2].item()),
                        bool(get_bool_buffer(base_env, "last_position_success")[env_id].item()),
                        bool(get_bool_buffer(base_env, "last_yaw_success")[env_id].item()),
                        bool(get_bool_buffer(base_env, "last_out_of_bounds")[env_id].item()),
                        bool(get_bool_buffer(base_env, "last_bad_rear_push")[env_id].item()),
                        bool(get_bool_buffer(base_env, "last_agv_escaped")[env_id].item()),
                        float(episode_reward[env_id].item()),
                        bool(terminated_1d[env_id].item()),
                        bool(truncated_1d[env_id].item()),
                    ]
                    summary_writer.writerow(row)
                    completed_episodes += 1

                    print(
                        f"[episode {ep_id:03d}] env={env_id}, success={terminal_success}, steps={steps}, "
                        f"final_dist={final_dist:.3f}, final_yaw_abs={final_yaw_abs:.3f}, "
                        f"path_lat_mean={float(path_lateral_mean.item()):.3f}, "
                        f"path_lat_max={float(path_lateral_error_max[env_id].item()):.3f}, "
                        f"progress={final_progress_ratio:.3f}, two_gate={float(two_pusher_gate_mean.item()):.3f}, "
                        f"contact=({float(contact_ratios[0].item()):.2f},{float(contact_ratios[1].item()):.2f},{float(contact_ratios[2].item()):.2f}), "
                        f"wall_clearance=(payload={float(payload_wall_clearance_min[env_id].item()):.2f}, "
                        f"agv=({float(agv_wall_clearance_min[env_id,0].item()):.2f},"
                        f"{float(agv_wall_clearance_min[env_id,1].item()):.2f},"
                        f"{float(agv_wall_clearance_min[env_id,2].item()):.2f})), "
                        f"agv_dist_max=({float(agv_payload_dist_max[env_id,0].item()):.2f},"
                        f"{float(agv_payload_dist_max[env_id,1].item()):.2f},"
                        f"{float(agv_payload_dist_max[env_id,2].item()):.2f}), "
                        f"escaped_agv={row[35]}, "
                        f"done_flags=(pos={row[40]}, yaw={row[41]}, oob={row[42]}, rear={row[43]}, escape={row[44]}), "
                        f"reward={float(episode_reward[env_id].item()):.2f}"
                    )

                    # Reset per-env accumulators.
                    if next_episode_id < args_cli.num_episodes:
                        active_episode_id[env_id] = next_episode_id
                        next_episode_id += 1
                    else:
                        active_episode_id[env_id] = -1
                    episode_reward[env_id] = 0.0
                    episode_steps[env_id] = 0
                    path_lateral_error_sum[env_id] = 0.0
                    path_lateral_error_max[env_id] = 0.0
                    path_progress_max[env_id] = 0.0
                    path_progress_ratio_last[env_id] = 0.0
                    contact_steps[env_id, :] = 0
                    contact_count_sum[env_id] = 0.0
                    two_pusher_gate_sum[env_id] = 0.0
                    two_pusher_gate_active_steps[env_id] = 0.0
                    push_utility_sum[env_id, :] = 0.0
                    action_norm_sum[env_id, :] = 0.0
                    right_turn_steps[env_id] = 0.0
                    left_turn_steps[env_id] = 0.0
                    right_turn_agv2_push_sum[env_id] = 0.0
                    right_turn_agv3_push_sum[env_id] = 0.0
                    left_turn_agv2_push_sum[env_id] = 0.0
                    left_turn_agv3_push_sum[env_id] = 0.0
                    after_wp2_steps[env_id] = 0.0
                    agv1_contact_after_wp2_steps[env_id] = 0.0
                    payload_wall_clearance_min[env_id] = float("inf")
                    agv_wall_clearance_min[env_id, :] = float("inf")
                    agv_payload_dist_max[env_id, :] = 0.0
                    payload_near_wall_steps[env_id] = 0.0
                    agv_near_wall_steps[env_id, :] = 0.0
                    prev_eval_actions[env_id, :] = 0.0

                prev_eval_actions[~done_1d] = actions[~done_1d].detach()
            global_step += 1

    env.close()
    print("\n[INFO] Three-AGV PPO evaluation finished.")
    print(f"[INFO] Completed episodes: {completed_episodes}/{args_cli.num_episodes}")
    print(f"[INFO] Summary CSV: {summary_csv}")
    print(f"[INFO] Trajectory CSV: {trajectory_csv}")
    if args_cli.save_reference_path:
        print(f"[INFO] Reference path CSV: {output_dir / 'reference_path_local.csv'}")
        if (output_dir / "boundary_left_local.csv").exists():
            print(f"[INFO] Boundary CSVs: {output_dir / 'boundary_left_local.csv'}, {output_dir / 'boundary_right_local.csv'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
