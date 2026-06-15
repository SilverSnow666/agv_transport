"""Evaluate centralized PPO policy for three-AGV cooperative pushing.

This version is tailored for the current V4.3 two-pusher curriculum:
- Uses the environment's geometric front-contact flags when available.
- Records payload path-tracking metrics for trajectory plotting.
- Records two-pusher credit metrics to distinguish single-pusher and two-pusher behavior.
- Records per-AGV action and action-rate statistics to diagnose idle AGV jitter.

Outputs:
1. episode_summary.csv
2. trajectory.csv
3. reference_path_local.csv

Expected environment:
    - action_space = 6
    - observation_space = 44
    - 3 AGVs + 1 payload + path waypoints

Action layout:
    [v1, w1, v2, w2, v3, w3]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate three-AGV centralized PPO policy with trajectory and two-pusher metrics."
)

parser.add_argument(
    "--task",
    type=str,
    default="Template-Agv-Transport-Direct-v0",
    help="Task name.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to PPO checkpoint .pt file.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=20,
    help="Number of episodes to evaluate.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=5000,
    help="Maximum total simulation steps.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/eval_three_agv_ppo",
    help="Directory for CSV outputs.",
)
parser.add_argument(
    "--eval_contact_threshold",
    type=float,
    default=1.20,
    help=(
        "Fallback center-distance threshold for contact statistics. "
        "The env geometric contact function is used when available."
    ),
)
parser.add_argument(
    "--save_reference_path",
    action="store_true",
    default=True,
    help="Save local planned path points to reference_path_local.csv.",
)
parser.add_argument(
    "--no_save_reference_path",
    action="store_false",
    dest="save_reference_path",
    help="Disable writing reference_path_local.csv.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Remove custom argparse arguments before Hydra parses sys.argv.
# Otherwise Hydra will report "unrecognized arguments" for --checkpoint, --num_envs, etc.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import agv_transport.tasks  # noqa: F401, E402


def quat_to_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion in wxyz format to yaw angle."""
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return torch.atan2(siny_cosp, cosy_cosp)


def get_payload_quat(base_env) -> torch.Tensor:
    """Get payload root quaternion in wxyz format."""
    payload_data = base_env.payload.data
    if hasattr(payload_data, "root_quat_w"):
        return payload_data.root_quat_w
    return payload_data.root_state_w[:, 3:7]


def get_env_actions(base_env, runner_actions: torch.Tensor) -> torch.Tensor:
    """Return the actual normalized actions stored by the env when available."""
    if hasattr(base_env, "actions"):
        return base_env.actions.detach()
    return runner_actions.detach()


def compute_contact_flags(
    base_env,
    payload_xy: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Compute contact flags for the three AGVs.

    Prefer the environment's own _compute_contact_flags(), because the current
    task uses geometric front-edge contact rather than center-distance contact.
    Fallback to center-distance contact for older environments.
    """
    if hasattr(base_env, "_compute_contact_flags"):
        return base_env._compute_contact_flags().bool()

    contact_flags = []
    for agv in base_env.agvs:
        agv_xy = agv.data.root_pos_w[:, :2]
        agv_payload_dist = torch.linalg.norm(agv_xy - payload_xy, dim=1)
        contact_flags.append(agv_payload_dist < threshold)
    return torch.stack(contact_flags, dim=1)


def compute_formation_errors(
    base_env,
    payload_xy: torch.Tensor,
    target_xy: torch.Tensor,
) -> torch.Tensor:
    """Compute formation error for three AGVs.

    Prefer the env implementation when available, so eval matches training.
    """
    if hasattr(base_env, "_compute_formation_errors"):
        return base_env._compute_formation_errors()

    payload_to_target = target_xy - payload_xy
    dist_to_target = torch.linalg.norm(payload_to_target, dim=1, keepdim=True).clamp_min(1e-6)
    push_dir = payload_to_target / dist_to_target
    lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=1)

    device = payload_xy.device
    stand_off_distances = torch.tensor(
        getattr(base_env.cfg, "formation_stand_off_distances", [0.90, 0.90, 0.90]),
        device=device,
    )
    lateral_offsets = torch.tensor(
        getattr(base_env.cfg, "formation_lateral_offsets", [0.0, 0.65, -0.65]),
        device=device,
    )

    errors = []
    for i, agv in enumerate(base_env.agvs):
        agv_xy = agv.data.root_pos_w[:, :2]
        desired_xy = payload_xy - push_dir * stand_off_distances[i] + lateral_dir * lateral_offsets[i]
        errors.append(torch.linalg.norm(agv_xy - desired_xy, dim=1))
    return torch.stack(errors, dim=1)


def get_path_metrics(base_env):
    """Return lookahead target, lateral error, progress and segment index."""
    if hasattr(base_env, "_compute_path_tracking_quantities"):
        return base_env._compute_path_tracking_quantities()

    # Fallback for very old environments: no path metrics available.
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
    """Compute total path length for each environment."""
    if hasattr(base_env, "_get_path_points_xy"):
        path_points = base_env._get_path_points_xy()
        segment_vec = path_points[:, 1:, :] - path_points[:, :-1, :]
        return torch.linalg.norm(segment_vec, dim=2).sum(dim=1).clamp_min(1e-6)

    # Fallback: use straight-line distance from payload init to target.
    final_target_xy = get_final_target_xy(base_env)
    payload_init = torch.tensor(base_env.cfg.payload_init_pos[:2], device=base_env.device).view(1, 2)
    start_xy = base_env.scene.env_origins[:, :2] + payload_init
    return torch.linalg.norm(final_target_xy - start_xy, dim=1).clamp_min(1e-6)


def compute_push_metrics(
    base_env,
    payload_xy: torch.Tensor,
    contact_flags: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute push utility and two-pusher gate.

    Returns:
        push_utility: [num_envs, 3]
        second_push_utility: [num_envs]
        top2_push_utility: [num_envs]
        two_pusher_gate: [num_envs]
        front_facing_score: [num_envs, 3]
    """
    contact_flags_float = contact_flags.float()

    if hasattr(base_env, "_compute_contact_geometry"):
        heading_to_payload, _, _, v_actions = base_env._compute_contact_geometry(payload_xy)
    else:
        heading_to_payload_list = []
        v_action_list = []
        for i, agv in enumerate(base_env.agvs):
            agv_xy = agv.data.root_pos_w[:, :2]
            agv_heading_xy = torch.stack(
                (torch.cos(base_env.agv_yaw[:, i]), torch.sin(base_env.agv_yaw[:, i])),
                dim=1,
            )
            to_payload = payload_xy - agv_xy
            dir_to_payload = to_payload / torch.linalg.norm(to_payload, dim=1, keepdim=True).clamp_min(1e-6)
            heading_to_payload_list.append(torch.sum(agv_heading_xy * dir_to_payload, dim=1))
            v_action_list.append(base_env.actions[:, 2 * i])
        heading_to_payload = torch.stack(heading_to_payload_list, dim=1)
        v_actions = torch.stack(v_action_list, dim=1)

    front_heading_min = getattr(base_env.cfg, "front_contact_heading_min", 0.20)
    front_facing_score = torch.clamp(
        (heading_to_payload - front_heading_min) / (1.0 - front_heading_min),
        min=0.0,
        max=1.0,
    )

    push_utility = contact_flags_float * front_facing_score * torch.clamp(v_actions, min=0.0)
    sorted_push_utility, _ = torch.sort(push_utility, dim=1, descending=True)
    second_push_utility = sorted_push_utility[:, 1]
    top2_push_utility = sorted_push_utility[:, 0] + sorted_push_utility[:, 1]
    threshold = getattr(base_env.cfg, "two_pusher_gate_threshold", 0.20)
    two_pusher_gate = torch.clamp(second_push_utility / threshold, min=0.0, max=1.0)

    return push_utility, second_push_utility, top2_push_utility, two_pusher_gate, front_facing_score


def compute_action_metrics(actions: torch.Tensor, prev_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-AGV action norm and action-rate norm."""
    action_norm = torch.stack(
        (
            torch.linalg.norm(actions[:, 0:2], dim=1),
            torch.linalg.norm(actions[:, 2:4], dim=1),
            torch.linalg.norm(actions[:, 4:6], dim=1),
        ),
        dim=1,
    )
    action_rate_norm = torch.stack(
        (
            torch.linalg.norm(actions[:, 0:2] - prev_actions[:, 0:2], dim=1),
            torch.linalg.norm(actions[:, 2:4] - prev_actions[:, 2:4], dim=1),
            torch.linalg.norm(actions[:, 4:6] - prev_actions[:, 4:6], dim=1),
        ),
        dim=1,
    )
    return action_norm, action_rate_norm


def get_runner_action(runner: Runner, obs: torch.Tensor) -> torch.Tensor:
    """Get deterministic/evaluation action from skrl runner agent."""
    with torch.inference_mode():
        return runner.agent.act(obs, timestep=0, timesteps=0)[0]


def write_reference_path(output_dir: Path, base_env) -> None:
    """Write local reference path points so plotting scripts can use the exact eval path."""
    reference_csv = output_dir / "reference_path_local.csv"
    start = tuple(getattr(base_env.cfg, "payload_init_pos", (0.0, 0.0, 0.0))[:2])
    waypoints = list(getattr(base_env.cfg, "waypoints", ()))
    path_points = [start] + [(float(x), float(y)) for x, y in waypoints]
    with open(reference_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point_idx", "x", "y"])
        for idx, (x, y) in enumerate(path_points):
            writer.writerow([idx, float(x), float(y)])


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Evaluate policy without logging post-reset states.

    Important Isaac Lab note:
        DirectRLEnv / vector wrappers may reset an environment inside env.step()
        when terminated/truncated is returned. Therefore, reading base_env state
        *after* env.step() can give the next episode's reset state. This script
        logs the physical state *before* stepping, then attaches the reward/done
        flags returned by the step. Episode terminal distance/yaw still use the
        env's last_* buffers when available.
    """
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

    if args_cli.save_reference_path:
        write_reference_path(output_dir, base_env)

    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, agent_cfg)

    print(f"[INFO] Loading PPO checkpoint from: {checkpoint_path}")
    runner.agent.load(str(checkpoint_path))
    if hasattr(runner, "set_running_mode"):
        runner.set_running_mode("eval")
    elif hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        print("[WARN] set_running_mode is not available in this skrl version. Continue evaluation with inference_mode.")

    obs, _ = env.reset()

    num_envs = args_cli.num_envs
    device = base_env.device
    action_dim = int(getattr(base_env.cfg, "action_space", 6))

    target_radius = base_env.cfg.target_radius
    target_yaw_radius = base_env.cfg.target_yaw_radius
    path_total_length = compute_total_path_length(base_env)

    active_episode_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    next_episode_id = 0
    for env_id in range(num_envs):
        if next_episode_id < args_cli.num_episodes:
            active_episode_id[env_id] = next_episode_id
            next_episode_id += 1

    episode_reward = torch.zeros(num_envs, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    min_final_goal_dist = torch.full((num_envs,), float("inf"), device=device)
    payload_yaw_abs_max = torch.zeros(num_envs, device=device)
    min_yaw_abs_inside_target = torch.full((num_envs,), float("inf"), device=device)

    contact_steps = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
    contact_count_sum = torch.zeros(num_envs, device=device)

    formation_error_sum = torch.zeros(num_envs, device=device)
    formation_error_max = torch.zeros(num_envs, device=device)

    path_lateral_error_sum = torch.zeros(num_envs, device=device)
    path_lateral_error_max = torch.zeros(num_envs, device=device)
    path_progress_max = torch.zeros(num_envs, device=device)
    path_progress_last = torch.zeros(num_envs, device=device)
    path_progress_ratio_last = torch.zeros(num_envs, device=device)

    two_pusher_gate_sum = torch.zeros(num_envs, device=device)
    two_pusher_gate_active_steps = torch.zeros(num_envs, device=device)
    second_push_sum = torch.zeros(num_envs, device=device)
    top2_push_sum = torch.zeros(num_envs, device=device)
    push_utility_sum = torch.zeros((num_envs, 3), device=device)

    action_norm_sum = torch.zeros((num_envs, 3), device=device)
    action_rate_sum = torch.zeros((num_envs, 3), device=device)
    low_utility_action_sum = torch.zeros((num_envs, 3), device=device)

    prev_eval_actions = torch.zeros((num_envs, action_dim), device=device)

    completed_episodes = 0
    global_step = 0

    def compute_push_metrics_with_actions(payload_xy, contact_flags, actions_for_step):
        contact_flags_float = contact_flags.float()
        if hasattr(base_env, "_compute_contact_geometry"):
            heading_to_payload, _, _, _ = base_env._compute_contact_geometry(payload_xy)
        else:
            heading_to_payload_list = []
            for i, agv in enumerate(base_env.agvs):
                agv_xy = agv.data.root_pos_w[:, :2]
                agv_heading_xy = torch.stack(
                    (torch.cos(base_env.agv_yaw[:, i]), torch.sin(base_env.agv_yaw[:, i])), dim=1
                )
                to_payload = payload_xy - agv_xy
                dir_to_payload = to_payload / torch.linalg.norm(to_payload, dim=1, keepdim=True).clamp_min(1e-6)
                heading_to_payload_list.append(torch.sum(agv_heading_xy * dir_to_payload, dim=1))
            heading_to_payload = torch.stack(heading_to_payload_list, dim=1)

        front_heading_min = getattr(base_env.cfg, "front_contact_heading_min", 0.20)
        front_facing_score = torch.clamp(
            (heading_to_payload - front_heading_min) / (1.0 - front_heading_min), min=0.0, max=1.0
        )
        v_actions = torch.stack(
            (actions_for_step[:, 0], actions_for_step[:, 2], actions_for_step[:, 4]), dim=1
        )
        push_utility = contact_flags_float * front_facing_score * torch.clamp(v_actions, min=0.0)
        sorted_push_utility, _ = torch.sort(push_utility, dim=1, descending=True)
        second_push_utility = sorted_push_utility[:, 1]
        top2_push_utility = sorted_push_utility[:, 0] + sorted_push_utility[:, 1]
        threshold = getattr(base_env.cfg, "two_pusher_gate_threshold", 0.20)
        two_pusher_gate = torch.clamp(second_push_utility / threshold, min=0.0, max=1.0)
        return push_utility, second_push_utility, top2_push_utility, two_pusher_gate

    with open(summary_csv, "w", newline="", encoding="utf-8") as summary_file, open(
        trajectory_csv, "w", newline="", encoding="utf-8"
    ) as traj_file:
        summary_writer = csv.writer(summary_file)
        traj_writer = csv.writer(traj_file)

        summary_writer.writerow([
            "episode", "env_id", "success", "episode_steps", "episode_time_s",
            "final_goal_dist", "min_final_goal_dist", "payload_yaw_abs_max",
            "final_payload_yaw_abs", "target_yaw_radius",
            "agv1_contact_ratio", "agv2_contact_ratio", "agv3_contact_ratio", "mean_contact_count",
            "formation_error_mean", "formation_error_max",
            "path_lateral_error_mean", "path_lateral_error_max",
            "final_path_progress", "final_path_progress_ratio", "max_path_progress_ratio",
            "two_pusher_gate_mean", "two_pusher_gate_active_ratio", "second_push_mean", "top2_push_mean",
            "agv1_push_utility_mean", "agv2_push_utility_mean", "agv3_push_utility_mean",
            "agv1_action_norm_mean", "agv2_action_norm_mean", "agv3_action_norm_mean",
            "agv1_action_rate_mean", "agv2_action_rate_mean", "agv3_action_rate_mean",
            "agv1_low_utility_action_mean", "agv2_low_utility_action_mean", "agv3_low_utility_action_mean",
            "episode_reward", "terminated", "truncated",
        ])

        traj_writer.writerow([
            "global_step", "episode", "env_id", "step_in_episode",
            "env_origin_x", "env_origin_y",
            "payload_x", "payload_y", "target_x", "target_y", "final_target_x", "final_target_y",
            "path_segment_idx", "path_lateral_error", "path_progress", "path_progress_ratio",
            "lookahead_target_dist", "final_goal_dist", "payload_yaw",
            "agv1_x", "agv1_y", "agv1_contact", "agv1_push_utility", "agv1_action_norm", "agv1_action_rate",
            "agv2_x", "agv2_y", "agv2_contact", "agv2_push_utility", "agv2_action_norm", "agv2_action_rate",
            "agv3_x", "agv3_y", "agv3_contact", "agv3_push_utility", "agv3_action_norm", "agv3_action_rate",
            "two_pusher_gate", "second_push_utility", "top2_push_utility",
            "formation_error_1", "formation_error_2", "formation_error_3",
            "action_v1", "action_w1", "action_v2", "action_w2", "action_v3", "action_w3",
            "reward", "terminated", "truncated", "success",
        ])

        while simulation_app.is_running() and global_step < args_cli.max_steps:
            if completed_episodes >= args_cli.num_episodes:
                break

            with torch.inference_mode():
                actions = get_runner_action(runner, obs).to(device)

                # Capture current physical state BEFORE env.step().
                # This avoids logging the auto-reset state of a completed env.
                payload_xy = base_env.payload.data.root_pos_w[:, :2].clone()
                agv1_xy_pre = base_env.agv1.data.root_pos_w[:, :2].clone()
                agv2_xy_pre = base_env.agv2.data.root_pos_w[:, :2].clone()
                agv3_xy_pre = base_env.agv3.data.root_pos_w[:, :2].clone()
                env_origin_xy = base_env.scene.env_origins[:, :2].clone()
                target_xy, path_lateral_error, path_progress, path_segment_idx = get_path_metrics(base_env)
                target_xy = target_xy.clone()
                path_lateral_error = path_lateral_error.clone()
                path_progress = path_progress.clone()
                path_segment_idx = path_segment_idx.clone()
                final_target_xy = get_final_target_xy(base_env).clone()
                path_progress_ratio = torch.clamp(path_progress / path_total_length, min=0.0, max=1.0)

                lookahead_target_dist = torch.linalg.norm(payload_xy - target_xy, dim=1)
                final_goal_dist = torch.linalg.norm(payload_xy - final_target_xy, dim=1)

                payload_quat = get_payload_quat(base_env).clone()
                payload_yaw = quat_to_yaw_wxyz(payload_quat)
                payload_yaw_abs = torch.abs(payload_yaw)

                contact_flags = compute_contact_flags(base_env, payload_xy, args_cli.eval_contact_threshold)
                contact_count = contact_flags.float().sum(dim=1)

                push_utility, second_push_utility, top2_push_utility, two_pusher_gate = compute_push_metrics_with_actions(
                    payload_xy, contact_flags, actions
                )
                action_norm, action_rate_norm = compute_action_metrics(actions, prev_eval_actions)

                low_utility_threshold = getattr(base_env.cfg, "idle_low_utility_threshold", 0.08)
                low_utility_weight = torch.clamp(
                    (low_utility_threshold - push_utility) / max(low_utility_threshold, 1e-6), min=0.0, max=1.0
                )
                low_utility_action = low_utility_weight * action_norm

                formation_errors = compute_formation_errors(base_env, payload_xy, target_xy)
                formation_error_mean_step = formation_errors.mean(dim=1)
                formation_error_max_step = formation_errors.max(dim=1).values

                position_success = final_goal_dist < target_radius
                yaw_success = payload_yaw_abs < target_yaw_radius
                success_state = position_success & yaw_success

                # Step the env AFTER capturing state.
                obs, rewards, terminated, truncated, info = env.step(actions)
                rewards_1d = rewards.reshape(-1).to(device)
                terminated_1d = terminated.reshape(-1).bool().to(device)
                truncated_1d = truncated.reshape(-1).bool().to(device)
                done_1d = terminated_1d | truncated_1d

                valid_env = active_episode_id >= 0
                episode_reward[valid_env] += rewards_1d[valid_env]
                episode_steps[valid_env] += 1

                min_final_goal_dist[valid_env] = torch.minimum(min_final_goal_dist[valid_env], final_goal_dist[valid_env])
                payload_yaw_abs_max[valid_env] = torch.maximum(payload_yaw_abs_max[valid_env], payload_yaw_abs[valid_env])
                min_yaw_abs_inside_target[position_success & valid_env] = torch.minimum(
                    min_yaw_abs_inside_target[position_success & valid_env], payload_yaw_abs[position_success & valid_env]
                )

                contact_steps[valid_env] += contact_flags[valid_env].long()
                contact_count_sum[valid_env] += contact_count[valid_env]
                formation_error_sum[valid_env] += formation_error_mean_step[valid_env]
                formation_error_max[valid_env] = torch.maximum(formation_error_max[valid_env], formation_error_max_step[valid_env])
                path_lateral_error_sum[valid_env] += path_lateral_error[valid_env]
                path_lateral_error_max[valid_env] = torch.maximum(path_lateral_error_max[valid_env], path_lateral_error[valid_env])
                path_progress_max[valid_env] = torch.maximum(path_progress_max[valid_env], path_progress[valid_env])
                path_progress_last[valid_env] = path_progress[valid_env]
                path_progress_ratio_last[valid_env] = path_progress_ratio[valid_env]
                two_pusher_gate_sum[valid_env] += two_pusher_gate[valid_env]
                two_pusher_gate_active_steps[valid_env] += (two_pusher_gate[valid_env] > 0.6).float()
                second_push_sum[valid_env] += second_push_utility[valid_env]
                top2_push_sum[valid_env] += top2_push_utility[valid_env]
                push_utility_sum[valid_env] += push_utility[valid_env]
                action_norm_sum[valid_env] += action_norm[valid_env]
                action_rate_sum[valid_env] += action_rate_norm[valid_env]
                low_utility_action_sum[valid_env] += low_utility_action[valid_env]

                actions_cpu = actions.detach().cpu()

                for env_id in range(num_envs):
                    ep_id = int(active_episode_id[env_id].item())
                    if ep_id < 0:
                        continue
                    traj_writer.writerow([
                        global_step, ep_id, env_id, int(episode_steps[env_id].item()),
                        float(env_origin_xy[env_id, 0].item()), float(env_origin_xy[env_id, 1].item()),
                        float(payload_xy[env_id, 0].item()), float(payload_xy[env_id, 1].item()),
                        float(target_xy[env_id, 0].item()), float(target_xy[env_id, 1].item()),
                        float(final_target_xy[env_id, 0].item()), float(final_target_xy[env_id, 1].item()),
                        int(path_segment_idx[env_id].item()),
                        float(path_lateral_error[env_id].item()), float(path_progress[env_id].item()), float(path_progress_ratio[env_id].item()),
                        float(lookahead_target_dist[env_id].item()), float(final_goal_dist[env_id].item()), float(payload_yaw[env_id].item()),
                        float(agv1_xy_pre[env_id, 0].item()), float(agv1_xy_pre[env_id, 1].item()),
                        bool(contact_flags[env_id, 0].item()), float(push_utility[env_id, 0].item()), float(action_norm[env_id, 0].item()), float(action_rate_norm[env_id, 0].item()),
                        float(agv2_xy_pre[env_id, 0].item()), float(agv2_xy_pre[env_id, 1].item()),
                        bool(contact_flags[env_id, 1].item()), float(push_utility[env_id, 1].item()), float(action_norm[env_id, 1].item()), float(action_rate_norm[env_id, 1].item()),
                        float(agv3_xy_pre[env_id, 0].item()), float(agv3_xy_pre[env_id, 1].item()),
                        bool(contact_flags[env_id, 2].item()), float(push_utility[env_id, 2].item()), float(action_norm[env_id, 2].item()), float(action_rate_norm[env_id, 2].item()),
                        float(two_pusher_gate[env_id].item()), float(second_push_utility[env_id].item()), float(top2_push_utility[env_id].item()),
                        float(formation_errors[env_id, 0].item()), float(formation_errors[env_id, 1].item()), float(formation_errors[env_id, 2].item()),
                        float(actions_cpu[env_id, 0].item()), float(actions_cpu[env_id, 1].item()),
                        float(actions_cpu[env_id, 2].item()), float(actions_cpu[env_id, 3].item()),
                        float(actions_cpu[env_id, 4].item()), float(actions_cpu[env_id, 5].item()),
                        float(rewards_1d[env_id].item()), bool(terminated_1d[env_id].item()), bool(truncated_1d[env_id].item()), bool(success_state[env_id].item()),
                    ])

                for env_id in range(num_envs):
                    if not bool(done_1d[env_id].item()):
                        continue
                    ep_id = int(active_episode_id[env_id].item())
                    if ep_id < 0:
                        continue

                    steps = int(episode_steps[env_id].item())
                    steps_safe = max(steps, 1)
                    contact_ratios = contact_steps[env_id].float() / steps_safe
                    mean_contact_count = contact_count_sum[env_id] / steps_safe
                    formation_error_mean = formation_error_sum[env_id] / steps_safe
                    path_lateral_mean = path_lateral_error_sum[env_id] / steps_safe
                    two_pusher_gate_mean = two_pusher_gate_sum[env_id] / steps_safe
                    two_pusher_active_ratio = two_pusher_gate_active_steps[env_id] / steps_safe
                    second_push_mean = second_push_sum[env_id] / steps_safe
                    top2_push_mean = top2_push_sum[env_id] / steps_safe
                    push_utility_mean = push_utility_sum[env_id] / steps_safe
                    action_norm_mean = action_norm_sum[env_id] / steps_safe
                    action_rate_mean = action_rate_sum[env_id] / steps_safe
                    low_utility_action_mean = low_utility_action_sum[env_id] / steps_safe

                    if hasattr(base_env, "last_success"):
                        terminal_success = bool(base_env.last_success[env_id].item())
                        final_dist_for_summary = float(base_env.last_payload_goal_dist[env_id].item())
                        final_yaw_abs_for_summary = float(base_env.last_payload_yaw_abs[env_id].item())
                    else:
                        terminal_success = (
                            float(min_final_goal_dist[env_id].item()) < target_radius
                            and float(min_yaw_abs_inside_target[env_id].item()) < target_yaw_radius
                        )
                        final_dist_for_summary = float(min_final_goal_dist[env_id].item())
                        final_yaw_abs_for_summary = float(payload_yaw_abs_max[env_id].item())

                    final_progress_ratio = float(path_progress_ratio_last[env_id].item())
                    max_progress_ratio = float(torch.clamp(path_progress_max[env_id] / path_total_length[env_id], min=0.0, max=1.0).item())

                    summary_writer.writerow([
                        ep_id, env_id, terminal_success, steps, steps * base_env.step_dt,
                        final_dist_for_summary, float(min_final_goal_dist[env_id].item()),
                        float(payload_yaw_abs_max[env_id].item()), float(final_yaw_abs_for_summary), float(target_yaw_radius),
                        float(contact_ratios[0].item()), float(contact_ratios[1].item()), float(contact_ratios[2].item()), float(mean_contact_count.item()),
                        float(formation_error_mean.item()), float(formation_error_max[env_id].item()),
                        float(path_lateral_mean.item()), float(path_lateral_error_max[env_id].item()),
                        float(path_progress_last[env_id].item()), final_progress_ratio, max_progress_ratio,
                        float(two_pusher_gate_mean.item()), float(two_pusher_active_ratio.item()), float(second_push_mean.item()), float(top2_push_mean.item()),
                        float(push_utility_mean[0].item()), float(push_utility_mean[1].item()), float(push_utility_mean[2].item()),
                        float(action_norm_mean[0].item()), float(action_norm_mean[1].item()), float(action_norm_mean[2].item()),
                        float(action_rate_mean[0].item()), float(action_rate_mean[1].item()), float(action_rate_mean[2].item()),
                        float(low_utility_action_mean[0].item()), float(low_utility_action_mean[1].item()), float(low_utility_action_mean[2].item()),
                        float(episode_reward[env_id].item()), bool(terminated_1d[env_id].item()), bool(truncated_1d[env_id].item()),
                    ])
                    completed_episodes += 1

                    print(
                        f"[episode {ep_id:03d}] env={env_id}, success={terminal_success}, steps={steps}, "
                        f"final_dist={final_dist_for_summary:.3f}, final_yaw_abs={final_yaw_abs_for_summary:.3f}, "
                        f"path_lat_mean={float(path_lateral_mean.item()):.3f}, path_lat_max={float(path_lateral_error_max[env_id].item()):.3f}, "
                        f"final_progress_ratio={final_progress_ratio:.3f}, max_progress_ratio={max_progress_ratio:.3f}, "
                        f"two_gate_mean={float(two_pusher_gate_mean.item()):.3f}, two_gate_active={float(two_pusher_active_ratio.item()):.3f}, "
                        f"contact=({float(contact_ratios[0].item()):.2f}, {float(contact_ratios[1].item()):.2f}, {float(contact_ratios[2].item()):.2f}), "
                        f"action_norm=({float(action_norm_mean[0].item()):.2f}, {float(action_norm_mean[1].item()):.2f}, {float(action_norm_mean[2].item()):.2f}), "
                        f"reward={float(episode_reward[env_id].item()):.2f}"
                    )

                    if next_episode_id < args_cli.num_episodes:
                        active_episode_id[env_id] = next_episode_id
                        next_episode_id += 1
                    else:
                        active_episode_id[env_id] = -1

                    episode_reward[env_id] = 0.0
                    episode_steps[env_id] = 0
                    min_final_goal_dist[env_id] = float("inf")
                    payload_yaw_abs_max[env_id] = 0.0
                    min_yaw_abs_inside_target[env_id] = float("inf")
                    contact_steps[env_id, :] = 0
                    contact_count_sum[env_id] = 0.0
                    formation_error_sum[env_id] = 0.0
                    formation_error_max[env_id] = 0.0
                    path_lateral_error_sum[env_id] = 0.0
                    path_lateral_error_max[env_id] = 0.0
                    path_progress_max[env_id] = 0.0
                    path_progress_last[env_id] = 0.0
                    path_progress_ratio_last[env_id] = 0.0
                    two_pusher_gate_sum[env_id] = 0.0
                    two_pusher_gate_active_steps[env_id] = 0.0
                    second_push_sum[env_id] = 0.0
                    top2_push_sum[env_id] = 0.0
                    push_utility_sum[env_id, :] = 0.0
                    action_norm_sum[env_id, :] = 0.0
                    action_rate_sum[env_id, :] = 0.0
                    low_utility_action_sum[env_id, :] = 0.0
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


if __name__ == "__main__":
    main()
    simulation_app.close()
