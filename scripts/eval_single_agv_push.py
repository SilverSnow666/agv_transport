"""Evaluate trained PPO policy for single-AGV pushing task.

Outputs:
    1. logs/eval_single_agv_push/episode_summary.csv
    2. logs/eval_single_agv_push/trajectory.csv
"""

import argparse
import csv
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate single-AGV push policy.")
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
    help="Path to trained checkpoint, for example trained_models/single_agv_push_stable_baseline.pt.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="Number of parallel environments for evaluation.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=20,
    help="Total number of evaluation episodes.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=3000,
    help="Safety limit for total simulation steps.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/eval_single_agv_push",
    help="Directory for CSV outputs.",
)
parser.add_argument(
    "--eval_contact_threshold",
    type=float,
    default=0.85,
    help="Distance threshold used only for evaluation contact ratio.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agv_transport.tasks  # noqa: F401

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

try:
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
except ImportError:
    from isaaclab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

from skrl.utils.runner.torch import Runner


def _to_1d_bool(x: torch.Tensor) -> torch.Tensor:
    """Convert terminated/truncated tensor to 1D bool tensor."""
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).bool()
    return torch.tensor(x).reshape(-1).bool()


def _to_1d_float(x: torch.Tensor) -> torch.Tensor:
    """Convert reward tensor to 1D float tensor."""
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).float()
    return torch.tensor(x).reshape(-1).float()


def main():
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "episode_summary.csv"
    trajectory_csv = output_dir / "trajectory.csv"

    # Environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Agent configuration from task registry
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # Create raw environment first.
    # We keep raw_env to read AGV/payload states directly.
    raw_env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap for skrl policy.
    env = SkrlVecEnvWrapper(raw_env)

    # Build runner and load checkpoint
    runner = Runner(env, agent_cfg)

    try:
        runner.load(str(checkpoint_path))
    except AttributeError:
        runner.agent.load(str(checkpoint_path))

    try:
        runner.set_running_mode("eval")
    except AttributeError:
        runner.agent.set_running_mode("eval")

    agent = runner.agent

    # Reset environment
    obs, _ = env.reset()

    base_env = raw_env.unwrapped
    num_envs = args_cli.num_envs
    target_radius = base_env.cfg.target_radius
    eval_success_radius = max(target_radius, 0.25)

    # Episode bookkeeping
    next_episode_id = 0
    active_episode_id = torch.full((num_envs,), -1, dtype=torch.long, device=base_env.device)

    for env_id in range(num_envs):
        if next_episode_id < args_cli.num_episodes:
            active_episode_id[env_id] = next_episode_id
            next_episode_id += 1

    episode_reward = torch.zeros(num_envs, device=base_env.device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
    contact_steps = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
    min_payload_target_dist = torch.full((num_envs,), float("inf"), device=base_env.device)

    completed_episodes = 0
    global_step = 0

    with open(summary_csv, mode="w", newline="", encoding="utf-8") as summary_file, open(
        trajectory_csv, mode="w", newline="", encoding="utf-8"
    ) as traj_file:
        summary_writer = csv.writer(summary_file)
        traj_writer = csv.writer(traj_file)

        summary_writer.writerow(
            [
                "episode",
                "env_id",
                "success",
                "episode_steps",
                "episode_time_s",
                "final_payload_target_dist",
                "min_payload_target_dist",
                "contact_steps",
                "contact_ratio",
                "episode_reward",
                "terminated",
                "truncated",
            ]
        )

        traj_writer.writerow(
            [
                "global_step",
                "episode",
                "env_id",
                "step_in_episode",
                "agv_x",
                "agv_y",
                "payload_x",
                "payload_y",
                "target_x",
                "target_y",
                "payload_target_dist",
                "agv_payload_dist",
                "contact_flag",
                "action_v",
                "action_w",
                "reward",
                "terminated",
                "truncated",
                "success",
            ]
        )

        while simulation_app.is_running() and global_step < args_cli.max_steps:
            if completed_episodes >= args_cli.num_episodes:
                break

            with torch.inference_mode():
                actions = agent.act(obs, timestep=global_step, timesteps=args_cli.max_steps)[0]
                obs, rewards, terminated, truncated, infos = env.step(actions)

                rewards_1d = _to_1d_float(rewards).to(base_env.device)
                terminated_1d = _to_1d_bool(terminated).to(base_env.device)
                truncated_1d = _to_1d_bool(truncated).to(base_env.device)
                done_1d = terminated_1d | truncated_1d

                agv_xy = base_env.agv.data.root_pos_w[:, :2]
                payload_xy = base_env.payload.data.root_pos_w[:, :2]
                target_xy = base_env._get_target_xy()

                payload_target_dist = torch.linalg.norm(payload_xy - target_xy, dim=1)

                # 评估用接触判断：只用于 contact_ratio 统计，不影响训练 reward
                agv_payload_dist = torch.linalg.norm(agv_xy - payload_xy, dim=1)
                contact_flag = agv_payload_dist < args_cli.eval_contact_threshold

                success = payload_target_dist < target_radius

                # Update episode statistics
                valid_env = active_episode_id >= 0

                episode_reward[valid_env] += rewards_1d[valid_env]
                episode_steps[valid_env] += 1
                contact_steps[valid_env] += contact_flag[valid_env].long()
                min_payload_target_dist[valid_env] = torch.minimum(
                    min_payload_target_dist[valid_env],
                    payload_target_dist[valid_env],
                )

                # Log trajectory
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
                            float(agv_xy[env_id, 0].item()),
                            float(agv_xy[env_id, 1].item()),
                            float(payload_xy[env_id, 0].item()),
                            float(payload_xy[env_id, 1].item()),
                            float(target_xy[env_id, 0].item()),
                            float(target_xy[env_id, 1].item()),
                            float(payload_target_dist[env_id].item()),
                            float(agv_payload_dist[env_id].item()),
                            bool(contact_flag[env_id].item()),
                            float(actions_cpu[env_id, 0].item()),
                            float(actions_cpu[env_id, 1].item()),
                            float(rewards_1d[env_id].item()),
                            bool(terminated_1d[env_id].item()),
                            bool(truncated_1d[env_id].item()),
                            bool(success[env_id].item()),
                        ]
                    )

                # Finalize done episodes
                for env_id in range(num_envs):
                    if not bool(done_1d[env_id].item()):
                        continue

                    ep_id = int(active_episode_id[env_id].item())
                    if ep_id < 0:
                        continue

                    steps = int(episode_steps[env_id].item())
                    contact = int(contact_steps[env_id].item())
                    contact_ratio = contact / max(steps, 1)

                    terminal_success = float(min_payload_target_dist[env_id].item()) < eval_success_radius

                    final_dist_for_summary = float(min_payload_target_dist[env_id].item())

                    summary_writer.writerow(
                        [
                            ep_id,
                            env_id,
                            terminal_success,
                            steps,
                            steps * base_env.step_dt,
                            final_dist_for_summary,
                            float(min_payload_target_dist[env_id].item()),
                            contact,
                            contact_ratio,
                            float(episode_reward[env_id].item()),
                            bool(terminated_1d[env_id].item()),
                            bool(truncated_1d[env_id].item()),
                        ]
                    )

                    completed_episodes += 1

                    print(
                        f"[episode {ep_id:03d}] "
                        f"env={env_id}, "
                        f"success={terminal_success}, "
                        f"steps={steps}, "
                        f"final_dist={final_dist_for_summary:.3f}, "
                        f"contact_ratio={contact_ratio:.2f}, "
                        f"reward={float(episode_reward[env_id].item()):.2f}"
                    )

                    # Assign a new episode id to this env if needed.
                    if next_episode_id < args_cli.num_episodes:
                        active_episode_id[env_id] = next_episode_id
                        next_episode_id += 1
                    else:
                        active_episode_id[env_id] = -1

                    # Reset statistics for this env.
                    episode_reward[env_id] = 0.0
                    episode_steps[env_id] = 0
                    contact_steps[env_id] = 0
                    min_payload_target_dist[env_id] = float("inf")

            global_step += 1

    env.close()

    print("\n[INFO] Evaluation finished.")
    print(f"[INFO] Completed episodes: {completed_episodes}/{args_cli.num_episodes}")
    print(f"[INFO] Summary CSV: {summary_csv}")
    print(f"[INFO] Trajectory CSV: {trajectory_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()