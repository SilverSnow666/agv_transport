"""Scripted push agent with CSV logging."""

import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted push agent with CSV logging.")
parser.add_argument("--task", type=str, default="Template-Agv-Transport-Direct-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=1200, help="Maximum number of simulation steps.")
parser.add_argument("--output", type=str, default="logs/scripted_push_log.csv", help="CSV output path.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agv_transport.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def compute_scripted_actions(obs_policy: torch.Tensor) -> torch.Tensor:
    """Compute scripted pushing action from 14-D observation.

    observation:
        0:2   agv_xy_rel
        2:4   payload_xy_rel
        4:6   target_xy_rel
        6:8   agv_to_payload_xy
        8:10  payload_to_target_xy
        10:12 agv_vel_xy
        12:14 payload_vel_xy
    """

    agv_xy = obs_policy[:, 0:2]
    payload_xy = obs_policy[:, 2:4]
    payload_to_target = obs_policy[:, 8:10]

    dist_to_target = torch.linalg.norm(payload_to_target, dim=1, keepdim=True).clamp_min(1e-6)
    push_dir = payload_to_target / dist_to_target

    stand_off_distance = 0.75
    desired_agv_xy = payload_xy - push_dir * stand_off_distance

    agv_to_desired = desired_agv_xy - agv_xy
    dist_to_desired = torch.linalg.norm(agv_to_desired, dim=1, keepdim=True).clamp_min(1e-6)
    move_to_desired_dir = agv_to_desired / dist_to_desired

    arrive_threshold = 0.18
    reached_push_pose = dist_to_desired < arrive_threshold

    actions = torch.where(
        reached_push_pose,
        push_dir,
        move_to_desired_dir,
    )

    slow_down = torch.clamp(dist_to_target / 0.8, min=0.25, max=1.0)
    actions = actions * slow_down

    return torch.clamp(actions, -1.0, 1.0)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)

    with open(args_cli.output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "step",
                "env_id",
                "agv_x",
                "agv_y",
                "payload_x",
                "payload_y",
                "target_x",
                "target_y",
                "payload_target_dist",
                "action_v",
                "action_w",
                "reward",
                "terminated",
                "truncated",
            ]
        )

        step_count = 0

        while simulation_app.is_running() and step_count < args_cli.max_steps:
            with torch.inference_mode():
                obs_policy = obs["policy"]
                actions = compute_scripted_actions(obs_policy)

                obs, rewards, terminated, truncated, infos = env.step(actions)

                # 记录 observation 中的状态
                obs_policy = obs["policy"]

                agv_xy = obs_policy[:, 0:2]
                payload_xy = obs_policy[:, 2:4]
                target_xy = obs_policy[:, 4:6]
                payload_to_target = obs_policy[:, 8:10]
                payload_target_dist = torch.linalg.norm(payload_to_target, dim=1)

                for env_id in range(args_cli.num_envs):
                    writer.writerow(
                        [
                            step_count,
                            env_id,
                            float(agv_xy[env_id, 0].cpu()),
                            float(agv_xy[env_id, 1].cpu()),
                            float(payload_xy[env_id, 0].cpu()),
                            float(payload_xy[env_id, 1].cpu()),
                            float(target_xy[env_id, 0].cpu()),
                            float(target_xy[env_id, 1].cpu()),
                            float(payload_target_dist[env_id].cpu()),
                            float(actions[env_id, 0].cpu()),
                            float(actions[env_id, 1].cpu()),
                            float(rewards[env_id].cpu()),
                            bool(terminated[env_id].cpu()),
                            bool(truncated[env_id].cpu()),
                        ]
                    )

                if step_count % 100 == 0:
                    mean_dist = payload_target_dist.mean().item()
                    mean_reward = rewards.mean().item()
                    print(
                        f"[step {step_count:05d}] "
                        f"mean_payload_target_dist={mean_dist:.3f}, "
                        f"mean_reward={mean_reward:.3f}"
                    )

            step_count += 1

    env.close()
    print(f"[INFO] CSV log saved to: {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()