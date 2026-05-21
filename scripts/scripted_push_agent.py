# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.

"""Scripted agent for single-AGV pushing task."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Scripted push agent for AGV transport task.")
parser.add_argument("--task", type=str, default="Template-Agv-Transport-Direct-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=3000, help="Maximum number of simulation steps.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agv_transport.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def compute_scripted_actions(obs_policy: torch.Tensor) -> torch.Tensor:
    """差速 AGV 手写推箱子控制器。

    observation:
        0:2    agv_xy_rel
        2:4    payload_xy_rel
        4:6    target_xy_rel
        6:8    agv_to_payload_xy
        8:10   payload_to_target_xy
        10:12  agv_heading_xy
        12:14  agv_vel_xy
        14:16  payload_vel_xy

    action:
        0: v，归一化线速度
        1: w，归一化角速度
    """

    agv_xy = obs_policy[:, 0:2]
    payload_xy = obs_policy[:, 2:4]
    payload_to_target = obs_policy[:, 8:10]
    agv_heading = obs_policy[:, 10:12]

    # 当前 AGV yaw
    agv_yaw = torch.atan2(agv_heading[:, 1], agv_heading[:, 0])

    # payload 指向 target 的方向
    dist_to_target = torch.linalg.norm(payload_to_target, dim=1, keepdim=True).clamp_min(1e-6)
    push_dir = payload_to_target / dist_to_target

    # 期望 AGV 位于 payload 后方
    stand_off_distance = 0.75
    desired_agv_xy = payload_xy - push_dir * stand_off_distance

    agv_to_desired = desired_agv_xy - agv_xy
    dist_to_desired = torch.linalg.norm(agv_to_desired, dim=1, keepdim=True).clamp_min(1e-6)

    # 如果还没到推送位姿，朝推送位姿运动
    # 如果已经到达推送位姿，朝目标方向推动
    arrive_threshold = 0.18
    reached_push_pose = dist_to_desired < arrive_threshold

    desired_vec = torch.where(
        reached_push_pose,
        push_dir,
        agv_to_desired / dist_to_desired,
    )

    desired_yaw = torch.atan2(desired_vec[:, 1], desired_vec[:, 0])
    yaw_error = torch.atan2(
        torch.sin(desired_yaw - agv_yaw),
        torch.cos(desired_yaw - agv_yaw),
    )

    # 角速度控制
    w_action = torch.clamp(1.5 * yaw_error, -1.0, 1.0)

    # 朝向误差较大时先转向，误差小时再前进
    heading_quality = torch.clamp(torch.cos(yaw_error), min=0.0, max=1.0)

    # 接近目标时减速
    slow_down = torch.clamp(dist_to_target.squeeze(-1) / 0.8, min=0.25, max=1.0)

    v_action = heading_quality * slow_down

    actions = torch.stack((v_action, w_action), dim=1)
    return torch.clamp(actions, -1.0, 1.0)

def main():
    """Run scripted push agent."""

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    obs, _ = env.reset()

    step_count = 0

    while simulation_app.is_running() and step_count < args_cli.max_steps:
        with torch.inference_mode():
            obs_policy = obs["policy"]

            actions = compute_scripted_actions(obs_policy)

            obs, rewards, terminated, truncated, infos = env.step(actions)

            if step_count % 100 == 0:
                mean_reward = rewards.mean().item()
                done_count = (terminated | truncated).sum().item()
                print(
                    f"[step {step_count:05d}] "
                    f"mean_reward={mean_reward:.3f}, "
                    f"done_count={done_count}"
                )

        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()