"""Scripted centralized controller for three-AGV pushing task.

This script is only for testing the three-AGV environment.
It does not train any policy.

Observation layout:
    0:2    payload_xy_rel
    2:4    target_xy_rel
    4:6    payload_to_target_xy
    6:8    payload_vel_xy

    For AGV i, i = 0, 1, 2:
        start = 8 + i * 8
        start + 0:2  agv_xy_rel
        start + 2:4  agv_to_payload_xy
        start + 4:6  agv_heading_xy
        start + 6:8  agv_vel_xy

Action layout:
    [v1, w1, v2, w2, v3, w3]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Three-AGV scripted pushing controller.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Agv-Transport-Direct-v0",
    help="Task name.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=2000,
    help="Maximum simulation steps.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agv_transport.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def compute_scripted_actions(obs_policy: torch.Tensor) -> torch.Tensor:
    """Centralized rule controller for three AGVs.

    The controller has two phases:
    1. Formation phase:
       Three AGVs move to desired positions behind the payload.
    2. Pushing phase:
       Once the formation is approximately reached, all AGVs push toward target.
    """

    device = obs_policy.device
    num_envs = obs_policy.shape[0]

    payload_xy = obs_policy[:, 0:2]
    payload_to_target = obs_policy[:, 4:6]

    dist_to_target = torch.linalg.norm(
        payload_to_target,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    push_dir = payload_to_target / dist_to_target

    # Lateral direction perpendicular to push direction.
    # If push_dir = [1, 0], lateral_dir = [0, 1].
    lateral_dir = torch.stack(
        (-push_dir[:, 1], push_dir[:, 0]),
        dim=1,
    )

    # 三车浅三角接触队形：
    # 三台车都尽量靠近 payload 后方，外侧车只略微靠后
    stand_off_distances = torch.tensor(
        [0.90, 0.90, 0.90],
        device=device,
    )

    lateral_offsets = torch.tensor(
        [0.0, 0.60, -0.60],
        device=device,
    )

    agv_xy_list = []
    agv_heading_list = []
    desired_xy_list = []
    dist_to_desired_list = []

    for i in range(3):
        start = 8 + i * 8

        agv_xy = obs_policy[:, start : start + 2]
        agv_heading = obs_policy[:, start + 4 : start + 6]

        desired_xy = (
                payload_xy
                - push_dir * stand_off_distances[i]
                + lateral_dir * lateral_offsets[i]
        )

        agv_to_desired = desired_xy - agv_xy
        dist_to_desired = torch.linalg.norm(
            agv_to_desired,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        agv_xy_list.append(agv_xy)
        agv_heading_list.append(agv_heading)
        desired_xy_list.append(desired_xy)
        dist_to_desired_list.append(dist_to_desired)

    # If all three AGVs are close to their formation positions, start pushing.
    dist_to_desired_all = torch.cat(dist_to_desired_list, dim=1)
    max_formation_error = torch.max(dist_to_desired_all, dim=1).values
    formation_ready = max_formation_error < 0.40

    all_actions = []

    # 推送阶段的方向：
    # 中间车正向推，外侧两车略微向内推，减少推偏并增加角部接触概率
    inward_gain = 0.12

    push_vec_list = [
        push_dir,
        push_dir - inward_gain * lateral_dir,
        push_dir + inward_gain * lateral_dir,
    ]

    push_vec_list = [
        vec / torch.linalg.norm(vec, dim=1, keepdim=True).clamp_min(1e-6)
        for vec in push_vec_list
    ]

    for i in range(3):
        agv_xy = agv_xy_list[i]
        agv_heading = agv_heading_list[i]
        desired_xy = desired_xy_list[i]
        dist_to_desired = dist_to_desired_list[i]

        agv_yaw = torch.atan2(agv_heading[:, 1], agv_heading[:, 0])

        agv_to_desired = desired_xy - agv_xy
        approach_dir = agv_to_desired / dist_to_desired

        # 始终保持队形：即使进入推送阶段，也继续追踪 payload 后方的移动队形点
        # 这样三台车不会在推送过程中挤到中间。
        formation_error_vec = desired_xy - agv_xy
        formation_error_dist = torch.linalg.norm(
            formation_error_vec,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        formation_correction_dir = formation_error_vec / formation_error_dist

        # 推送阶段：前进方向 + 队形修正方向
        # correction_gain 越大，三车越重视保持间距；越小，越重视向前推。
        correction_gain = 0.5

        push_with_formation = push_dir + correction_gain * formation_correction_dir
        push_with_formation = push_with_formation / torch.linalg.norm(
            push_with_formation,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        desired_vec = torch.where(
            formation_ready.unsqueeze(-1),
            push_with_formation,
            approach_dir,
        )

        desired_yaw = torch.atan2(desired_vec[:, 1], desired_vec[:, 0])

        yaw_error = torch.atan2(
            torch.sin(desired_yaw - agv_yaw),
            torch.cos(desired_yaw - agv_yaw),
        )

        # Normalized angular action
        w_action = torch.clamp(1.8 * yaw_error, -1.0, 1.0)

        # Move only when heading is reasonably aligned.
        heading_quality = torch.clamp(
            torch.cos(yaw_error),
            min=0.0,
            max=1.0,
        )

        # Formation phase speed: depends on distance to desired formation point.
        formation_speed = torch.clamp(
            dist_to_desired.squeeze(-1) / 0.8,
            min=0.15,
            max=0.75,
        )

        # Pushing phase speed: slows down near target.
        pushing_speed = torch.clamp(
            dist_to_target.squeeze(-1) / 1.2,
            min=0.25,
            max=0.75,
        )

        base_speed = torch.where(
            formation_ready,
            pushing_speed,
            formation_speed,
        )

        # If an AGV already reached its formation point while others have not,
        # let it wait instead of disturbing the payload.
        wait_in_formation = (
            (~formation_ready)
            & (dist_to_desired.squeeze(-1) < 0.15)
        )

        v_action = heading_quality * base_speed
        v_action = torch.where(
            wait_in_formation,
            torch.zeros_like(v_action),
            v_action,
        )

        all_actions.append(v_action)
        all_actions.append(w_action)

    actions = torch.stack(all_actions, dim=1)

    if actions.shape != (num_envs, 6):
        raise RuntimeError(
            f"Expected actions shape ({num_envs}, 6), got {actions.shape}"
        )

    return torch.clamp(actions, -1.0, 1.0)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    obs, _ = env.reset()

    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            if isinstance(obs, dict):
                obs_policy = obs["policy"]
            else:
                obs_policy = obs

            actions = compute_scripted_actions(obs_policy)

            obs, reward, terminated, truncated, info = env.step(actions)

        step_count += 1

        if step_count >= args_cli.max_steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()