"""Diagnose payload pushing physics consistency.

This script does not train any policy.
It uses scripted actions to test whether eccentric pushing causes payload yaw.

Cases:
    single_center   : only AGV1 pushes from center
    single_offset_p : only AGV1 pushes from positive lateral offset
    single_offset_n : only AGV1 pushes from negative lateral offset
    three_symmetric : three AGVs push symmetrically
"""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose payload pushing physics.")
parser.add_argument("--task", type=str, default="Template-Agv-Transport-Direct-v0")
parser.add_argument(
    "--case",
    type=str,
    default="single_center",
    choices=[
        "single_center",
        "single_offset_p",
        "single_offset_n",
        "three_symmetric",
        "straight_agv1",
        "straight_agv2",
        "straight_agv3",
    ],
    help="Diagnosis case.",
)

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=720)
parser.add_argument("--output_dir", type=str, default="logs/diagnose_payload_physics")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agv_transport.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def compute_single_agv_action(obs_policy: torch.Tensor, lateral_offset: float) -> torch.Tensor:
    """Single AGV scripted pushing.

    Uses AGV1 only.
    AGV2 and AGV3 are disabled.
    """
    device = obs_policy.device
    num_envs = obs_policy.shape[0]

    payload_xy = obs_policy[:, 0:2]
    payload_to_target = obs_policy[:, 4:6]

    dist_to_target = torch.linalg.norm(payload_to_target, dim=1, keepdim=True).clamp_min(1e-6)
    push_dir = payload_to_target / dist_to_target

    lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=1)

    agv_start = 8
    agv_xy = obs_policy[:, agv_start:agv_start + 2]
    agv_heading = obs_policy[:, agv_start + 4:agv_start + 6]
    agv_yaw = torch.atan2(agv_heading[:, 1], agv_heading[:, 0])

    stand_off_distance = 0.90

    desired_xy = (
        payload_xy
        - push_dir * stand_off_distance
        + lateral_dir * lateral_offset
    )

    agv_to_desired = desired_xy - agv_xy
    dist_to_desired = torch.linalg.norm(agv_to_desired, dim=1, keepdim=True).clamp_min(1e-6)
    approach_dir = agv_to_desired / dist_to_desired

    ready_to_push = dist_to_desired.squeeze(-1) < 0.25

    desired_vec = torch.where(
        ready_to_push.unsqueeze(-1),
        push_dir,
        approach_dir,
    )

    desired_yaw = torch.atan2(desired_vec[:, 1], desired_vec[:, 0])

    yaw_error = torch.atan2(
        torch.sin(desired_yaw - agv_yaw),
        torch.cos(desired_yaw - agv_yaw),
    )

    w = torch.clamp(1.2 * yaw_error, -1.0, 1.0)

    heading_quality = torch.clamp(torch.cos(yaw_error), min=0.0, max=1.0)

    approach_speed = torch.clamp(
        dist_to_desired.squeeze(-1) / 0.8,
        min=0.15,
        max=0.75,
    )

    push_speed = torch.clamp(
        dist_to_target.squeeze(-1) / 1.2,
        min=0.25,
        max=0.75,
    )

    base_speed = torch.where(ready_to_push, push_speed, approach_speed)
    v = heading_quality * base_speed

    actions = torch.zeros((num_envs, 6), device=device)
    actions[:, 0] = v
    actions[:, 1] = w

    return torch.clamp(actions, -1.0, 1.0)


def compute_three_agv_action(obs_policy: torch.Tensor) -> torch.Tensor:
    """Three AGV symmetric scripted pushing."""
    device = obs_policy.device
    num_envs = obs_policy.shape[0]

    payload_xy = obs_policy[:, 0:2]
    payload_to_target = obs_policy[:, 4:6]

    dist_to_target = torch.linalg.norm(payload_to_target, dim=1, keepdim=True).clamp_min(1e-6)
    push_dir = payload_to_target / dist_to_target

    lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=1)

    stand_off_distances = torch.tensor([0.90, 0.90, 0.90], device=device)
    lateral_offsets = torch.tensor([0.0, 0.65, -0.65], device=device)

    agv_xy_list = []
    agv_heading_list = []
    desired_xy_list = []
    dist_list = []

    for i in range(3):
        start = 8 + i * 8
        agv_xy = obs_policy[:, start:start + 2]
        agv_heading = obs_policy[:, start + 4:start + 6]

        desired_xy = (
            payload_xy
            - push_dir * stand_off_distances[i]
            + lateral_dir * lateral_offsets[i]
        )

        diff = desired_xy - agv_xy
        dist = torch.linalg.norm(diff, dim=1, keepdim=True).clamp_min(1e-6)

        agv_xy_list.append(agv_xy)
        agv_heading_list.append(agv_heading)
        desired_xy_list.append(desired_xy)
        dist_list.append(dist)

    max_formation_error = torch.cat(dist_list, dim=1).max(dim=1).values
    formation_ready = max_formation_error < 0.40

    actions_parts = []

    for i in range(3):
        agv_xy = agv_xy_list[i]
        agv_heading = agv_heading_list[i]
        desired_xy = desired_xy_list[i]
        dist = dist_list[i]

        agv_yaw = torch.atan2(agv_heading[:, 1], agv_heading[:, 0])

        diff = desired_xy - agv_xy
        approach_dir = diff / dist

        correction_dir = approach_dir
        correction_gain = 0.5

        push_with_formation = push_dir + correction_gain * correction_dir
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

        w = torch.clamp(1.2 * yaw_error, -1.0, 1.0)
        heading_quality = torch.clamp(torch.cos(yaw_error), min=0.0, max=1.0)

        formation_speed = torch.clamp(
            dist.squeeze(-1) / 0.8,
            min=0.15,
            max=0.75,
        )

        push_speed = torch.clamp(
            dist_to_target.squeeze(-1) / 1.2,
            min=0.25,
            max=0.75,
        )

        base_speed = torch.where(formation_ready, push_speed, formation_speed)
        v = heading_quality * base_speed

        actions_parts.append(v)
        actions_parts.append(w)

    actions = torch.stack(actions_parts, dim=1)

    if actions.shape != (num_envs, 6):
        raise RuntimeError(f"Expected action shape ({num_envs}, 6), got {actions.shape}")

    return torch.clamp(actions, -1.0, 1.0)

def align_all_agvs_to_push_direction(base_env) -> None:
    """Align all AGV headings with the payload-to-target direction.

    This makes the straight-pushing tests independent of steering angle.
    """
    with torch.inference_mode():
        payload_xy = base_env.payload.data.root_pos_w[:, :2]
        target_xy = base_env._get_target_xy()

        payload_to_target = target_xy - payload_xy
        dist = torch.linalg.norm(
            payload_to_target,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        push_dir = payload_to_target / dist
        push_yaw = torch.atan2(push_dir[:, 1], push_dir[:, 0])

        base_env.agv_yaw[:, :] = push_yaw.unsqueeze(1)

        for i, agv in enumerate(base_env.agvs):
            agv_state = agv.data.root_state_w.clone()
            agv_state[:, 3:7] = base_env._yaw_to_quat(push_yaw)
            agv_state[:, 7:] = 0.0

            agv.write_root_pose_to_sim(agv_state[:, :7])
            agv.write_root_velocity_to_sim(agv_state[:, 7:])

def compute_straight_selected_agv_action(
    obs_policy: torch.Tensor,
    agv_index: int,
    speed: float = 0.60,
) -> torch.Tensor:
    """Selected AGV moves straight forward without steering.

    This is used to isolate the effect of eccentric contact point.
    The selected AGV has:
        v = constant positive speed
        w = 0

    Other AGVs are disabled.
    """
    device = obs_policy.device
    num_envs = obs_policy.shape[0]

    actions = torch.zeros((num_envs, 6), device=device)

    actions[:, 2 * agv_index] = speed
    actions[:, 2 * agv_index + 1] = 0.0

    return torch.clamp(actions, -1.0, 1.0)

def select_actions(case_name: str, obs_policy: torch.Tensor) -> torch.Tensor:
    if case_name == "single_center":
        return compute_single_agv_action(obs_policy, lateral_offset=0.0)

    if case_name == "single_offset_p":
        return compute_single_agv_action(obs_policy, lateral_offset=0.65)

    if case_name == "single_offset_n":
        return compute_single_agv_action(obs_policy, lateral_offset=-0.65)

    if case_name == "three_symmetric":
        return compute_three_agv_action(obs_policy)

    if case_name == "straight_agv1":
        return compute_straight_selected_agv_action(
            obs_policy,
            agv_index=0,
            speed=0.60,
        )

    if case_name == "straight_agv2":
        return compute_straight_selected_agv_action(
            obs_policy,
            agv_index=1,
            speed=0.60,
        )

    if case_name == "straight_agv3":
        return compute_straight_selected_agv_action(
            obs_policy,
            agv_index=2,
            speed=0.60,
        )

    raise ValueError(f"Unknown case: {case_name}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    obs, _ = env.reset()
    if args_cli.case in ["straight_agv1", "straight_agv2", "straight_agv3"]:
        align_all_agvs_to_push_direction(base_env)

    output_dir = Path(args_cli.output_dir) / args_cli.case
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "trajectory.csv"

    yaw_abs_max = torch.zeros(args_cli.num_envs, device=base_env.device)
    min_dist = torch.full((args_cli.num_envs,), float("inf"), device=base_env.device)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "payload_x",
                "payload_y",
                "target_x",
                "target_y",
                "payload_target_dist",
                "payload_yaw",
                "payload_yaw_abs",
                "agv1_x",
                "agv1_y",
                "agv2_x",
                "agv2_y",
                "agv3_x",
                "agv3_y",
                "reward",
                "terminated",
                "truncated",
            ]
        )

        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                obs_policy = obs["policy"] if isinstance(obs, dict) else obs
                actions = select_actions(args_cli.case, obs_policy)

                obs, reward, terminated, truncated, info = env.step(actions)

                payload_xy = base_env.payload.data.root_pos_w[:, :2]
                target_xy = base_env._get_target_xy()
                dist = torch.linalg.norm(payload_xy - target_xy, dim=1)

                payload_yaw = base_env._get_payload_yaw()
                payload_yaw_abs = torch.abs(payload_yaw)

                yaw_abs_max = torch.maximum(yaw_abs_max, payload_yaw_abs)
                min_dist = torch.minimum(min_dist, dist)

                agv1_xy = base_env.agv1.data.root_pos_w[:, :2]
                agv2_xy = base_env.agv2.data.root_pos_w[:, :2]
                agv3_xy = base_env.agv3.data.root_pos_w[:, :2]

                writer.writerow(
                    [
                        step,
                        float(payload_xy[0, 0].item()),
                        float(payload_xy[0, 1].item()),
                        float(target_xy[0, 0].item()),
                        float(target_xy[0, 1].item()),
                        float(dist[0].item()),
                        float(payload_yaw[0].item()),
                        float(payload_yaw_abs[0].item()),
                        float(agv1_xy[0, 0].item()),
                        float(agv1_xy[0, 1].item()),
                        float(agv2_xy[0, 0].item()),
                        float(agv2_xy[0, 1].item()),
                        float(agv3_xy[0, 0].item()),
                        float(agv3_xy[0, 1].item()),
                        float(reward.reshape(-1)[0].item()),
                        bool(terminated.reshape(-1)[0].item()),
                        bool(truncated.reshape(-1)[0].item()),
                    ]
                )

                if bool(terminated.reshape(-1)[0].item()) or bool(truncated.reshape(-1)[0].item()):
                    break

    print("\n[INFO] Physics diagnosis finished")
    print(f"[INFO] Case: {args_cli.case}")
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] min_dist: {float(min_dist[0].item()):.3f}")
    print(f"[INFO] yaw_abs_max: {float(yaw_abs_max[0].item()):.3f} rad")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()