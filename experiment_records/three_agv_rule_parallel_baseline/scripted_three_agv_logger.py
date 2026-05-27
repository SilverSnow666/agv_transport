"""Logger for scripted three-AGV cooperative pushing task.

This script runs the centralized scripted controller and records:
1. episode_summary.csv
2. trajectory.csv

It is used to evaluate the three-AGV rule-based cooperative pushing stage.
"""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Log three-AGV scripted pushing task.")
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
    "--num_episodes",
    type=int,
    default=5,
    help="Number of episodes to record.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=3000,
    help="Maximum total simulation steps.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/three_agv_scripted_eval",
    help="Directory for CSV outputs.",
)
parser.add_argument(
    "--eval_contact_threshold",
    type=float,
    default=0.95,
    help="Distance threshold for approximate AGV-payload contact statistics.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agv_transport.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def quat_to_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion in wxyz format to yaw angle."""
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return torch.atan2(siny_cosp, cosy_cosp)


def compute_scripted_actions(obs_policy: torch.Tensor) -> torch.Tensor:
    """Centralized rule controller for three AGVs.

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

    lateral_dir = torch.stack(
        (-push_dir[:, 1], push_dir[:, 0]),
        dim=1,
    )

    # 大 payload 三车并排/浅弧形推送队形
    stand_off_distances = torch.tensor(
        [0.90, 0.90, 0.90],
        device=device,
    )

    lateral_offsets = torch.tensor(
        [0.0, 0.65, -0.65],
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

    dist_to_desired_all = torch.cat(dist_to_desired_list, dim=1)
    max_formation_error = torch.max(dist_to_desired_all, dim=1).values
    formation_ready = max_formation_error < 0.40

    all_actions = []

    for i in range(3):
        agv_xy = agv_xy_list[i]
        agv_heading = agv_heading_list[i]
        desired_xy = desired_xy_list[i]
        dist_to_desired = dist_to_desired_list[i]

        agv_yaw = torch.atan2(agv_heading[:, 1], agv_heading[:, 0])

        agv_to_desired = desired_xy - agv_xy
        approach_dir = agv_to_desired / dist_to_desired

        formation_error_vec = desired_xy - agv_xy
        formation_error_dist = torch.linalg.norm(
            formation_error_vec,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        formation_correction_dir = formation_error_vec / formation_error_dist

        # 推送阶段仍然保持队形，避免三台车挤到一起
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

        w_action = torch.clamp(1.2 * yaw_error, -1.0, 1.0)

        heading_quality = torch.clamp(
            torch.cos(yaw_error),
            min=0.0,
            max=1.0,
        )

        formation_speed = torch.clamp(
            dist_to_desired.squeeze(-1) / 0.8,
            min=0.15,
            max=0.75,
        )

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


def compute_formation_errors(
    base_env,
    payload_xy: torch.Tensor,
    target_xy: torch.Tensor,
) -> torch.Tensor:
    """Compute formation error for three AGVs.

    Return:
        Tensor with shape [num_envs, 3]
    """

    payload_to_target = target_xy - payload_xy
    dist_to_target = torch.linalg.norm(
        payload_to_target,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    push_dir = payload_to_target / dist_to_target

    lateral_dir = torch.stack(
        (-push_dir[:, 1], push_dir[:, 0]),
        dim=1,
    )

    device = payload_xy.device

    stand_off_distances = torch.tensor(
        [0.90, 0.90, 0.90],
        device=device,
    )

    lateral_offsets = torch.tensor(
        [0.0, 0.65, -0.65],
        device=device,
    )

    errors = []

    for i, agv in enumerate(base_env.agvs):
        agv_xy = agv.data.root_pos_w[:, :2]

        desired_xy = (
            payload_xy
            - push_dir * stand_off_distances[i]
            + lateral_dir * lateral_offsets[i]
        )

        error = torch.linalg.norm(agv_xy - desired_xy, dim=1)
        errors.append(error)

    return torch.stack(errors, dim=1)


def compute_contact_flags(
    base_env,
    payload_xy: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Compute approximate contact flags for three AGVs.

    Return:
        Tensor with shape [num_envs, 3]
    """

    contact_flags = []

    for agv in base_env.agvs:
        agv_xy = agv.data.root_pos_w[:, :2]
        agv_payload_dist = torch.linalg.norm(agv_xy - payload_xy, dim=1)
        contact_flags.append(agv_payload_dist < threshold)

    return torch.stack(contact_flags, dim=1)


def get_payload_quat(base_env) -> torch.Tensor:
    """Get payload root quaternion in wxyz format."""

    payload_data = base_env.payload.data

    if hasattr(payload_data, "root_quat_w"):
        return payload_data.root_quat_w

    return payload_data.root_state_w[:, 3:7]

def compute_single_agv_actions(obs_policy: torch.Tensor) -> torch.Tensor:
    """Case A: only AGV1 pushes the large payload.

    AGV2 and AGV3 are disabled.
    Action layout:
        [v1, w1, 0, 0, 0, 0]
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

    # AGV1 observation
    agv1_start = 8
    agv1_xy = obs_policy[:, agv1_start : agv1_start + 2]
    agv1_heading = obs_policy[:, agv1_start + 4 : agv1_start + 6]

    agv1_yaw = torch.atan2(agv1_heading[:, 1], agv1_heading[:, 0])

    # Desired position behind payload
    stand_off_distance = 0.90
    lateral_dir = torch.stack(
        (-push_dir[:, 1], push_dir[:, 0]),
        dim=1,
    )

    single_lateral_offset = -0.55

    desired_xy = (
            payload_xy
            - push_dir * stand_off_distance
            + lateral_dir * single_lateral_offset
    )

    agv_to_desired = desired_xy - agv1_xy
    dist_to_desired = torch.linalg.norm(
        agv_to_desired,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    approach_dir = agv_to_desired / dist_to_desired

    # If AGV1 is close to the pushing position, start pushing.
    ready_to_push = dist_to_desired.squeeze(-1) < 0.25

    desired_vec = torch.where(
        ready_to_push.unsqueeze(-1),
        push_dir,
        approach_dir,
    )

    desired_yaw = torch.atan2(desired_vec[:, 1], desired_vec[:, 0])

    yaw_error = torch.atan2(
        torch.sin(desired_yaw - agv1_yaw),
        torch.cos(desired_yaw - agv1_yaw),
    )

    w1 = torch.clamp(1.2 * yaw_error, -1.0, 1.0)

    heading_quality = torch.clamp(
        torch.cos(yaw_error),
        min=0.0,
        max=1.0,
    )

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

    base_speed = torch.where(
        ready_to_push,
        push_speed,
        approach_speed,
    )

    v1 = heading_quality * base_speed

    actions = torch.zeros(
        (num_envs, 6),
        device=device,
    )

    actions[:, 0] = v1
    actions[:, 1] = w1

    return torch.clamp(actions, -1.0, 1.0)

def main():
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "episode_summary.csv"
    trajectory_csv = output_dir / "trajectory.csv"

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    obs, _ = env.reset()

    num_envs = args_cli.num_envs
    device = base_env.device

    target_radius = max(base_env.cfg.target_radius, 0.25)

    active_episode_id = torch.full(
        (num_envs,),
        -1,
        dtype=torch.long,
        device=device,
    )

    next_episode_id = 0
    for env_id in range(num_envs):
        if next_episode_id < args_cli.num_episodes:
            active_episode_id[env_id] = next_episode_id
            next_episode_id += 1

    episode_reward = torch.zeros(num_envs, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    min_payload_target_dist = torch.full(
        (num_envs,),
        float("inf"),
        device=device,
    )

    payload_yaw_abs_max = torch.zeros(num_envs, device=device)

    contact_steps = torch.zeros(
        (num_envs, 3),
        dtype=torch.long,
        device=device,
    )

    contact_count_sum = torch.zeros(num_envs, device=device)

    formation_error_sum = torch.zeros(num_envs, device=device)
    formation_error_max = torch.zeros(num_envs, device=device)

    completed_episodes = 0
    global_step = 0

    with open(summary_csv, "w", newline="", encoding="utf-8") as summary_file, open(
        trajectory_csv, "w", newline="", encoding="utf-8"
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
                "payload_yaw_abs_max",
                "agv1_contact_ratio",
                "agv2_contact_ratio",
                "agv3_contact_ratio",
                "mean_contact_count",
                "formation_error_mean",
                "formation_error_max",
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
                "payload_x",
                "payload_y",
                "target_x",
                "target_y",
                "payload_target_dist",
                "payload_yaw",
                "agv1_x",
                "agv1_y",
                "agv1_contact",
                "agv2_x",
                "agv2_y",
                "agv2_contact",
                "agv3_x",
                "agv3_y",
                "agv3_contact",
                "formation_error_1",
                "formation_error_2",
                "formation_error_3",
                "action_v1",
                "action_w1",
                "action_v2",
                "action_w2",
                "action_v3",
                "action_w3",
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
                if isinstance(obs, dict):
                    obs_policy = obs["policy"]
                else:
                    obs_policy = obs

                actions = compute_single_agv_actions(obs_policy)

                obs, rewards, terminated, truncated, info = env.step(actions)

                rewards_1d = rewards.reshape(-1).to(device)
                terminated_1d = terminated.reshape(-1).bool().to(device)
                truncated_1d = truncated.reshape(-1).bool().to(device)
                done_1d = terminated_1d | truncated_1d

                payload_xy = base_env.payload.data.root_pos_w[:, :2]
                target_xy = base_env._get_target_xy()

                payload_target_dist = torch.linalg.norm(
                    payload_xy - target_xy,
                    dim=1,
                )

                payload_quat = get_payload_quat(base_env)
                payload_yaw = quat_to_yaw_wxyz(payload_quat)

                contact_flags = compute_contact_flags(
                    base_env,
                    payload_xy,
                    args_cli.eval_contact_threshold,
                )

                contact_count = contact_flags.float().sum(dim=1)

                formation_errors = compute_formation_errors(
                    base_env,
                    payload_xy,
                    target_xy,
                )

                formation_error_mean_step = formation_errors.mean(dim=1)
                formation_error_max_step = formation_errors.max(dim=1).values

                success = payload_target_dist < target_radius

                valid_env = active_episode_id >= 0

                episode_reward[valid_env] += rewards_1d[valid_env]
                episode_steps[valid_env] += 1

                min_payload_target_dist[valid_env] = torch.minimum(
                    min_payload_target_dist[valid_env],
                    payload_target_dist[valid_env],
                )

                payload_yaw_abs_max[valid_env] = torch.maximum(
                    payload_yaw_abs_max[valid_env],
                    torch.abs(payload_yaw[valid_env]),
                )

                contact_steps[valid_env] += contact_flags[valid_env].long()
                contact_count_sum[valid_env] += contact_count[valid_env]

                formation_error_sum[valid_env] += formation_error_mean_step[valid_env]
                formation_error_max[valid_env] = torch.maximum(
                    formation_error_max[valid_env],
                    formation_error_max_step[valid_env],
                )

                actions_cpu = actions.detach().cpu()

                for env_id in range(num_envs):
                    ep_id = int(active_episode_id[env_id].item())
                    if ep_id < 0:
                        continue

                    agv_positions = [
                        base_env.agv1.data.root_pos_w[env_id, :2],
                        base_env.agv2.data.root_pos_w[env_id, :2],
                        base_env.agv3.data.root_pos_w[env_id, :2],
                    ]

                    traj_writer.writerow(
                        [
                            global_step,
                            ep_id,
                            env_id,
                            int(episode_steps[env_id].item()),
                            float(payload_xy[env_id, 0].item()),
                            float(payload_xy[env_id, 1].item()),
                            float(target_xy[env_id, 0].item()),
                            float(target_xy[env_id, 1].item()),
                            float(payload_target_dist[env_id].item()),
                            float(payload_yaw[env_id].item()),
                            float(agv_positions[0][0].item()),
                            float(agv_positions[0][1].item()),
                            bool(contact_flags[env_id, 0].item()),
                            float(agv_positions[1][0].item()),
                            float(agv_positions[1][1].item()),
                            bool(contact_flags[env_id, 1].item()),
                            float(agv_positions[2][0].item()),
                            float(agv_positions[2][1].item()),
                            bool(contact_flags[env_id, 2].item()),
                            float(formation_errors[env_id, 0].item()),
                            float(formation_errors[env_id, 1].item()),
                            float(formation_errors[env_id, 2].item()),
                            float(actions_cpu[env_id, 0].item()),
                            float(actions_cpu[env_id, 1].item()),
                            float(actions_cpu[env_id, 2].item()),
                            float(actions_cpu[env_id, 3].item()),
                            float(actions_cpu[env_id, 4].item()),
                            float(actions_cpu[env_id, 5].item()),
                            float(rewards_1d[env_id].item()),
                            bool(terminated_1d[env_id].item()),
                            bool(truncated_1d[env_id].item()),
                            bool(success[env_id].item()),
                        ]
                    )

                for env_id in range(num_envs):
                    if not bool(done_1d[env_id].item()):
                        continue

                    ep_id = int(active_episode_id[env_id].item())
                    if ep_id < 0:
                        continue

                    steps = int(episode_steps[env_id].item())
                    steps_safe = max(steps, 1)

                    contact_ratios = (
                        contact_steps[env_id].float() / steps_safe
                    )

                    mean_contact_count = (
                        contact_count_sum[env_id] / steps_safe
                    )

                    formation_error_mean = (
                        formation_error_sum[env_id] / steps_safe
                    )

                    terminal_success = (
                        float(min_payload_target_dist[env_id].item())
                        < target_radius
                    )

                    final_dist_for_summary = float(
                        min_payload_target_dist[env_id].item()
                    )

                    summary_writer.writerow(
                        [
                            ep_id,
                            env_id,
                            terminal_success,
                            steps,
                            steps * base_env.step_dt,
                            final_dist_for_summary,
                            float(min_payload_target_dist[env_id].item()),
                            float(payload_yaw_abs_max[env_id].item()),
                            float(contact_ratios[0].item()),
                            float(contact_ratios[1].item()),
                            float(contact_ratios[2].item()),
                            float(mean_contact_count.item()),
                            float(formation_error_mean.item()),
                            float(formation_error_max[env_id].item()),
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
                        f"yaw_abs_max={float(payload_yaw_abs_max[env_id].item()):.3f}, "
                        f"contact=({float(contact_ratios[0].item()):.2f}, "
                        f"{float(contact_ratios[1].item()):.2f}, "
                        f"{float(contact_ratios[2].item()):.2f}), "
                        f"mean_contact_count={float(mean_contact_count.item()):.2f}, "
                        f"formation_error_mean={float(formation_error_mean.item()):.3f}, "
                        f"reward={float(episode_reward[env_id].item()):.2f}"
                    )

                    if next_episode_id < args_cli.num_episodes:
                        active_episode_id[env_id] = next_episode_id
                        next_episode_id += 1
                    else:
                        active_episode_id[env_id] = -1

                    episode_reward[env_id] = 0.0
                    episode_steps[env_id] = 0
                    min_payload_target_dist[env_id] = float("inf")
                    payload_yaw_abs_max[env_id] = 0.0
                    contact_steps[env_id, :] = 0
                    contact_count_sum[env_id] = 0.0
                    formation_error_sum[env_id] = 0.0
                    formation_error_max[env_id] = 0.0

            global_step += 1

    env.close()

    print("\n[INFO] Three-AGV scripted evaluation finished.")
    print(f"[INFO] Completed episodes: {completed_episodes}/{args_cli.num_episodes}")
    print(f"[INFO] Summary CSV: {summary_csv}")
    print(f"[INFO] Trajectory CSV: {trajectory_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()