"""Evaluate centralized PPO policy for three-AGV cooperative pushing.

This script loads a trained skrl PPO checkpoint and records:
1. episode_summary.csv
2. trajectory.csv

It is used to compare PPO policy with the scripted three-AGV rule baseline.

Expected environment:
    - action_space = 6
    - observation_space = 38
    - 3 AGVs + 1 large payload + 1 target

Action layout:
    [v1, w1, v2, w2, v3, w3]
"""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate three-AGV centralized PPO policy."
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
    help="Distance threshold for approximate AGV-payload contact statistics.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import agv_transport.tasks  # noqa: F401


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


def compute_contact_flags(
    base_env,
    payload_xy: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Compute approximate contact flags for three AGVs.

    Returns:
        Tensor with shape [num_envs, 3]
    """
    contact_flags = []

    for agv in base_env.agvs:
        agv_xy = agv.data.root_pos_w[:, :2]
        agv_payload_dist = torch.linalg.norm(
            agv_xy - payload_xy,
            dim=1,
        )
        contact_flags.append(agv_payload_dist < threshold)

    return torch.stack(contact_flags, dim=1)


def compute_formation_errors(
    base_env,
    payload_xy: torch.Tensor,
    target_xy: torch.Tensor,
) -> torch.Tensor:
    """Compute formation error for three AGVs.

    Returns:
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

    if hasattr(base_env.cfg, "formation_stand_off_distances"):
        stand_off_distances = torch.tensor(
            base_env.cfg.formation_stand_off_distances,
            device=device,
        )
    else:
        stand_off_distances = torch.tensor(
            [0.90, 0.90, 0.90],
            device=device,
        )

    if hasattr(base_env.cfg, "formation_lateral_offsets"):
        lateral_offsets = torch.tensor(
            base_env.cfg.formation_lateral_offsets,
            device=device,
        )
    else:
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

        error = torch.linalg.norm(
            agv_xy - desired_xy,
            dim=1,
        )
        errors.append(error)

    return torch.stack(errors, dim=1)


def get_runner_action(runner: Runner, obs: torch.Tensor) -> torch.Tensor:
    """Get deterministic/evaluation action from skrl runner agent."""
    with torch.inference_mode():
        actions = runner.agent.act(
            obs,
            timestep=0,
            timesteps=0,
        )[0]

    return actions


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "episode_summary.csv"
    trajectory_csv = output_dir / "trajectory.csv"

    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Override environment settings from command line.
    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Disable skrl writing during evaluation.
    if "experiment" in agent_cfg.get("agent", {}):
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = raw_env.unwrapped

    env = SkrlVecEnvWrapper(
        raw_env,
        ml_framework="torch",
    )

    runner = Runner(
        env,
        agent_cfg,
    )

    print(f"[INFO] Loading PPO checkpoint from: {checkpoint_path}")
    runner.agent.load(str(checkpoint_path))
    runner.set_running_mode("eval")

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
                actions = get_runner_action(runner, obs)

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

                    contact_ratios = contact_steps[env_id].float() / steps_safe

                    mean_contact_count = contact_count_sum[env_id] / steps_safe

                    formation_error_mean = formation_error_sum[env_id] / steps_safe

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

    print("\n[INFO] Three-AGV PPO evaluation finished.")
    print(f"[INFO] Completed episodes: {completed_episodes}/{args_cli.num_episodes}")
    print(f"[INFO] Summary CSV: {summary_csv}")
    print(f"[INFO] Trajectory CSV: {trajectory_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()