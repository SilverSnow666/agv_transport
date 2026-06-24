"""Scripted controller for V6.0 three-AGV carrying task.

Purpose:
- Verify that the new carrying environment can reset, step and move the payload.
- This is not an optimal controller and should not replace PPO training.

Observation layout from AgvCarryEnv:
0:2   payload_xy_rel
2:3   payload_z
3:5   target_xy_rel
5:7   payload_to_target normalized-ish
7:8   target_dist normalized-ish
8:10  move_dir
10:12 lateral_dir
12:15 payload_rpy
15:18 payload_lin_vel
18:21 payload_ang_vel
21:22 slip_error
22:23 support_margin
23:26 contact_flags
For each AGV i:
  start = 26 + i * 10
  start+0:start+2   agv_xy_rel
  start+2:start+3   agv_z
  start+3:start+5   agv_heading
  start+5:start+7   agv_vel_xy
  start+7:start+9   agv_to_support_target
  start+9:start+10  formation_error
Action layout: [v1, w1, v2, w2, v3, w3]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Three-AGV scripted carrying controller.")
parser.add_argument("--task", type=str, default="Template-Agv-Carry-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=1200, help="Maximum simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import agv_transport.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def compute_scripted_actions(obs_policy: torch.Tensor) -> torch.Tensor:
    """稳定版 V6.0 scripted controller。

    关键改动：
    - 不再让每台 AGV 直接朝自己的 support target 反向追点；
    - 三台车以 payload->target 的 move_dir 为主航向同步前进；
    - 队形误差只用于“加减速 + 小角度侧向修正”，避免出现 180° 掉头和原地打转。

    这个控制器只用于验证环境最小闭环，不代表最终 PPO 策略。
    """
    device = obs_policy.device
    num_envs = obs_policy.shape[0]

    # observation 中 target_dist 是 /3.5 之后的归一化量，这里还原成近似米制距离。
    target_dist_m = obs_policy[:, 7] * 3.5
    move_dir = obs_policy[:, 8:10]
    lateral_dir = obs_policy[:, 10:12]
    roll_pitch_abs = torch.linalg.norm(obs_policy[:, 12:14], dim=1)
    slip_error = obs_policy[:, 21]

    # 不让稳定门控降到 0，否则车只会原地调头/等待，payload 更容易滑移。
    stability_gate = torch.clamp(1.0 - roll_pitch_abs / 0.25 - slip_error / 0.30, min=0.35, max=1.0)

    # 基础前进速度：接近目标时自动降速。
    base_speed = torch.clamp(target_dist_m / 3.0, min=0.18, max=0.52) * stability_gate
    near_goal = target_dist_m < 0.30
    base_speed = torch.where(near_goal, 0.35 * base_speed, base_speed)

    all_actions = []
    for i in range(3):
        start = 26 + i * 10
        agv_heading = obs_policy[:, start + 3 : start + 5]
        agv_to_support = obs_policy[:, start + 7 : start + 9]

        # 将队形误差投影到运输方向和横向方向。
        # long_error > 0：AGV 落后于支撑点，应略微加速。
        # long_error < 0：AGV 超前于支撑点，应减速，但不倒车。
        long_error = torch.sum(agv_to_support * move_dir, dim=1)
        lat_error = torch.sum(agv_to_support * lateral_dir, dim=1)

        # 速度只做前向加减，不允许为了追支撑点而原地 180° 掉头。
        # Fix4：降低三车差速修正增益。
        # Fix3 中 long/lat 修正偏强，三台车顶着同一个 payload 时容易形成净 yaw 力矩，
        # 表现为 payload 被带走但持续自转。脚本验证阶段优先保证“同步平移”。
        v_action = base_speed + 0.28 * long_error
        v_action = torch.clamp(v_action, min=0.06, max=0.62)

        # 横向误差只做很小的航向偏置，避免三车为了追各自支撑点而扭动货物。
        steer = torch.clamp(0.45 * lat_error, min=-0.20, max=0.20).unsqueeze(-1)
        desired_vec = move_dir + steer * lateral_dir
        desired_vec = desired_vec / torch.linalg.norm(desired_vec, dim=1, keepdim=True).clamp_min(1e-6)

        agv_yaw = torch.atan2(agv_heading[:, 1], agv_heading[:, 0])
        desired_yaw = torch.atan2(desired_vec[:, 1], desired_vec[:, 0])
        yaw_error = torch.atan2(torch.sin(desired_yaw - agv_yaw), torch.cos(desired_yaw - agv_yaw))

        # 角速度限幅更小，避免三车顶着货物高速原地旋转。
        w_action = torch.clamp(0.85 * yaw_error, min=-0.35, max=0.35)

        # 航向误差大时也保留少量前进速度，避免纯原地转圈。
        heading_quality = torch.clamp(torch.cos(yaw_error), min=0.25, max=1.0)
        v_action = torch.clamp(v_action * heading_quality, min=0.04, max=0.62)

        all_actions.append(v_action)
        all_actions.append(w_action)

    actions = torch.stack(all_actions, dim=1)
    if actions.shape != (num_envs, 6):
        raise RuntimeError(f"Expected actions shape ({num_envs}, 6), got {actions.shape}")
    return torch.clamp(actions, -1.0, 1.0)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs_policy = obs["policy"] if isinstance(obs, dict) else obs
            actions = compute_scripted_actions(obs_policy)
            obs, reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            if step_count >= args_cli.max_steps:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
