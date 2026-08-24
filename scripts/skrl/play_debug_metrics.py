# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Play a skrl checkpoint and diagnose per-AGV support loss on the carry task.

V5 keeps the V4 evaluation statistics and additionally reports:
- per-AGV contact flags and XY/Z contact sub-conditions;
- signed longitudinal/lateral error relative to each support target;
- an immediate CONTACT_CHANGE event whenever one AGV gains or loses support.

The script is read-only: it does not change actions, rewards, environment physics,
or checkpoint behavior.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl with AGV carry metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# AGV carry debug/evaluation options
parser.add_argument(
    "--debug_metrics",
    action="store_true",
    default=False,
    help="Print carry-task metrics during playback.",
)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=20,
    help="Print carry-task metrics every N environment steps when --debug_metrics is enabled.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop playback after N global steps. Use 0 to run until the simulator is closed or --eval_episodes is reached.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=0,
    help="Stop after this many completed episodes and print an evaluation summary. Use 0 to disable.",
)
parser.add_argument(
    "--success_reward_threshold",
    type=float,
    default=10.0,
    help="A terminal step with reward above this value is counted as success. Use a value below success_reward_scale.",
)
parser.add_argument(
    "--disable_contact_events",
    action="store_true",
    default=False,
    help="Disable immediate CONTACT_CHANGE prints. Periodic metrics are unaffected.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import agv_transport.tasks  # noqa: F401


def _as_1d_float_tensor(x, device=None) -> torch.Tensor:
    """Convert tensor-like x to a 1D float tensor without breaking torch/no_grad usage."""
    if x is None:
        if device is None:
            return torch.empty(0)
        return torch.empty(0, device=device)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device)
    return x.detach().float().reshape(-1)


def _as_1d_bool_tensor(x, device=None) -> torch.Tensor:
    if x is None:
        if device is None:
            return torch.empty(0, dtype=torch.bool)
        return torch.empty(0, dtype=torch.bool, device=device)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device)
    return x.detach().bool().reshape(-1)


def _mean_float(x) -> float:
    x = _as_1d_float_tensor(x)
    if x.numel() == 0:
        return float("nan")
    return float(x.mean().cpu())


def _safe_metric_tensor(raw_env, name: str, default: float = 0.0) -> torch.Tensor:
    """Read raw_env.last_* tensors safely; fall back to zeros with num_envs length."""
    device = raw_env.device if hasattr(raw_env, "device") else None
    n = int(getattr(raw_env, "num_envs", 1))
    value = getattr(raw_env, name, None)
    if value is None:
        return torch.full((n,), float(default), device=device)
    value = _as_1d_float_tensor(value, device=device)
    if value.numel() == n:
        return value
    if value.numel() == 1:
        return value.repeat(n)
    return torch.full((n,), float(default), device=device)


def _compute_carry_debug_tensors(raw_env) -> dict[str, torch.Tensor]:
    """Read AGV carry metrics from the unwrapped task env.

    Important: call this BEFORE env.step() when you want terminal-state metrics,
    because Isaac Lab/skrl wrappers may reset completed envs immediately after step.
    """
    with torch.no_grad():
        payload_xy = raw_env.payload.data.root_pos_w[:, :2]
        target_xy = raw_env._get_target_xy()
        goal_dist = torch.linalg.norm(target_xy - payload_xy, dim=1)

        move_dir, lateral_dir = raw_env._compute_move_frame(payload_xy, target_xy)
        support_targets = raw_env._compute_support_targets(payload_xy, move_dir, lateral_dir)
        contact_flags = raw_env._compute_support_contact_flags(support_targets)
        contact_count = contact_flags.float().sum(dim=1)
        formation_errors = raw_env._compute_formation_errors(support_targets)

        # Signed support-frame errors. Positive longitudinal error means the AGV is
        # ahead of its target along the payload-to-goal direction. Positive lateral
        # error means it is to the left of its target in the local support frame.
        agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in raw_env.agvs], dim=1)
        support_error_vec = agv_xy - support_targets
        longitudinal_errors = torch.sum(support_error_vec * move_dir.unsqueeze(1), dim=2)
        lateral_errors = torch.sum(support_error_vec * lateral_dir.unsqueeze(1), dim=2)

        slip_error = raw_env._compute_slip_error(payload_xy, move_dir, lateral_dir)
        support_margin = raw_env._compute_support_polygon_margin(payload_xy)
        roll, pitch, _ = raw_env._get_payload_rpy()
        payload_vertical_speed_abs = torch.abs(raw_env.payload.data.root_lin_vel_w[:, 2])
        if hasattr(raw_env, "_compute_support_z_gaps"):
            support_z_gaps = raw_env._compute_support_z_gaps()
        else:
            payload_bottom_z = raw_env.payload.data.root_pos_w[:, 2] - 0.5 * raw_env.cfg.payload_size[2]
            support_z_gaps = torch.stack(
                [
                    torch.abs(payload_bottom_z - (agv.data.root_pos_w[:, 2] + 0.5 * raw_env.cfg.agv_size[2]))
                    for agv in raw_env.agvs
                ],
                dim=1,
            )
        support_z_gap_mean = torch.mean(support_z_gaps, dim=1)
        support_z_gap_max = torch.max(support_z_gaps, dim=1).values
        xy_ok = formation_errors < float(raw_env.cfg.support_contact_xy_margin)
        z_ok = support_z_gaps < float(raw_env.cfg.support_contact_z_margin)

        agv_speed_to_goal = torch.sum(raw_env.agv_planar_vel * move_dir.unsqueeze(1), dim=2)
        rear_speed_diff = torch.abs(agv_speed_to_goal[:, 1] - agv_speed_to_goal[:, 2])
        all_speed_std = torch.std(agv_speed_to_goal, dim=1, unbiased=False)

        linear_actions = raw_env.actions[:, 0::2]
        if bool(getattr(raw_env.cfg, "forward_only_linear_speed", True)):
            raw_linear_cmds = 0.5 * (linear_actions + 1.0)
        else:
            raw_linear_cmds = torch.clamp(linear_actions, min=0.0)

        applied_linear_cmds = raw_linear_cmds.clone()
        if bool(getattr(raw_env.cfg, "tie_rear_linear_speed", False)):
            blend = float(getattr(raw_env.cfg, "tie_rear_linear_speed_blend", 1.0))
            rear_mean = 0.5 * (raw_linear_cmds[:, 1] + raw_linear_cmds[:, 2])
            applied_linear_cmds[:, 1] = blend * rear_mean + (1.0 - blend) * raw_linear_cmds[:, 1]
            applied_linear_cmds[:, 2] = blend * rear_mean + (1.0 - blend) * raw_linear_cmds[:, 2]

        if bool(getattr(raw_env.cfg, "limit_front_linear_speed", False)):
            max_front_lead = float(getattr(raw_env.cfg, "max_front_command_lead", 0.08))
            rear_applied_mean = 0.5 * (applied_linear_cmds[:, 1] + applied_linear_cmds[:, 2])
            front_ceiling = torch.clamp(rear_applied_mean + max_front_lead, min=0.0, max=1.0)
            applied_linear_cmds[:, 0] = torch.minimum(applied_linear_cmds[:, 0], front_ceiling)

        rear_cmd_diff_raw = torch.abs(raw_linear_cmds[:, 1] - raw_linear_cmds[:, 2])
        rear_cmd_diff_applied = torch.abs(applied_linear_cmds[:, 1] - applied_linear_cmds[:, 2])
        rear_speed_mean = 0.5 * (agv_speed_to_goal[:, 1] + agv_speed_to_goal[:, 2])
        front_speed_lead = agv_speed_to_goal[:, 0] - rear_speed_mean
        rear_raw_cmd_mean = 0.5 * (raw_linear_cmds[:, 1] + raw_linear_cmds[:, 2])
        rear_applied_cmd_mean = 0.5 * (applied_linear_cmds[:, 1] + applied_linear_cmds[:, 2])
        front_cmd_lead_raw = raw_linear_cmds[:, 0] - rear_raw_cmd_mean
        front_cmd_lead_applied = applied_linear_cmds[:, 0] - rear_applied_cmd_mean

        return {
            "goal": goal_dist.detach(),
            "contact": contact_count.detach(),
            "contact1": contact_flags[:, 0].float().detach(),
            "contact2": contact_flags[:, 1].float().detach(),
            "contact3": contact_flags[:, 2].float().detach(),
            "formation1": formation_errors[:, 0].detach(),
            "formation2": formation_errors[:, 1].detach(),
            "formation3": formation_errors[:, 2].detach(),
            "longitudinal1": longitudinal_errors[:, 0].detach(),
            "longitudinal2": longitudinal_errors[:, 1].detach(),
            "longitudinal3": longitudinal_errors[:, 2].detach(),
            "lateral1": lateral_errors[:, 0].detach(),
            "lateral2": lateral_errors[:, 1].detach(),
            "lateral3": lateral_errors[:, 2].detach(),
            "z_gap1": support_z_gaps[:, 0].detach(),
            "z_gap2": support_z_gaps[:, 1].detach(),
            "z_gap3": support_z_gaps[:, 2].detach(),
            "xy_ok1": xy_ok[:, 0].float().detach(),
            "xy_ok2": xy_ok[:, 1].float().detach(),
            "xy_ok3": xy_ok[:, 2].float().detach(),
            "z_ok1": z_ok[:, 0].float().detach(),
            "z_ok2": z_ok[:, 1].float().detach(),
            "z_ok3": z_ok[:, 2].float().detach(),
            "slip": slip_error.detach(),
            "support_margin": support_margin.detach(),
            "support_z_gap_mean": support_z_gap_mean.detach(),
            "support_z_gap_max": support_z_gap_max.detach(),
            "payload_vertical_speed_abs": payload_vertical_speed_abs.detach(),
            "rear_vdiff": rear_speed_diff.detach(),
            "all_vstd": all_speed_std.detach(),
            "cmd_diff_raw": rear_cmd_diff_raw.detach(),
            "cmd_diff_applied": rear_cmd_diff_applied.detach(),
            "front_vlead": front_speed_lead.detach(),
            "front_cmd_lead_raw": front_cmd_lead_raw.detach(),
            "front_cmd_lead_applied": front_cmd_lead_applied.detach(),
            "agv1_v": agv_speed_to_goal[:, 0].detach(),
            "agv2_v": agv_speed_to_goal[:, 1].detach(),
            "agv3_v": agv_speed_to_goal[:, 2].detach(),
            "roll": torch.abs(roll).detach(),
            "pitch": torch.abs(pitch).detach(),
            "support_lost": _safe_metric_tensor(raw_env, "last_support_lost"),
            "drop": _safe_metric_tensor(raw_env, "last_payload_dropped"),
            "tip": _safe_metric_tensor(raw_env, "last_payload_tipped"),
            "oob": _safe_metric_tensor(raw_env, "last_out_of_bounds"),
        }


def _with_step_outputs(metrics: dict[str, torch.Tensor], rewards, terminated, truncated, success_threshold: float):
    """Attach reward/done/success_by_reward to a pre-step metric dictionary."""
    device = next(iter(metrics.values())).device
    rewards_1d = _as_1d_float_tensor(rewards, device=device)
    term_1d = _as_1d_bool_tensor(terminated, device=device)
    trunc_1d = _as_1d_bool_tensor(truncated, device=device)

    n = metrics["goal"].numel()
    if rewards_1d.numel() == 1 and n > 1:
        rewards_1d = rewards_1d.repeat(n)
    if term_1d.numel() == 0:
        term_1d = torch.zeros(n, dtype=torch.bool, device=device)
    if trunc_1d.numel() == 0:
        trunc_1d = torch.zeros(n, dtype=torch.bool, device=device)

    done = term_1d | trunc_1d
    success_by_reward = done & (rewards_1d > success_threshold)

    out = dict(metrics)
    out["reward"] = rewards_1d.detach()
    out["done"] = done.detach().float()
    out["success_by_reward"] = success_by_reward.detach().float()
    out["terminated"] = term_1d.detach().float()
    out["truncated"] = trunc_1d.detach().float()
    return out


def _mean_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: _mean_float(v) for k, v in metrics.items()}


def _print_carry_debug_metrics(step: int, metrics: dict[str, torch.Tensor], *, terminal: bool = False) -> None:
    m = _mean_metrics(metrics)
    tag = "CARRY_TERMINAL" if terminal else "CARRY_DEBUG"
    print(
        f"[{tag} step={step:06d}] "
        f"reward={m.get('reward', float('nan')):.3f} "
        f"done={m.get('done', 0.0):.2f} "
        f"succR={m.get('success_by_reward', 0.0):.2f} "
        f"goal_pre={m['goal']:.3f} "
        f"contact={m['contact']:.2f}[{m['contact1']:.0f},{m['contact2']:.0f},{m['contact3']:.0f}] "
        f"xy_ok=[{m['xy_ok1']:.0f},{m['xy_ok2']:.0f},{m['xy_ok3']:.0f}] "
        f"z_ok=[{m['z_ok1']:.0f},{m['z_ok2']:.0f},{m['z_ok3']:.0f}] "
        f"form=({m['formation1']:.3f},{m['formation2']:.3f},{m['formation3']:.3f}) "
        f"long=({m['longitudinal1']:+.3f},{m['longitudinal2']:+.3f},{m['longitudinal3']:+.3f}) "
        f"lat=({m['lateral1']:+.3f},{m['lateral2']:+.3f},{m['lateral3']:+.3f}) "
        f"slip={m['slip']:.3f} "
        f"zgap={m.get('support_z_gap_mean', float('nan')):.3f}/{m.get('support_z_gap_max', float('nan')):.3f} "
        f"vz={m.get('payload_vertical_speed_abs', float('nan')):.3f} "
        f"rear_vdiff={m['rear_vdiff']:.4f} "
        f"cmd_raw={m['cmd_diff_raw']:.4f} "
        f"cmd_applied={m['cmd_diff_applied']:.4f} "
        f"front_lead(v/raw/app)={m['front_vlead']:.4f}/{m['front_cmd_lead_raw']:.4f}/{m['front_cmd_lead_applied']:.4f} "
        f"v=({m['agv1_v']:.3f},{m['agv2_v']:.3f},{m['agv3_v']:.3f}) "
        f"roll={m['roll']:.3f} "
        f"pitch={m['pitch']:.3f} "
        f"support_lost={m['support_lost']:.2f} "
        f"drop={m['drop']:.2f} "
        f"tip={m['tip']:.2f} "
        f"oob={m['oob']:.2f}"
    )


class ContactEventTracker:
    """Print an immediate diagnostic whenever an AGV contact flag changes."""

    def __init__(self, num_envs: int, device):
        self.num_envs = int(num_envs)
        self.device = device
        # -1 denotes an uninitialized/new episode state.
        self.previous = torch.full((self.num_envs, 3), -1, dtype=torch.int8, device=device)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=device)

    @staticmethod
    def _flag_triplet(metrics: dict[str, torch.Tensor], env_id: int) -> list[int]:
        return [
            int(metrics["contact1"][env_id].item() >= 0.5),
            int(metrics["contact2"][env_id].item() >= 0.5),
            int(metrics["contact3"][env_id].item() >= 0.5),
        ]

    def update(self, metrics: dict[str, torch.Tensor], global_step: int) -> None:
        done = metrics["done"].bool().reshape(-1)
        self.episode_steps += 1

        for env_id in range(self.num_envs):
            current = torch.tensor(self._flag_triplet(metrics, env_id), dtype=torch.int8, device=self.device)
            previous = self.previous[env_id]
            initialized = bool(torch.all(previous >= 0).item())

            if initialized and bool(torch.any(current != previous).item()):
                lost = [f"AGV{i + 1}" for i in range(3) if previous[i] == 1 and current[i] == 0]
                restored = [f"AGV{i + 1}" for i in range(3) if previous[i] == 0 and current[i] == 1]
                reasons = []
                for i in range(3):
                    if previous[i] == 1 and current[i] == 0:
                        xy_ok = bool(metrics[f"xy_ok{i + 1}"][env_id].item() >= 0.5)
                        z_ok = bool(metrics[f"z_ok{i + 1}"][env_id].item() >= 0.5)
                        if not xy_ok and not z_ok:
                            reason = "XY+Z"
                        elif not xy_ok:
                            reason = "XY"
                        elif not z_ok:
                            reason = "Z"
                        else:
                            reason = "unknown"
                        reasons.append(f"AGV{i + 1}:{reason}")

                print(
                    f"[CONTACT_CHANGE step={global_step:06d} env={env_id} "
                    f"ep_step={int(self.episode_steps[env_id].item()):04d}] "
                    f"flags={previous.tolist()}->{current.tolist()} "
                    f"lost={','.join(lost) if lost else '-'} "
                    f"restored={','.join(restored) if restored else '-'} "
                    f"reason={','.join(reasons) if reasons else '-'} "
                    f"form=({metrics['formation1'][env_id].item():.3f},"
                    f"{metrics['formation2'][env_id].item():.3f},"
                    f"{metrics['formation3'][env_id].item():.3f}) "
                    f"long=({metrics['longitudinal1'][env_id].item():+.3f},"
                    f"{metrics['longitudinal2'][env_id].item():+.3f},"
                    f"{metrics['longitudinal3'][env_id].item():+.3f}) "
                    f"lat=({metrics['lateral1'][env_id].item():+.3f},"
                    f"{metrics['lateral2'][env_id].item():+.3f},"
                    f"{metrics['lateral3'][env_id].item():+.3f}) "
                    f"zgap=({metrics['z_gap1'][env_id].item():.3f},"
                    f"{metrics['z_gap2'][env_id].item():.3f},"
                    f"{metrics['z_gap3'][env_id].item():.3f})"
                )

            self.previous[env_id] = current
            if env_id < done.numel() and bool(done[env_id].item()):
                self.previous[env_id].fill_(-1)
                self.episode_steps[env_id] = 0


class EpisodeStats:
    """Accumulate per-env episode statistics and print final evaluation summary."""

    def __init__(self, num_envs: int, device, success_threshold: float):
        self.num_envs = int(num_envs)
        self.device = device
        self.success_threshold = success_threshold
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.contact_sums = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.formation_sums = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.slip_sums = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.front_vlead_positive_sums = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.completed = []

    def update(self, step_metrics: dict[str, torch.Tensor], global_step: int) -> int:
        rewards = step_metrics["reward"].reshape(-1)
        done = step_metrics["done"].bool().reshape(-1)
        success = step_metrics["success_by_reward"].bool().reshape(-1)

        if rewards.numel() == 1 and self.num_envs > 1:
            rewards = rewards.repeat(self.num_envs)
        self.returns += rewards[: self.num_envs]
        self.lengths += 1
        self.contact_sums[:, 0] += step_metrics["contact1"][: self.num_envs]
        self.contact_sums[:, 1] += step_metrics["contact2"][: self.num_envs]
        self.contact_sums[:, 2] += step_metrics["contact3"][: self.num_envs]
        self.formation_sums[:, 0] += step_metrics["formation1"][: self.num_envs]
        self.formation_sums[:, 1] += step_metrics["formation2"][: self.num_envs]
        self.formation_sums[:, 2] += step_metrics["formation3"][: self.num_envs]
        self.slip_sums += step_metrics["slip"][: self.num_envs]
        self.front_vlead_positive_sums += torch.clamp(step_metrics["front_vlead"][: self.num_envs], min=0.0)

        done_indices = torch.nonzero(done[: self.num_envs], as_tuple=False).flatten()
        for env_id_tensor in done_indices:
            env_id = int(env_id_tensor.item())
            episode_length = max(int(self.lengths[env_id].item()), 1)
            ep = {
                "env_id": env_id,
                "global_step": int(global_step),
                "length": episode_length,
                "return": float(self.returns[env_id].item()),
                "agv1_contact_fraction": float(self.contact_sums[env_id, 0].item()) / episode_length,
                "agv2_contact_fraction": float(self.contact_sums[env_id, 1].item()) / episode_length,
                "agv3_contact_fraction": float(self.contact_sums[env_id, 2].item()) / episode_length,
                "episode_contact_count_mean": float(self.contact_sums[env_id].sum().item()) / episode_length,
                "agv1_formation_mean": float(self.formation_sums[env_id, 0].item()) / episode_length,
                "agv2_formation_mean": float(self.formation_sums[env_id, 1].item()) / episode_length,
                "agv3_formation_mean": float(self.formation_sums[env_id, 2].item()) / episode_length,
                "episode_slip_mean": float(self.slip_sums[env_id].item()) / episode_length,
                "front_vlead_positive_mean": float(self.front_vlead_positive_sums[env_id].item()) / episode_length,
                "success": bool(success[env_id].item()),
                "terminal_reward": float(rewards[env_id].item()),
                "terminal_goal_pre": float(step_metrics["goal"][env_id].item()),
                "terminal_contact": float(step_metrics["contact"][env_id].item()),
                "terminal_slip": float(step_metrics["slip"][env_id].item()),
                "terminal_z_gap_mean": float(step_metrics.get("support_z_gap_mean", torch.zeros_like(step_metrics["slip"]))[env_id].item()),
                "terminal_z_gap_max": float(step_metrics.get("support_z_gap_max", torch.zeros_like(step_metrics["slip"]))[env_id].item()),
                "terminal_vz_abs": float(step_metrics.get("payload_vertical_speed_abs", torch.zeros_like(step_metrics["slip"]))[env_id].item()),
                "terminal_rear_vdiff": float(step_metrics["rear_vdiff"][env_id].item()),
                "support_lost": float(step_metrics["support_lost"][env_id].item()),
                "drop": float(step_metrics["drop"][env_id].item()),
                "tip": float(step_metrics["tip"][env_id].item()),
                "oob": float(step_metrics["oob"][env_id].item()),
            }
            self.completed.append(ep)
            print(
                f"[EPISODE_END ep={len(self.completed):04d} env={env_id} global_step={global_step:06d}] "
                f"success_by_reward={int(ep['success'])} "
                f"length={ep['length']} "
                f"return={ep['return']:.3f} "
                f"contact_frac=({ep['agv1_contact_fraction']:.3f},{ep['agv2_contact_fraction']:.3f},{ep['agv3_contact_fraction']:.3f}) "
                f"contact_mean={ep['episode_contact_count_mean']:.3f} "
                f"slip_mean={ep['episode_slip_mean']:.3f} "
                f"front_vlead_pos_mean={ep['front_vlead_positive_mean']:.4f} "
                f"terminal_reward={ep['terminal_reward']:.3f} "
                f"terminal_goal_pre={ep['terminal_goal_pre']:.3f} "
                f"contact={ep['terminal_contact']:.2f} "
                f"slip={ep['terminal_slip']:.3f} "
                f"zgap={ep['terminal_z_gap_mean']:.3f}/{ep['terminal_z_gap_max']:.3f} "
                f"vz={ep['terminal_vz_abs']:.3f} "
                f"rear_vdiff={ep['terminal_rear_vdiff']:.4f} "
                f"support_lost={ep['support_lost']:.2f} "
                f"drop={ep['drop']:.2f} "
                f"tip={ep['tip']:.2f} "
                f"oob={ep['oob']:.2f}"
            )
            self.returns[env_id] = 0.0
            self.lengths[env_id] = 0
            self.contact_sums[env_id] = 0.0
            self.formation_sums[env_id] = 0.0
            self.slip_sums[env_id] = 0.0
            self.front_vlead_positive_sums[env_id] = 0.0
        return len(done_indices)

    def print_summary(self) -> None:
        n = len(self.completed)
        if n == 0:
            print("[EVAL_SUMMARY] episodes=0")
            return

        def avg(key: str) -> float:
            return sum(float(ep[key]) for ep in self.completed) / n

        success_rate = sum(1 for ep in self.completed if ep["success"]) / n
        print(
            "[EVAL_SUMMARY] "
            f"episodes={n} "
            f"success_rate={success_rate:.3f} "
            f"mean_length={avg('length'):.1f} "
            f"mean_return={avg('return'):.3f} "
            f"mean_contact_count={avg('episode_contact_count_mean'):.3f} "
            f"agv_contact_fraction=({avg('agv1_contact_fraction'):.3f},{avg('agv2_contact_fraction'):.3f},{avg('agv3_contact_fraction'):.3f}) "
            f"mean_formation=({avg('agv1_formation_mean'):.3f},{avg('agv2_formation_mean'):.3f},{avg('agv3_formation_mean'):.3f}) "
            f"mean_slip={avg('episode_slip_mean'):.3f} "
            f"mean_front_vlead_positive={avg('front_vlead_positive_mean'):.4f} "
            f"mean_terminal_goal_pre={avg('terminal_goal_pre'):.3f} "
            f"mean_terminal_contact={avg('terminal_contact'):.2f} "
            f"mean_terminal_slip={avg('terminal_slip'):.3f} "
            f"mean_terminal_z_gap={avg('terminal_z_gap_mean'):.3f} "
            f"mean_terminal_z_gap_max={avg('terminal_z_gap_max'):.3f} "
            f"mean_terminal_vz_abs={avg('terminal_vz_abs'):.3f} "
            f"mean_terminal_rear_vdiff={avg('terminal_rear_vdiff'):.4f} "
            f"support_lost_rate={avg('support_lost'):.3f} "
            f"drop_rate={avg('drop'):.3f} "
            f"tip_rate={avg('tip'):.3f} "
            f"oob_rate={avg('oob'):.3f}"
        )


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # keep a handle to the unwrapped task environment for read-only debug metrics before skrl wraps it
    metrics_enabled = bool(args_cli.debug_metrics or args_cli.eval_episodes > 0)
    metric_env = env.unwrapped if metrics_enabled else None
    episode_stats = None
    contact_event_tracker = None
    if metrics_enabled:
        num_metric_envs = int(getattr(metric_env, "num_envs", env_cfg.scene.num_envs))
        metric_device = getattr(metric_env, "device", torch.device("cpu"))
        episode_stats = EpisodeStats(num_metric_envs, metric_device, args_cli.success_reward_threshold)
        if not args_cli.disable_contact_events:
            contact_event_tracker = ContactEventTracker(num_metric_envs, metric_device)
        print(
            f"[INFO] Carry metrics enabled: interval={args_cli.debug_interval}, "
            f"max_steps={args_cli.max_steps if args_cli.max_steps > 0 else 'unlimited'}, "
            f"eval_episodes={args_cli.eval_episodes if args_cli.eval_episodes > 0 else 'disabled'}, "
            f"success_reward_threshold={args_cli.success_reward_threshold}"
        )

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # Read terminal-relevant metrics BEFORE env.step(). This avoids printing reset-state goal=3.2 on done.
        pre_step_metrics = None
        if metrics_enabled and metric_env is not None:
            pre_step_metrics = _compute_carry_debug_tensors(metric_env)

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, rewards, terminated, truncated, _ = env.step(actions)

        timestep += 1

        if metrics_enabled and pre_step_metrics is not None and episode_stats is not None:
            step_metrics = _with_step_outputs(
                pre_step_metrics,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                success_threshold=args_cli.success_reward_threshold,
            )
            done_tensor = step_metrics["done"].bool()
            any_done = bool(torch.any(done_tensor).item())
            should_print = bool(args_cli.debug_metrics) and timestep % max(args_cli.debug_interval, 1) == 0

            if contact_event_tracker is not None:
                contact_event_tracker.update(step_metrics, timestep)
            if should_print:
                _print_carry_debug_metrics(timestep, step_metrics, terminal=False)
            if any_done:
                _print_carry_debug_metrics(timestep, step_metrics, terminal=True)
            # Always update episode accumulators, not only on terminal steps.
            # The previous v2 script called update only when done=True, so length/return
            # were reported as only the final step. This keeps full episode length/return.
            episode_stats.update(step_metrics, timestep)
            if args_cli.eval_episodes > 0 and len(episode_stats.completed) >= args_cli.eval_episodes:
                break

        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if episode_stats is not None:
        episode_stats.print_summary()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
