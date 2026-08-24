"""Metric helpers for the 3-AGV carry task.

This file does not change training. Import it from an existing IsaacLab/skrl
play or evaluation script and call `compute_carry_metrics(raw_env)` after
`env.step(actions)`.

Typical usage inside your play loop:

    from carry_metrics import unwrap_isaaclab_env, compute_carry_metrics, EpisodeMetricAccumulator

    raw_env = unwrap_isaaclab_env(env)
    acc = EpisodeMetricAccumulator()
    ...
    obs, rewards, terminated, truncated, infos = env.step(actions)
    metrics = compute_carry_metrics(raw_env)
    acc.update(metrics, terminated | truncated, rewards)
    if step % 20 == 0:
        print_metrics(step, metrics)

The helper only reads fields already present in AgvCarryEnv. It is meant for
post-training diagnosis: success rate, support stability, speed synchronization,
slip, attitude, and failure reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


def unwrap_isaaclab_env(env: Any) -> Any:
    """Best-effort unwrapping for Gymnasium / skrl / IsaacLab wrappers."""
    current = env
    visited = set()
    for _ in range(10):
        ident = id(current)
        if ident in visited:
            break
        visited.add(ident)
        if hasattr(current, "payload") and hasattr(current, "agvs"):
            return current
        for attr in ("unwrapped", "_env", "env"):
            if hasattr(current, attr):
                nxt = getattr(current, attr)
                if nxt is not None and nxt is not current:
                    current = nxt
                    break
        else:
            break
    return current


@torch.no_grad()
def compute_carry_metrics(raw_env: Any) -> Dict[str, torch.Tensor]:
    """Compute per-env diagnostic metrics from AgvCarryEnv.

    Returns tensors with shape [num_envs] unless noted. Call `.mean()` or use
    `EpisodeMetricAccumulator` for summaries.
    """
    env = unwrap_isaaclab_env(raw_env)
    device = env.device

    payload_xy = env.payload.data.root_pos_w[:, :2]
    target_xy = env._get_target_xy()
    goal_dist = torch.linalg.norm(target_xy - payload_xy, dim=1)
    move_dir, lateral_dir = env._compute_move_frame(payload_xy, target_xy)
    support_targets = env._compute_support_targets(payload_xy, move_dir, lateral_dir)
    formation_errors = env._compute_formation_errors(support_targets)
    contact_flags = env._compute_support_contact_flags(support_targets)
    contact_count = contact_flags.float().sum(dim=1)
    slip_error = env._compute_slip_error(payload_xy, move_dir, lateral_dir)
    support_margin = env._compute_support_polygon_margin(payload_xy)
    roll, pitch, _ = env._get_payload_rpy()

    payload_vel_xy = env.payload.data.root_lin_vel_w[:, :2]
    payload_speed_to_goal = torch.sum(payload_vel_xy * move_dir, dim=1)

    agv_speed_to_goal = torch.sum(env.agv_planar_vel * move_dir.unsqueeze(1), dim=2)
    agv_lateral_vel = torch.sum(env.agv_planar_vel * lateral_dir.unsqueeze(1), dim=2)
    rear_speed_diff = torch.abs(agv_speed_to_goal[:, 1] - agv_speed_to_goal[:, 2])
    all_speed_std = torch.std(agv_speed_to_goal, dim=1, unbiased=False)

    agv_xy = torch.stack([agv.data.root_pos_w[:, :2] for agv in env.agvs], dim=1)
    support_error_vec = agv_xy - support_targets
    lateral_error = torch.sum(support_error_vec * lateral_dir.unsqueeze(1), dim=2)

    # Reconstruct raw/applied forward-only linear commands when available.
    if hasattr(env, "_compute_linear_cmds"):
        raw_linear_cmds, applied_linear_cmds = env._compute_linear_cmds()
    else:
        linear_actions = env.actions[:, 0::2]
        if bool(getattr(env.cfg, "forward_only_linear_speed", True)):
            raw_linear_cmds = 0.5 * (linear_actions + 1.0)
        else:
            raw_linear_cmds = torch.clamp(linear_actions, min=0.0)
        applied_linear_cmds = raw_linear_cmds
    rear_cmd_diff_raw = torch.abs(raw_linear_cmds[:, 1] - raw_linear_cmds[:, 2])
    rear_cmd_diff_applied = torch.abs(applied_linear_cmds[:, 1] - applied_linear_cmds[:, 2])

    success = env._compute_success(goal_dist, roll, pitch, slip_error, contact_count, support_margin)
    dropped = env.payload.data.root_pos_w[:, 2] < env.cfg.payload_min_z
    tipped = torch.maximum(torch.abs(roll), torch.abs(pitch)) > env.cfg.tip_roll_pitch_threshold
    out_of_bounds = env._compute_out_of_bounds(payload_xy)
    support_lost = (
        (contact_count < float(env.cfg.critical_support_contacts))
        & (env.episode_length_buf > env.cfg.support_loss_grace_steps)
    )

    return {
        "success": success.float(),
        "goal_dist": goal_dist,
        "payload_speed_to_goal": payload_speed_to_goal,
        "contact_count": contact_count,
        "contact_agv1": contact_flags[:, 0].float(),
        "contact_agv2": contact_flags[:, 1].float(),
        "contact_agv3": contact_flags[:, 2].float(),
        "formation_error_mean": formation_errors.mean(dim=1),
        "formation_error_agv1": formation_errors[:, 0],
        "formation_error_agv2": formation_errors[:, 1],
        "formation_error_agv3": formation_errors[:, 2],
        "slip_error": slip_error,
        "support_margin": support_margin,
        "roll_abs": torch.abs(roll),
        "pitch_abs": torch.abs(pitch),
        "agv1_speed_to_goal": agv_speed_to_goal[:, 0],
        "agv2_speed_to_goal": agv_speed_to_goal[:, 1],
        "agv3_speed_to_goal": agv_speed_to_goal[:, 2],
        "rear_speed_diff": rear_speed_diff,
        "all_speed_std": all_speed_std,
        "agv1_lateral_vel": agv_lateral_vel[:, 0],
        "agv2_lateral_vel": agv_lateral_vel[:, 1],
        "agv3_lateral_vel": agv_lateral_vel[:, 2],
        "agv2_lateral_error": lateral_error[:, 1],
        "agv3_lateral_error": lateral_error[:, 2],
        "rear_cmd_diff_raw": rear_cmd_diff_raw,
        "rear_cmd_diff_applied": rear_cmd_diff_applied,
        "support_lost": support_lost.float(),
        "dropped": dropped.float(),
        "tipped": tipped.float(),
        "out_of_bounds": out_of_bounds.float(),
        "episode_length": env.episode_length_buf.float().to(device),
    }


def summarize_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Return mean scalars for printing or CSV writing."""
    return {k: float(v.detach().mean().cpu()) for k, v in metrics.items()}


def print_metrics(step: int, metrics: Dict[str, torch.Tensor], prefix: str = "") -> None:
    s = summarize_metrics(metrics)
    print(
        f"{prefix}step={step:05d} "
        f"goal={s['goal_dist']:.3f} "
        f"succ={s['success']:.2f} "
        f"contact={s['contact_count']:.2f} "
        f"slip={s['slip_error']:.3f} "
        f"rear_vdiff={s['rear_speed_diff']:.3f} "
        f"cmd_diff_raw={s['rear_cmd_diff_raw']:.3f} "
        f"cmd_diff_applied={s['rear_cmd_diff_applied']:.3f} "
        f"roll={s['roll_abs']:.3f} pitch={s['pitch_abs']:.3f} "
        f"support_lost={s['support_lost']:.2f}"
    )


@dataclass
class EpisodeMetricAccumulator:
    """Accumulates per-episode summaries across vectorized envs."""

    completed: List[Dict[str, float]] = field(default_factory=list)

    def update(
        self,
        metrics: Dict[str, torch.Tensor],
        done: torch.Tensor,
        rewards: torch.Tensor | None = None,
    ) -> None:
        done = done.bool().detach().cpu()
        if done.ndim > 1:
            done = done.view(-1)
        if not torch.any(done):
            return

        for env_id in torch.nonzero(done, as_tuple=False).view(-1).tolist():
            row = {k: float(v[env_id].detach().cpu()) for k, v in metrics.items() if v.ndim > 0 and v.shape[0] > env_id}
            if rewards is not None:
                r = rewards.detach().view(-1).cpu()
                if env_id < len(r):
                    row["last_reward"] = float(r[env_id])
            self.completed.append(row)

    def summary(self) -> Dict[str, float]:
        if not self.completed:
            return {}
        keys = sorted(self.completed[0].keys())
        out: Dict[str, float] = {}
        for key in keys:
            values = [row[key] for row in self.completed if key in row]
            if values:
                out[key + "_mean"] = sum(values) / len(values)
                out[key + "_max"] = max(values)
                out[key + "_min"] = min(values)
        out["num_episodes"] = float(len(self.completed))
        return out
