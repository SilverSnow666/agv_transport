"""Read-only controller capture and pure NumPy time-series diagnostics.

Command k is issued at t[k] using pre-step state k, then held for one control
interval. Existing evaluation state fields describe the end of that interval.
"""

from __future__ import annotations

import numpy as np


def _array(tensor):
    return tensor.detach().cpu().numpy().copy()


def capture_pre_step(env):
    roll, pitch, _ = env._get_payload_rpy()
    geometric, axis_z = env._geometric_lift_targets()
    return {
        "board_roll_pre_rad": float(roll[0]),
        "board_pitch_pre_rad": float(pitch[0]),
        "geometric_target_m": _array(geometric[0]),
        "axis_z": _array(axis_z[0]),
        "actual_height_pre_m": _array(env.lift_height[0]),
        "terrain_roll_pre_rad": _array(env.agv_terrain_roll[0]),
        "terrain_pitch_pre_rad": _array(env.agv_terrain_pitch[0]),
        "support_relative_pre_m": _array(env._support_top_relative_heights()[0]),
        "previous_action": _array(env.actions[0]),
    }


def append_control_fields(row, env, pre, dt):
    row["command_time_s"] = row["time_s"] - dt
    row["control_dt_s"] = dt
    row["board_roll_pre_rad"] = pre["board_roll_pre_rad"]
    row["board_pitch_pre_rad"] = pre["board_pitch_pre_rad"]
    base = _array(env.base_leveling_target_height[0])
    residual = _array(env.last_residual_height[0])
    final = _array(env.lift_target_height[0])
    feedback = _array(env.leveling_feedback_height[0])
    action = _array(env.actions[0])
    actual = _array(env.lift_height[0])
    projected_feedback = feedback / pre["axis_z"]
    lo, hi = float(env.cfg.lift_min_height), float(env.cfg.lift_max_height)
    expected_base = np.clip(pre["geometric_target_m"] + projected_feedback, lo, hi)
    expected_final = np.clip(base + residual, lo, hi)
    base_error = float(np.max(np.abs(base - expected_base)))
    final_error = float(np.max(np.abs(final - expected_final)))
    if max(base_error, final_error) > 1.0e-6:
        raise RuntimeError(f"Control decomposition inconsistent: base={base_error}, final={final_error}")
    row["base_reconstruction_error_m"] = base_error
    row["final_reconstruction_error_m"] = final_error
    for i in range(3):
        j = i + 1
        fields = {
            "geometric_target_m": pre["geometric_target_m"][i],
            "base_target_m": base[i],
            "feedback_world_height_m": feedback[i],
            "feedback_axis_height_m": projected_feedback[i],
            "action": action[i],
            "action_rate_s_inv": (action[i] - pre["previous_action"][i]) / dt,
            "final_target_m": final[i],
            "actual_height_pre_m": pre["actual_height_pre_m"][i],
            "actual_interval_velocity_m_s": (actual[i] - pre["actual_height_pre_m"][i]) / dt,
            "terrain_roll_pre_rad": pre["terrain_roll_pre_rad"][i],
            "terrain_pitch_pre_rad": pre["terrain_pitch_pre_rad"][i],
            "support_relative_pre_m": pre["support_relative_pre_m"][i],
            "tracking_error_post_m": final[i] - actual[i],
            "base_clamp_correction_m": base[i] - pre["geometric_target_m"][i] - projected_feedback[i],
            "final_clamp_correction_m": final[i] - base[i] - residual[i],
        }
        row.update({f"lift{j}_{key}": float(value) for key, value in fields.items()})
        row[f"lift{j}_at_limit"] = int(abs(residual[i]) >= float(env.cfg.residual_height_limit) - 1e-7)


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def correlation(a, b):
    """Return None for a constant series: undefined is not zero correlation."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def plateau_events(values, times, dt, limit=0.003, tolerance=1e-7):
    """Contiguous same-polarity limit intervals; boundaries may be censored."""
    events = []
    start = None
    polarity = 0
    for k in range(len(values) + 1):
        sign = 0 if k == len(values) or abs(values[k]) < limit - tolerance else int(np.sign(values[k]))
        if sign != polarity:
            if polarity:
                events.append({
                    "start_s": float(times[start]), "end_s": float(times[k - 1] + dt),
                    "duration_s": float((k - start) * dt), "sign": polarity,
                    "samples": k - start, "left_censored": int(start == 0),
                    "right_censored": int(k == len(values)),
                })
            start = k if sign else None
            polarity = sign
    return events


def sign_switches(values, deadband):
    """Hysteretic sign transitions, carrying the last sign through the deadband."""
    previous = 0
    count = 0
    for value in values:
        sign = 1 if value > deadband else -1 if value < -deadband else 0
        if sign:
            count += int(previous != 0 and previous != sign)
            previous = sign
    return count


def analyze_channel(rows, channel, warmup_s=0.0, deadband_m=0.00005):
    selected = [r for r in rows if float(r["command_time_s"]) >= warmup_s - 1e-9]
    if len(selected) < 2:
        raise ValueError("Need at least two samples after warmup")
    def col(name):
        return np.array([float(r[name]) for r in selected])
    dt = float(selected[0]["control_dt_s"])
    times = col("command_time_s")
    if not np.allclose(np.diff(times), dt, atol=1e-8, rtol=0):
        raise ValueError("Nonuniform sampling or missing intervals")
    residual = col(f"residual{channel}_m")
    base = col(f"lift{channel}_base_target_m")
    final = col(f"lift{channel}_final_target_m")
    geo = col(f"lift{channel}_geometric_target_m")
    fb = col(f"lift{channel}_feedback_axis_height_m")
    # Exclude transitions from reset and across the chosen warmup boundary.
    rd, bd, fd = (np.diff(v) / dt for v in (residual, base, final))
    events = plateau_events(residual, times, dt)
    durations = [e["duration_s"] for e in events]
    run_time = len(selected) * dt
    limit_time = sum(durations)
    result = {
        "channel": channel, "warmup_s": warmup_s, "samples": len(selected),
        "observed_duration_s": run_time, "deadband_m": deadband_m,
        "residual_mean_mm": 1000 * float(np.mean(residual)),
        "residual_rms_mm": 1000 * rms(residual),
        "residual_rate_rms_mm_s": 1000 * rms(rd),
        "action_rate_rms_s_inv": rms(rd) / 0.003,
        "base_target_rate_rms_mm_s": 1000 * rms(bd),
        "geometric_target_rate_rms_mm_s": 1000 * rms(np.diff(geo) / dt),
        "feedback_target_rate_rms_mm_s": 1000 * rms(np.diff(fb) / dt),
        "final_target_rate_rms_mm_s": 1000 * rms(fd),
        "actual_interval_velocity_rms_mm_s": 1000 * rms(col(f"lift{channel}_actual_interval_velocity_m_s")),
        "reported_lift_velocity_rms_mm_s": 1000 * rms(col(f"lift{channel}_velocity_m_s")),
        "at_limit_fraction": limit_time / run_time,
        "limit_event_count": len(events),
        "limit_mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "limit_max_duration_s": max(durations, default=0.0),
        "limit_time_in_runs_ge_0_5s_fraction": sum(d for d in durations if d >= 0.5) / limit_time if limit_time else None,
        "limit_short_runs_le_2samples": sum(e["samples"] <= 2 for e in events),
        "sign_switch_count": sign_switches(residual, deadband_m),
        "sign_switches_per_s": sign_switches(residual, deadband_m) / run_time,
        "adjacent_limit_polarity_reversals": int(np.sum((residual[1:] * residual[:-1] < 0) & (np.abs(residual[1:]) >= 0.003 - 1e-7) & (np.abs(residual[:-1]) >= 0.003 - 1e-7))),
        "corr_residual_base": correlation(residual, base),
        "corr_residual_pre_roll": correlation(residual, col("board_roll_pre_rad")),
        "corr_residual_pre_pitch": correlation(residual, col("board_pitch_pre_rad")),
        "corr_base_rate_residual_rate": correlation(bd, rd),
        "base_rate_mean_square_mm2_s2": 1e6 * float(np.mean(bd**2)),
        "residual_rate_mean_square_mm2_s2": 1e6 * float(np.mean(rd**2)),
        "rate_cross_term_mm2_s2": 2e6 * float(np.mean(bd * rd)),
        "final_rate_mean_square_mm2_s2": 1e6 * float(np.mean(fd**2)),
        "max_final_clamp_correction_mm": 1000 * float(np.max(np.abs(col(f"lift{channel}_final_clamp_correction_m")))),
        "max_base_reconstruction_error_mm": 1000 * float(np.max(col("base_reconstruction_error_m"))),
        "max_final_reconstruction_error_mm": 1000 * float(np.max(col("final_reconstruction_error_m"))),
    }
    return result, events


def analyze_group(rows, warmup_s=0.0, deadband_m=0.00005):
    """Pooled three-channel rates plus common/differential residual motion."""
    selected = [r for r in rows if float(r["command_time_s"]) >= warmup_s - 1e-9]
    if len(selected) < 2:
        raise ValueError("Need at least two samples after warmup")
    dt = float(selected[0]["control_dt_s"])
    def matrix(pattern):
        return np.array([[float(r[pattern.format(j)]) for j in (1, 2, 3)] for r in selected])
    residual = matrix("residual{}_m")
    base = matrix("lift{}_base_target_m")
    final = matrix("lift{}_final_target_m")
    common = residual.mean(axis=1)
    times = np.array([float(r["command_time_s"]) for r in selected])
    events = [event for j in range(3) for event in plateau_events(residual[:, j], times, dt)]
    durations = [event["duration_s"] for event in events]
    limit_time = sum(durations)
    switches = [sign_switches(residual[:, j], deadband_m) for j in range(3)]
    heights_pre = matrix("lift{}_actual_height_pre_m")
    heights_post = matrix("lift{}_height_m")
    actual_velocity = matrix("lift{}_actual_interval_velocity_m_s")
    common_velocity = actual_velocity.mean(axis=1)
    mean_heights = np.r_[heights_pre[0].mean(), heights_post.mean(axis=1)]
    rd, bd, fd = (np.diff(v, axis=0) / dt for v in (residual, base, final))
    return {
        "warmup_s": warmup_s, "samples": len(selected),
        "at_limit_fraction": limit_time / (3 * len(selected) * dt),
        "limit_event_count": len(events),
        "limit_mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "limit_max_duration_s": max(durations, default=0.0),
        "limit_time_in_runs_ge_0_5s_fraction": sum(d for d in durations if d >= 0.5) / limit_time if limit_time else None,
        "sign_switches_min_per_channel": min(switches),
        "sign_switches_max_per_channel": max(switches),
        "residual_rms_mm": 1000 * rms(residual),
        "residual_common_mean_mm": 1000 * float(common.mean()),
        "residual_common_rms_mm": 1000 * rms(common),
        "residual_differential_rms_mm": 1000 * rms(residual - common[:, None]),
        "mean_actual_height_change_mm": 1000 * float(heights_post[-1].mean() - heights_pre[0].mean()),
        "mean_actual_height_range_mm": 1000 * float(np.ptp(mean_heights)),
        "actual_common_velocity_rms_mm_s": 1000 * rms(common_velocity),
        "actual_differential_velocity_rms_mm_s": 1000 * rms(actual_velocity - common_velocity[:, None]),
        "common_base_minus_actual_pre_rms_mm": 1000 * rms((base - heights_pre).mean(axis=1)),
        "corr_common_velocity_residual": correlation(common_velocity, common),
        "base_target_rate_rms_mm_s": 1000 * rms(bd),
        "residual_rate_rms_mm_s": 1000 * rms(rd),
        "final_target_rate_rms_mm_s": 1000 * rms(fd),
        "actual_interval_velocity_rms_mm_s": 1000 * rms(actual_velocity),
        "reported_lift_velocity_rms_mm_s": 1000 * rms(matrix("lift{}_velocity_m_s")),
        "base_rate_mean_square_mm2_s2": 1e6 * float(np.mean(bd**2)),
        "residual_rate_mean_square_mm2_s2": 1e6 * float(np.mean(rd**2)),
        "rate_cross_term_mm2_s2": 2e6 * float(np.mean(bd * rd)),
        "final_rate_mean_square_mm2_s2": 1e6 * float(np.mean(fd**2)),
    }
