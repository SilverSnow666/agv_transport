"""Visual contact-only Board/Cargo validation on prescribed rough terrain.

This is not a PPO evaluation. The three residual actions are fixed at zero, so
the Lift command is the geometric + Board-feedback controller only. The Board
and Cargo receive no virtual velocity coupling or artificial damping.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument(
    "--settle_duration",
    type=float,
    default=1.0,
    help="Hold the AGVs still for this long so the Board settles onto real contacts before transport.",
)
parser.add_argument("--target_speed", type=float, default=0.10)
parser.add_argument("--terrain_amplitude", type=float, default=0.050)
parser.add_argument("--phase_x", type=float, default=-2.68)
parser.add_argument("--phase_y", type=float, default=-0.49)
parser.add_argument(
    "--controller",
    choices=("neutral", "geometric", "geometric_feedback"),
    default="neutral",
    help=(
        "Lift controller. The default neutral mode makes the unassisted physical "
        "response visible; select geometric_feedback to demonstrate active leveling."
    ),
)
parser.add_argument(
    "--force_front_support_loss_at",
    type=float,
    default=None,
    help="Move front AGV1 2 m ahead at this transport time to verify physical tipping/fall.",
)
parser.add_argument(
    "--log_path",
    type=str,
    default="logs/v7_physics_contact/physics_contact_demo.csv",
)
parser.add_argument(
    "--real_time",
    action="store_true",
    help="Pace the simulation near wall-clock time for visual inspection.",
)
parser.add_argument(
    "--show_lift_collision_proxies",
    action="store_true",
    help=(
        "Show the actual Lift collision boxes. Their pose should follow the "
        "visible Lift heads and AGV terrain attitude."
    ),
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import agv_transport.tasks  # noqa: F401


TASK = "Template-Agv-Level-PhysicsContact-Direct-v0"
ASSIST_FIELDS = (
    "virtual_friction_coupling",
    "slip_correction_gain",
    "payload_vertical_damping",
    "payload_roll_pitch_damping",
    "payload_yaw_damping",
    "payload_yaw_alignment_gain",
    "payload_yaw_alignment_coupling",
    "max_payload_yaw_rate",
)
CSV_FIELDS = (
    "time_s",
    "front_support_forced_lost",
    "lift1_contact_force_n",
    "lift2_contact_force_n",
    "lift3_contact_force_n",
    "cargo_contact_force_n",
    "physics_lift_contact_count",
    "analytical_support_count_proxy",
    "board_x_m",
    "board_y_m",
    "board_z_m",
    "board_roll_deg",
    "board_pitch_deg",
    "board_yaw_deg",
    "board_vx_m_s",
    "board_vy_m_s",
    "board_vz_m_s",
    "board_wx_rad_s",
    "board_wy_rad_s",
    "board_wz_rad_s",
    "cargo_relative_x_m",
    "cargo_relative_y_m",
    "cargo_relative_z_m",
    "lift1_height_m",
    "lift2_height_m",
    "lift3_height_m",
    "agv1_roll_deg",
    "agv1_pitch_deg",
    "agv2_roll_deg",
    "agv2_pitch_deg",
    "agv3_roll_deg",
    "agv3_pitch_deg",
    "lift1_roll_deg",
    "lift1_pitch_deg",
    "lift2_roll_deg",
    "lift2_pitch_deg",
    "lift3_roll_deg",
    "lift3_pitch_deg",
    "virtual_carry_active",
    "virtual_stabilization_active",
)


def _quat_to_roll_pitch_deg(quat: torch.Tensor) -> tuple[float, float]:
    """Convert one Isaac Lab wxyz quaternion to roll/pitch in degrees."""
    w, x, y, z = (float(value) for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    return math.degrees(roll), math.degrees(pitch)


def _validate_args() -> None:
    values = (
        args_cli.duration,
        args_cli.settle_duration,
        args_cli.target_speed,
        args_cli.terrain_amplitude,
        args_cli.phase_x,
        args_cli.phase_y,
    )
    if not all(math.isfinite(value) for value in values):
        parser.error("All numeric arguments must be finite")
    if args_cli.duration <= 0.0:
        parser.error("--duration must be positive")
    if args_cli.settle_duration < 0.0:
        parser.error("--settle_duration must be non-negative")
    if not 0.0 <= args_cli.target_speed <= 0.35:
        parser.error("--target_speed must be within 0..0.35 m/s")
    if not 0.0 <= args_cli.terrain_amplitude <= 0.050:
        parser.error("--terrain_amplitude must be within 0..0.050 m")
    if (
        args_cli.force_front_support_loss_at is not None
        and not 0.0 <= args_cli.force_front_support_loss_at < args_cli.duration
    ):
        parser.error("--force_front_support_loss_at must be in [0, duration)")


def _make_env():
    cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    cfg.seed = 0
    cfg.leveling_controller_mode = args_cli.controller
    cfg.enable_bumpy_support = True
    cfg.enable_visual_terrain_mesh = True
    cfg.visual_terrain_height_scale = 1.0
    cfg.visual_terrain_z_offset = 0.0
    cfg.debug_show_lift_collision_proxies = args_cli.show_lift_collision_proxies
    # The visual mesh is generated before reset from the scalar cfg fields,
    # while the per-environment terrain uses the deterministic randomization
    # ranges below. Keep both descriptions identical.
    cfg.bump_amplitude = args_cli.terrain_amplitude
    cfg.bump_phase_x = args_cli.phase_x
    cfg.bump_phase_y = args_cli.phase_y
    # The infinite fallback plane must remain below the deepest visual valley;
    # placing it at z=0 hides the negative half of the sinusoidal mesh.
    cfg.visual_terrain_ground_z = -(args_cli.terrain_amplitude + 0.03)
    cfg.residual_domain_randomization = True
    cfg.residual_speed_range = (args_cli.target_speed,) * 2
    cfg.residual_terrain_amplitude_range = (args_cli.terrain_amplitude,) * 2
    cfg.residual_terrain_phase_x_range = (args_cli.phase_x,) * 2
    cfg.residual_terrain_phase_y_range = (args_cli.phase_y,) * 2
    cfg.residual_cargo_mass_range = (float(cfg.cargo_mass),) * 2
    cfg.residual_cargo_offset_x_range = (0.0, 0.0)
    cfg.residual_cargo_offset_y_range = (0.0, 0.0)
    cfg.episode_length_s = max(float(cfg.episode_length_s), args_cli.duration + 5.0)
    cfg.enable_virtual_friction_carry = False
    for field in ASSIST_FIELDS:
        setattr(cfg, field, 0.0)
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    raw.sim.set_camera_view(eye=(2.7, -3.2, 1.25), target=(0.4, 0.0, 0.20))
    return env


def _assert_isolation(raw) -> None:
    if bool(raw.cfg.enable_virtual_friction_carry):
        raise RuntimeError("Physical mode must disable virtual carry")
    leaking = {field: float(getattr(raw.cfg, field)) for field in ASSIST_FIELDS if float(getattr(raw.cfg, field))}
    if leaking:
        raise RuntimeError(f"Artificial assistance leaked into physical mode: {leaking}")
    if raw.payload_contact_sensor is None:
        raise RuntimeError("Physical mode requires the real Payload contact sensor")
    if str(raw.cfg.lift_drive_mode) != "dynamic_velocity":
        raise RuntimeError("Physical mode requires dynamic velocity-driven Lift plates")
    if tuple(float(value) for value in raw.cfg.lift_plate_size) != (0.18, 0.16, 0.015):
        raise RuntimeError("Physical mode must match Lift collision pads to visible heads")


def _row(raw, elapsed: float, forced: bool) -> dict[str, float]:
    contact_forces = raw.get_payload_filtered_contact_forces()[0]
    threshold = float(raw.cfg.physics_contact_force_threshold)
    roll, pitch, yaw = raw._get_payload_rpy()
    cargo_relative_position = raw._cargo_relative_state()[0][0]
    board_state = raw.payload.data.root_state_w[0]
    analytical_targets = raw._compute_support_targets(
        raw.payload.data.root_pos_w[:, :2],
        *raw._compute_move_frame(raw.payload.data.root_pos_w[:, :2], raw._get_target_xy()),
    )
    analytical_count = raw._compute_support_contact_flags(analytical_targets)[0].sum()
    agv_attitudes = [
        _quat_to_roll_pitch_deg(agv.data.root_state_w[0, 3:7]) for agv in raw.agvs
    ]
    lift_attitudes = [
        _quat_to_roll_pitch_deg(lift.data.root_state_w[0, 3:7]) for lift in raw.lifts
    ]
    return {
        "time_s": elapsed,
        "front_support_forced_lost": float(forced),
        "lift1_contact_force_n": float(contact_forces[0]),
        "lift2_contact_force_n": float(contact_forces[1]),
        "lift3_contact_force_n": float(contact_forces[2]),
        "cargo_contact_force_n": float(contact_forces[3]),
        "physics_lift_contact_count": float(torch.sum(contact_forces[:3] > threshold)),
        "analytical_support_count_proxy": float(analytical_count),
        "board_x_m": float(board_state[0]),
        "board_y_m": float(board_state[1]),
        "board_z_m": float(board_state[2]),
        "board_roll_deg": math.degrees(float(roll[0])),
        "board_pitch_deg": math.degrees(float(pitch[0])),
        "board_yaw_deg": math.degrees(float(yaw[0])),
        "board_vx_m_s": float(board_state[7]),
        "board_vy_m_s": float(board_state[8]),
        "board_vz_m_s": float(board_state[9]),
        "board_wx_rad_s": float(board_state[10]),
        "board_wy_rad_s": float(board_state[11]),
        "board_wz_rad_s": float(board_state[12]),
        "cargo_relative_x_m": float(cargo_relative_position[0]),
        "cargo_relative_y_m": float(cargo_relative_position[1]),
        "cargo_relative_z_m": float(cargo_relative_position[2]),
        "lift1_height_m": float(raw.lift_height[0, 0]),
        "lift2_height_m": float(raw.lift_height[0, 1]),
        "lift3_height_m": float(raw.lift_height[0, 2]),
        "agv1_roll_deg": agv_attitudes[0][0],
        "agv1_pitch_deg": agv_attitudes[0][1],
        "agv2_roll_deg": agv_attitudes[1][0],
        "agv2_pitch_deg": agv_attitudes[1][1],
        "agv3_roll_deg": agv_attitudes[2][0],
        "agv3_pitch_deg": agv_attitudes[2][1],
        "lift1_roll_deg": lift_attitudes[0][0],
        "lift1_pitch_deg": lift_attitudes[0][1],
        "lift2_roll_deg": lift_attitudes[1][0],
        "lift2_pitch_deg": lift_attitudes[1][1],
        "lift3_roll_deg": lift_attitudes[2][0],
        "lift3_pitch_deg": lift_attitudes[2][1],
        "virtual_carry_active": float(raw.last_virtual_carry_active[0]),
        "virtual_stabilization_active": float(raw.last_virtual_stabilization_active[0]),
    }


def main() -> None:
    _validate_args()
    env = _make_env()
    raw = env.unwrapped
    _assert_isolation(raw)
    output = Path(args_cli.log_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    dt = float(raw.cfg.sim.dt) * int(raw.cfg.decimation)
    actions = torch.zeros(raw.action_space.shape, device=raw.device)
    obs, _ = env.reset(seed=0)
    raw.randomized_target_speed.fill_(0.0)
    print(
        "[PHYSICS] contact-only mode: virtual carry/damping/alignment=OFF; "
        f"controller={args_cli.controller}, speed={args_cli.target_speed:.3f} m/s, "
        f"terrain={1000.0 * args_cli.terrain_amplitude:.1f} mm, "
        f"fallback ground z={raw.cfg.visual_terrain_ground_z:.3f} m"
    )
    print(
        "[GEOMETRY] free-contact Lift pads=180 x 160 x 15 mm, matched to visible heads; "
        "Board joints/attachments=NONE"
    )
    print(
        "[MATERIAL] dry wooden crate on plywood Board: "
        f"static/dynamic friction={raw.cfg.wood_contact_static_friction:.2f}/"
        f"{raw.cfg.wood_contact_dynamic_friction:.2f}"
    )
    elapsed = 0.0
    next_report = 0.0
    with torch.inference_mode():
        settle_elapsed = 0.0
        while simulation_app.is_running() and settle_elapsed < args_cli.settle_duration:
            wall_start = time.perf_counter()
            raw.base_z_disturbance[0, 0] = 0.0
            obs, _, _, _, _ = env.step(actions)
            settle_elapsed += dt
            if not torch.isfinite(obs["policy"]).all():
                raise RuntimeError("Non-finite observation while settling physical contacts")
            if args_cli.real_time:
                time.sleep(max(0.0, dt - (time.perf_counter() - wall_start)))
        raw.randomized_target_speed.fill_(float(args_cli.target_speed))
        settled_forces = raw.get_payload_filtered_contact_forces()[0, :3]
        settled_count = int(
            torch.sum(
                settled_forces > float(raw.cfg.physics_contact_force_threshold)
            )
        )
        print(
            f"[SETTLE] duration={settle_elapsed:.2f}s, real Lift contacts={settled_count}, "
            f"forces={[round(float(value), 2) for value in settled_forces]} N"
        )
        if args_cli.force_front_support_loss_at is not None:
            # Freeze the controller targets after settling so removing one
            # carrier cannot make the controller move the two remaining
            # supports in compensation.
            raw.cfg.leveling_controller_mode = "external"
        front_support_removed = False
        while simulation_app.is_running() and elapsed < args_cli.duration:
            wall_start = time.perf_counter()
            forced = (
                args_cli.force_front_support_loss_at is not None
                and elapsed >= args_cli.force_front_support_loss_at
            )
            raw.base_z_disturbance[0, 0] = 0.0
            if forced and not front_support_removed:
                # This is an explicit diagnostic fault injection: move AGV1
                # completely outside the Board footprint while leaving AGV2,
                # AGV3, the Board and Cargo untouched. Subsequent motion is
                # resolved only by gravity and PhysX contacts.
                front_state = raw.agvs[0].data.root_state_w.clone()
                front_state[:, 0] += 2.0
                front_state[:, 7:13] = 0.0
                raw.agvs[0].write_root_pose_to_sim(front_state[:, :7])
                raw.agvs[0].write_root_velocity_to_sim(front_state[:, 7:13])
                raw._update_lift_poses(force_pose=True)
                front_support_removed = True
                print(
                    f"[FAULT] t={elapsed:.2f}s: AGV1 moved 2.0 m ahead; "
                    "front support removed"
                )
            obs, _, _, _, _ = env.step(actions)
            elapsed += dt
            if not torch.isfinite(obs["policy"]).all():
                raise RuntimeError("Non-finite observation in physical mode")
            if bool(raw.last_virtual_carry_active[0]) or bool(raw.last_virtual_stabilization_active[0]):
                raise RuntimeError("Virtual assistance unexpectedly became active")
            row = _row(raw, elapsed, forced)
            rows.append(row)
            if elapsed >= next_report:
                print(
                    f"[STATE] t={elapsed:5.2f}s, contacts={int(row['physics_lift_contact_count'])}, "
                    f"roll/pitch={row['board_roll_deg']:+.2f}/{row['board_pitch_deg']:+.2f} deg, "
                    f"Board x={row['board_x_m']:+.3f} m, "
                    f"Cargo xy=({1000 * row['cargo_relative_x_m']:+.1f},"
                    f"{1000 * row['cargo_relative_y_m']:+.1f}) mm"
                )
                next_report += 1.0
            if args_cli.real_time:
                time.sleep(max(0.0, dt - (time.perf_counter() - wall_start)))

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    max_tilt = max(max(abs(row["board_roll_deg"]), abs(row["board_pitch_deg"])) for row in rows)
    min_contacts = min(row["physics_lift_contact_count"] for row in rows)
    board_forward = rows[-1]["board_x_m"] - rows[0]["board_x_m"]
    print(
        f"[RESULT] max Board tilt={max_tilt:.3f} deg, min real Lift contacts={min_contacts:.0f}, "
        f"Board dx={board_forward:.3f} m, CSV={output}"
    )
    print("[NOTE] AGV terrain pose remains prescribed; Board/Cargo dynamics are contact-only.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
