# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Four-case V7 board-stability ablation benchmark.

A: direct AGV-top support, no virtual carry, no leveling
B: Lift support, no virtual carry, no leveling
C: Lift support + virtual carry/damping, no leveling
D: Lift support + virtual carry/damping + geometric leveling

``support_count_proxy`` is analytical geometry, not a contact-sensor reading.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import traceback
import types

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="V7 board-stability ablation benchmark")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Template-Agv-Level-Carry-Direct-v0")
parser.add_argument("--case", choices=("A", "B", "C", "D", "all"), default="all")
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--settle_duration", type=float, default=1.0)
parser.add_argument("--target_speed", type=float, default=0.10)
parser.add_argument("--log_dir", type=str, default="logs/v7_ablation")
parser.add_argument("--screenshot_path", type=str, default=None)
parser.add_argument(
    "--force_front_support_loss_at",
    type=float,
    default=None,
    help="Diagnostic only: lower AGV1 by 1 m at this run time to force a two-support state.",
)
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


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    direct: bool
    vf: bool
    geometric: bool


CASES = {
    "A": CaseSpec("A", "direct_support", True, False, False),
    "B": CaseSpec("B", "lift_no_vf", False, False, False),
    "C": CaseSpec("C", "lift_vf", False, True, False),
    "D": CaseSpec("D", "lift_vf_geometric", False, True, True),
}

FIELDS = (
    "time", "case", "direct_support", "virtual_friction", "geometric_leveling",
    "cargo_enabled", "payload_vertical_damping", "payload_roll_pitch_damping",
    "payload_yaw_damping", "payload_yaw_alignment_coupling",
    "virtual_carry_active", "virtual_stabilization_active",
    "agv1_z", "agv2_z", "agv3_z",
    "agv1_roll_deg", "agv2_roll_deg", "agv3_roll_deg",
    "agv1_pitch_deg", "agv2_pitch_deg", "agv3_pitch_deg",
    "lift1_height", "lift2_height", "lift3_height",
    "support_top_z1", "support_top_z2", "support_top_z3", "support_height_rms",
    "support_count_proxy", "full_support_proxy", "critical_support_loss_proxy",
    "board_x", "board_y", "board_z", "board_roll_deg", "board_pitch_deg", "board_yaw_deg",
    "board_vx", "board_vy", "board_vz", "board_wx", "board_wy", "board_wz",
    "board_rp_angular_speed", "board_vertical_accel", "board_z_jerk", "board_forward_displacement",
)


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0


def capture_viewport(file_path, spec):
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    output_path = Path(file_path).expanduser().resolve()
    if args_cli.case == "all":
        output_path = output_path.with_name(
            f"{output_path.stem}_{spec.key}{output_path.suffix or '.png'}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = capture_viewport_to_file(get_active_viewport(), file_path=str(output_path))
    future = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
    while simulation_app.is_running() and not future.done():
        simulation_app.update()
    if not future.done() or not future.result() or not output_path.is_file():
        raise RuntimeError(f"Viewport capture failed: {output_path}")
    print(f"[CHECK] Screenshot={output_path}")


def actions_for(raw, speed):
    vmax = float(raw.cfg.max_agv_linear_speed)
    speed = min(max(float(speed), 0.0), vmax)
    cmd = speed / vmax
    a = 2.0 * cmd - 1.0 if bool(getattr(raw.cfg, "forward_only_linear_speed", True)) else cmd
    out = torch.zeros(raw.action_space.shape, device=raw.device)
    out[:, 0::2] = min(max(a, -1.0), 1.0)
    return out, speed, a


def stop_actions(raw):
    out = torch.zeros(raw.action_space.shape, device=raw.device)
    if bool(getattr(raw.cfg, "forward_only_linear_speed", True)):
        out[:, 0::2] = -1.0
    return out


def agv_tops(raw):
    h = 0.5 * float(raw.cfg.agv_size[2])
    tops = []
    for agv in raw.agvs:
        q = agv.data.root_quat_w[0].unsqueeze(0)
        tops.append(agv.data.root_pos_w[0] + h * raw._quat_rotate_z(q)[0])
    return torch.stack(tops)


def lift_tops(raw):
    h = 0.5 * float(raw.cfg.lift_plate_size[2])
    tops = []
    for lift in raw.lifts:
        q = lift.data.root_quat_w[0].unsqueeze(0)
        tops.append(lift.data.root_pos_w[0] + h * raw._quat_rotate_z(q)[0])
    return torch.stack(tops)


def park_lifts(self, env_ids=None):
    if env_ids is None:
        env_ids = self.payload._ALL_INDICES
    n = len(env_ids)
    for lift in self.lifts:
        pose = torch.zeros((n, 7), device=self.device)
        pose[:, 0:2] = self.scene.env_origins[env_ids, :2]
        pose[:, 2] = -10.0
        pose[:, 3] = 1.0
        lift.write_root_pose_to_sim(pose, env_ids=env_ids)
        lift.write_root_velocity_to_sim(torch.zeros((n, 6), device=self.device), env_ids=env_ids)
    self._set_native_lift_visual_visibility(False, env_ids)


def place_direct_board(raw):
    tops = agv_tops(raw)
    pose = raw.payload.data.root_pose_w.clone()
    pose[:, 2] = torch.max(tops[:, 2]) + 0.5 * float(raw.cfg.payload_size[2]) + float(raw.cfg.board_support_clearance)
    pose[:, 3:7] = 0.0; pose[:, 3] = 1.0
    raw.payload.write_root_pose_to_sim(pose)
    raw.payload.write_root_velocity_to_sim(torch.zeros((1, 6), device=raw.device))


def set_lift_target(raw, spec):
    neutral = float(raw.cfg.lift_neutral_height)
    lo, hi = float(raw.cfg.lift_min_height), float(raw.cfg.lift_max_height)
    if spec.geometric:
        top = lift_tops(raw)
        err = top[:, 2].mean() - top[:, 2]
        q = torch.stack([a.data.root_quat_w[0] for a in raw.agvs])
        axis_z = raw._quat_rotate_z(q)[:, 2].clamp_min(0.25)
        target = raw.lift_height[0] + err / axis_z
    else:
        target = torch.full_like(raw.lift_height[0], neutral)
    raw.lift_target_height[0] = torch.clamp(target, min=lo, max=hi)


def support_proxy(raw, spec, top):
    payload_xy = raw.payload.data.root_pos_w[:, :2]
    target_xy = raw._get_target_xy()
    move, lateral = raw._compute_move_frame(payload_xy, target_xy)
    targets = raw._compute_support_targets(payload_xy, move, lateral)
    xy_ok = raw._compute_formation_errors(targets)[0] < float(raw.cfg.support_contact_xy_margin)

    pos = raw.payload.data.root_pos_w[0]
    q = raw.payload.data.root_quat_w[0].unsqueeze(0)
    n = raw._quat_rotate_z(q)[0]
    c = pos - 0.5 * float(raw.cfg.payload_size[2]) * n
    nz = n[2].clamp_min(0.10)
    dx, dy = top[:, 0] - c[0], top[:, 1] - c[1]
    bottom_z = c[2] - (n[0] * dx + n[1] * dy) / nz
    z_ok = torch.abs(bottom_z - top[:, 2]) < float(raw.cfg.support_contact_z_margin)
    return xy_ok & z_ok


ASSIST_NAMES = (
    "virtual_friction_coupling", "slip_correction_gain", "payload_vertical_damping",
    "payload_roll_pitch_damping", "payload_yaw_damping",
    "payload_yaw_alignment_gain", "payload_yaw_alignment_coupling",
)


def make_env():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    cfg.seed = 0
    cfg.enable_front_position_guard = False
    cfg.enable_rear_lateral_guard = False
    cfg.enable_bumpy_support = True
    # The benchmark concerns the carrier Board only.  V7.2 Cargo must never
    # change its mass/contact response, even if a future task default enables it.
    cfg.enable_cargo = False
    # Build once and reset the same scene for every case.  Recreating Isaac Sim
    # physics scenes in one process is both expensive and less comparable.
    cfg.enable_virtual_friction_carry = True
    # Keep the benchmark running even when legacy Lift-based termination logic says support is lost.
    cfg.critical_support_contacts = 0.0
    cfg.payload_min_z = -100.0
    cfg.tip_roll_pitch_threshold = math.pi
    cfg.workspace_limit = max(float(getattr(cfg, "workspace_limit", 5.0)), 100.0)
    cfg.episode_length_s = max(float(cfg.episode_length_s), float(args_cli.duration) + float(args_cli.settle_duration) + 2.0)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    assert raw.cargo is None, "Cargo must be absent from the Board ablation"
    print(
        f"[GEOMETRY] AGV collider={tuple(cfg.agv_size)} m; "
        f"root flat-Z={cfg.agv_center_z:.6f} m; "
        f"support flat-Z={cfg.agv_top_z:.6f} m; original iwhub visual pose. "
        "Use this geometry consistently for all four cases; legacy 160 mm runs differ."
    )
    if args_cli.screenshot_path is not None:
        raw.sim.set_camera_view(eye=(1.25, -1.35, 0.42), target=(0.20, 0.0, 0.14))
    raw._ablation_lift_update = raw._update_lift_poses
    raw._ablation_assist_defaults = {name: float(getattr(raw.cfg, name)) for name in ASSIST_NAMES}
    return env


def configure_case(raw, spec):
    raw.cfg.enable_virtual_friction_carry = bool(spec.vf)
    for name, value in raw._ablation_assist_defaults.items():
        setattr(raw.cfg, name, value if spec.vf else 0.0)
    raw._set_native_lift_visual_visibility(not spec.direct)
    raw._update_lift_poses = (
        types.MethodType(park_lifts, raw) if spec.direct else raw._ablation_lift_update
    )
    assert bool(raw.cfg.enable_virtual_friction_carry) is spec.vf
    if not spec.vf:
        for name in ASSIST_NAMES:
            assert float(getattr(raw.cfg, name)) == 0.0, f"{name} leaked into Case {spec.key}"


def settle(env, spec):
    raw = env.unwrapped
    configure_case(raw, spec)
    dt = float(raw.cfg.sim.dt) * int(raw.cfg.decimation)
    elapsed = 0.0
    with torch.inference_mode():
        env.reset(seed=0)
        raw.base_z_disturbance.zero_()
        if spec.direct:
            place_direct_board(raw)
            raw._update_lift_poses()
        a = stop_actions(raw)
        while simulation_app.is_running() and elapsed < max(float(args_cli.settle_duration), 0.0):
            set_lift_target(raw, spec)
            env.step(a)
            elapsed += dt


def summarize(rows):
    return {
        "roll_rms": rms([r["board_roll_deg"] for r in rows]),
        "pitch_rms": rms([r["board_pitch_deg"] for r in rows]),
        "max_roll": max(abs(r["board_roll_deg"]) for r in rows),
        "max_pitch": max(abs(r["board_pitch_deg"]) for r in rows),
        "az_rms": rms([r["board_vertical_accel"] for r in rows]),
        "w_rms": rms([r["board_rp_angular_speed"] for r in rows]),
        "jerk_rms": rms([r["board_z_jerk"] for r in rows]),
        "support_rms_mm": 1000.0 * rms([r["support_height_rms"] for r in rows]),
        "loss_frac": sum(1.0 - r["full_support_proxy"] for r in rows) / len(rows),
        "critical_loss_frac": sum(r["critical_support_loss_proxy"] for r in rows) / len(rows),
        "mean_support": sum(r["support_count_proxy"] for r in rows) / len(rows),
        "z_range_mm": 1000.0 * (max(r["board_z"] for r in rows) - min(r["board_z"] for r in rows)),
        "forward_m": rows[-1]["board_forward_displacement"],
    }


def run(env, spec, log_dir):
    raw = env.unwrapped
    settle(env, spec)
    a, speed, lin_a = actions_for(raw, args_cli.target_speed)
    with torch.inference_mode():
        dt = float(raw.cfg.sim.dt) * int(raw.cfg.decimation)
        p0 = raw.payload.data.root_pos_w[0, :2].clone()
        target = raw._get_target_xy()[0]
        move = (target - p0) / torch.linalg.norm(target - p0).clamp_min(1e-6)
        prev_vz = float(raw.payload.data.root_lin_vel_w[0, 2]); prev_az = None
        rows = []; elapsed = 0.0
        print(
            f"[INFO] {spec.key} {spec.label}: direct={spec.direct}, vf={spec.vf}, "
            f"geometric={spec.geometric}, cargo={raw.cargo is not None}, "
            f"speed={speed:.3f}, action={lin_a:.6f}"
        )
        print(
            f"[ISOLATION] vertical={float(raw.cfg.payload_vertical_damping):.3f}, "
            f"roll_pitch={float(raw.cfg.payload_roll_pitch_damping):.3f}, "
            f"yaw={float(raw.cfg.payload_yaw_damping):.3f}, "
            f"yaw_alignment={float(raw.cfg.payload_yaw_alignment_coupling):.3f}"
        )
        while simulation_app.is_running() and elapsed < float(args_cli.duration):
            set_lift_target(raw, spec)
            raw.base_z_disturbance[0] = 0.0
            if (
                args_cli.force_front_support_loss_at is not None
                and elapsed >= float(args_cli.force_front_support_loss_at)
            ):
                raw.base_z_disturbance[0, 0] = -1.0
            env.step(a)
            elapsed += dt
            top = agv_tops(raw) if spec.direct else lift_tops(raw)
            z = top[:, 2]
            support_rms = float(torch.sqrt(torch.mean(torch.square(z - z.mean()))))
            count = float(support_proxy(raw, spec, top).float().sum())
            pos = raw.payload.data.root_pos_w[0]; lv = raw.payload.data.root_lin_vel_w[0]; av = raw.payload.data.root_ang_vel_w[0]
            roll, pitch, yaw = raw._get_payload_rpy()
            vz = float(lv[2]); az = (vz - prev_vz) / dt
            jerk = 0.0 if prev_az is None else (az - prev_az) / dt
            prev_vz, prev_az = vz, az
            row = dict(
                time=elapsed, case=spec.key, direct_support=float(spec.direct), virtual_friction=float(spec.vf), geometric_leveling=float(spec.geometric),
                cargo_enabled=float(raw.cargo is not None),
                payload_vertical_damping=float(raw.cfg.payload_vertical_damping),
                payload_roll_pitch_damping=float(raw.cfg.payload_roll_pitch_damping),
                payload_yaw_damping=float(raw.cfg.payload_yaw_damping),
                payload_yaw_alignment_coupling=float(raw.cfg.payload_yaw_alignment_coupling),
                virtual_carry_active=float(raw.last_virtual_carry_active[0]),
                virtual_stabilization_active=float(raw.last_virtual_stabilization_active[0]),
                agv1_z=float(raw.agvs[0].data.root_pos_w[0,2]), agv2_z=float(raw.agvs[1].data.root_pos_w[0,2]), agv3_z=float(raw.agvs[2].data.root_pos_w[0,2]),
                agv1_roll_deg=float(torch.rad2deg(raw.agv_terrain_roll[0,0])), agv2_roll_deg=float(torch.rad2deg(raw.agv_terrain_roll[0,1])), agv3_roll_deg=float(torch.rad2deg(raw.agv_terrain_roll[0,2])),
                agv1_pitch_deg=float(torch.rad2deg(raw.agv_terrain_pitch[0,0])), agv2_pitch_deg=float(torch.rad2deg(raw.agv_terrain_pitch[0,1])), agv3_pitch_deg=float(torch.rad2deg(raw.agv_terrain_pitch[0,2])),
                lift1_height=float(raw.lift_height[0,0]), lift2_height=float(raw.lift_height[0,1]), lift3_height=float(raw.lift_height[0,2]),
                support_top_z1=float(z[0]), support_top_z2=float(z[1]), support_top_z3=float(z[2]), support_height_rms=support_rms,
                support_count_proxy=count, full_support_proxy=float(count >= 3.0), critical_support_loss_proxy=float(count < 2.0),
                board_x=float(pos[0]), board_y=float(pos[1]), board_z=float(pos[2]), board_roll_deg=float(torch.rad2deg(roll[0])), board_pitch_deg=float(torch.rad2deg(pitch[0])), board_yaw_deg=float(torch.rad2deg(yaw[0])),
                board_vx=float(lv[0]), board_vy=float(lv[1]), board_vz=vz, board_wx=float(av[0]), board_wy=float(av[1]), board_wz=float(av[2]),
                board_rp_angular_speed=math.hypot(float(av[0]), float(av[1])), board_vertical_accel=az, board_z_jerk=jerk,
                board_forward_displacement=float(torch.dot(pos[:2] - p0, move)),
            )
            if not all(math.isfinite(float(v)) for k, v in row.items() if k != "case"):
                raise RuntimeError(f"NaN/Inf in Case {spec.key} at t={elapsed:.3f}s")
            rows.append(row)
        path = log_dir / f"case_{spec.key}_{spec.label}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        s = summarize(rows)
        print(
            f"[RESULT] {spec.key}: roll RMS={s['roll_rms']:.4f}deg, "
            f"pitch RMS={s['pitch_rms']:.4f}deg, max |roll|={s['max_roll']:.4f}deg, "
            f"max |pitch|={s['max_pitch']:.4f}deg"
        )
        print(
            f"[RESULT] {spec.key}: az RMS={s['az_rms']:.4f}m/s2, "
            f"RP w RMS={s['w_rms']:.4f}rad/s, jerk RMS={s['jerk_rms']:.4f}m/s3"
        )
        print(f"[RESULT] {spec.key}: support RMS={s['support_rms_mm']:.3f}mm, mean support={s['mean_support']:.3f}/3, loss={100*s['loss_frac']:.2f}%, critical loss={100*s['critical_loss_frac']:.2f}%, Z range={s['z_range_mm']:.3f}mm, forward={s['forward_m']:.3f}m")
        print(f"[CHECK] CSV={path}")
    if args_cli.screenshot_path is not None:
        capture_viewport(args_cli.screenshot_path, spec)
    return s


def improvement(a, b): return 100.0 * (a - b) / max(abs(a), 1e-12)


def compare(name, a, b):
    print(f"[ABLATION] {name}")
    for key, label in (
        ("roll_rms", "roll RMS"), ("pitch_rms", "pitch RMS"),
        ("max_roll", "max |roll|"), ("max_pitch", "max |pitch|"),
        ("az_rms", "vertical acceleration RMS"), ("w_rms", "RP angular-speed RMS"),
        ("jerk_rms", "Z jerk RMS"), ("z_range_mm", "Board Z range"),
        ("support_rms_mm", "support-height RMS"), ("loss_frac", "contact-loss proxy fraction"),
    ):
        delta = a[key] - b[key]
        print(
            f"[ABLATION] {label}: {a[key]:.6g} -> {b[key]:.6g}; "
            f"absolute reduction={delta:+.6g}, percent reduction={improvement(a[key], b[key]):+.2f}%"
        )
    support_delta = b["mean_support"] - a["mean_support"]
    support_pct = 100.0 * support_delta / max(abs(a["mean_support"]), 1.0e-12)
    print(
        f"[ABLATION] mean support proxy: {a['mean_support']:.3f} -> {b['mean_support']:.3f}; "
        f"absolute increase={support_delta:+.3f}, percent increase={support_pct:+.2f}%"
    )


def main():
    log_dir = Path(args_cli.log_dir).expanduser().resolve(); log_dir.mkdir(parents=True, exist_ok=True)
    keys = tuple(CASES) if args_cli.case == "all" else (args_cli.case,)
    env = make_env()
    try:
        out = {k: run(env, CASES[k], log_dir) for k in keys}
    finally:
        env.close()
    print("[NOTE] support/contact fields are analytical proxies, not PhysX contact-sensor measurements.")
    if args_cli.case == "all":
        compare("A -> B : Lift support/contact-topology contribution", out["A"], out["B"])
        compare("B -> C : virtual friction/damping contribution", out["B"], out["C"])
        compare("C -> D : geometric active-leveling contribution", out["C"], out["D"])
        if out["B"]["forward_m"] < 0.25 * float(args_cli.target_speed) * float(args_cli.duration):
            print("[NOTE] Case B transports the Board poorly without virtual carry assist. This is a valid result for the current kinematic-AGV model; no hidden assist was re-enabled.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); raise
    finally:
        simulation_app.close()
