# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-environment static validation for the telescopic Lift visual."""

import argparse
import asyncio
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate the telescopic Lift visual in a static scene.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Template-Agv-Level-Carry-Direct-v0")
parser.add_argument("--duration", type=float, default=None)
parser.add_argument(
    "--lift_height_mm",
    type=float,
    default=None,
    help="Optional common static Lift height in millimetres; defaults to 30 mm neutral.",
)
parser.add_argument("--screenshot_path", type=str, default=None)
parser.add_argument("--visual_mount_height_mm", type=float, default=None, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import agv_transport.tasks  # noqa: F401


def _capture_viewport(file_path: str) -> Path:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    output_path = Path(file_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = capture_viewport_to_file(get_active_viewport(), file_path=str(output_path))
    future = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
    while simulation_app.is_running() and not future.done():
        simulation_app.update()
    if not future.done() or not future.result() or not output_path.is_file():
        raise RuntimeError(f"Viewport capture failed: {output_path}")
    return output_path


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.enable_rear_lateral_guard = False
    env_cfg.enable_front_position_guard = False
    env_cfg.enable_bumpy_support = False
    env_cfg.enable_virtual_friction_carry = False
    env_cfg.enable_cargo = False
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")
    env.reset()
    raw_env = env.unwrapped

    if args_cli.visual_mount_height_mm is not None:
        print("[WARN] --visual_mount_height_mm is deprecated and ignored.")

    neutral = float(raw_env.cfg.lift_neutral_height)
    requested_height = neutral if args_cli.lift_height_mm is None else 0.001 * float(args_cli.lift_height_mm)
    lo = float(raw_env.cfg.lift_min_height)
    hi = float(raw_env.cfg.lift_max_height)
    if not lo <= requested_height <= hi:
        raise ValueError(
            f"Requested lift height {1000.0 * requested_height:.1f} mm is outside "
            f"[{1000.0 * lo:.1f}, {1000.0 * hi:.1f}] mm"
        )

    if hasattr(raw_env, "set_lift_stack_initial_height"):
        # Static selection moves the physical Lift and Board together. This
        # avoids the old misleading screenshot where the Lift moved but the
        # Board remained at the 30 mm neutral reset height.
        with torch.inference_mode():
            raw_env.set_lift_stack_initial_height(requested_height)
    else:
        raw_env.lift_target_height.fill_(requested_height)
        raw_env.lift_height.fill_(requested_height)
        raw_env.lift_velocity.zero_()
        raw_env._update_lift_poses()

    if args_cli.screenshot_path is not None:
        raw_env.sim.set_camera_view(eye=(1.45, -1.65, 0.25), target=(0.15, 0.0, 0.14))

    print(
        f"[CHECK] lift travel={1000.0 * lo:.1f}-{1000.0 * hi:.1f} mm, "
        f"neutral={1000.0 * neutral:.1f} mm, target={1000.0 * requested_height:.1f} mm"
    )
    if hasattr(raw_env, "lift_visual_column_lengths"):
        lengths = 1000.0 * raw_env.lift_visual_column_lengths()[0]
        print(
            f"[CHECK] visual=embedded base + two telescopic posts + moving head; "
            f"post lengths={lengths.tolist()} mm"
        )
    else:
        print("[WARN] telescopic Lift visual helper is unavailable; using legacy task visual")
    print(
        f"[CHECK] hidden physical Lift plate debug-visible="
        f"{bool(raw_env.cfg.debug_show_lift_collision_proxies)}"
    )

    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    elapsed = 0.0
    while simulation_app.is_running() and (
        args_cli.duration is None or elapsed < float(args_cli.duration)
    ):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=raw_env.device)
            if bool(getattr(raw_env.cfg, "forward_only_linear_speed", True)):
                actions[:, 0::2] = -1.0
            env.step(actions)
            elapsed += step_dt

            values = torch.cat(
                (
                    raw_env.lift_height.reshape(-1),
                    *[lift.data.root_pos_w.reshape(-1) for lift in raw_env.lifts],
                    raw_env.payload.data.root_pos_w.reshape(-1),
                )
            )
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(f"NaN/Inf detected at t={elapsed:.3f} s")

    heights = 1000.0 * raw_env.lift_height[0]
    board_z = 1000.0 * float(raw_env.payload.data.root_pos_w[0, 2])
    print(
        f"[RESULT] static: duration={elapsed:.3f} s, "
        f"lift heights={heights.tolist()} mm, Board Z={board_z:.3f} mm"
    )
    if args_cli.screenshot_path is not None:
        screenshot_path = _capture_viewport(args_cli.screenshot_path)
        print(f"[INFO] Viewport screenshot written to: {screenshot_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
