# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-environment static validation for the embedded Lift visual."""

import argparse
import asyncio
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Validate the embedded Lift visual in a static scene.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="Template-Agv-Level-Carry-Direct-v0", help="Name of the task."
)
parser.add_argument("--duration", type=float, default=None, help="Optional finite validation duration in seconds.")
parser.add_argument(
    "--lift_height_mm",
    type=float,
    default=None,
    help="Optional common Lift target in millimetres; defaults to the 30 mm neutral height.",
)
parser.add_argument(
    "--screenshot_path", type=str, default=None, help="Optional viewport screenshot output path."
)
parser.add_argument(
    "--visual_mount_height_mm",
    type=float,
    default=None,
    help="Optional diagnostic override for the visual mount height above the AGV root.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import agv_transport.tasks  # noqa: F401


def _capture_viewport(file_path: str) -> Path:
    """Capture the active GUI viewport after the validation settles."""
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
    """Run the single-environment embedded-Lift validation."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # V7.0-A static visualization test:
    # disable old V6 motion-level feedback controllers
    env_cfg.enable_rear_lateral_guard = False
    env_cfg.enable_front_position_guard = False
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    raw_env = env.unwrapped
    if args_cli.visual_mount_height_mm is not None:
        raw_env._lift_visual_mount_height = 0.001 * float(args_cli.visual_mount_height_mm)
        raw_env._update_lift_poses()
    if args_cli.screenshot_path is not None:
        raw_env.sim.set_camera_view(eye=(1.25, -1.35, 0.42), target=(0.20, 0.0, 0.14))
    neutral = float(raw_env.cfg.lift_neutral_height)
    requested_height = (
        neutral if args_cli.lift_height_mm is None else 0.001 * float(args_cli.lift_height_mm)
    )
    if not float(raw_env.cfg.lift_min_height) <= requested_height <= float(raw_env.cfg.lift_max_height):
        raise ValueError(
            f"Requested lift height {1000.0 * requested_height:.1f} mm is outside "
            f"[{1000.0 * float(raw_env.cfg.lift_min_height):.1f}, "
            f"{1000.0 * float(raw_env.cfg.lift_max_height):.1f}] mm"
        )
    raw_env.lift_target_height.fill_(requested_height)
    if args_cli.lift_height_mm is not None:
        # Directly select a static visual pose.  The Board initial pose remains
        # untouched; dynamic actuator tracking is covered by the disturbance test.
        raw_env.lift_height.fill_(requested_height)
        raw_env.lift_velocity.zero_()
        raw_env._update_lift_poses()
    initial_visible = 1000.0 * raw_env._lift_actuator_external_lengths()[0]
    print(
        f"[CHECK] lift travel={1000.0 * float(raw_env.cfg.lift_min_height):.1f}-"
        f"{1000.0 * float(raw_env.cfg.lift_max_height):.1f} mm, "
        f"neutral={1000.0 * neutral:.1f} mm, "
        f"target={1000.0 * requested_height:.1f} mm"
    )
    print(
        f"[CHECK] visual mount={1000.0 * raw_env._lift_visual_mount_height:.2f} mm above AGV root, "
        f"initial roof-to-head rod lengths={initial_visible.tolist()} mm"
    )
    print(
        f"[CHECK] physical plate visible/debug="
        f"{bool(raw_env.cfg.debug_show_lift_collision_proxies)}, "
        f"visual head size={tuple(1000.0 * float(v) for v in raw_env.cfg.lift_head_visual_size)} mm, "
        f"rod diameter={2000.0 * float(raw_env.cfg.lift_actuator_visual_radius):.1f} mm"
    )
    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    elapsed = 0.0
    # simulate environment
    while simulation_app.is_running() and (
        args_cli.duration is None or elapsed < float(args_cli.duration)
    ):
        with torch.inference_mode():
            # V7.0-A static test:
            # action layout = [v1, w1, v2, w2, v3, w3]
            #
            # Because forward_only_linear_speed maps:
            # -1 -> 0 speed
            #  0 -> 50% speed
            # +1 -> 100% speed
            actions = torch.zeros(
                env.action_space.shape,
                device=env.unwrapped.device,
            )

            # Stop all three AGVs
            actions[:, 0] = -1.0
            actions[:, 2] = -1.0
            actions[:, 4] = -1.0

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
    visible_lengths = 1000.0 * raw_env._lift_actuator_external_lengths()[0]
    print(
        f"[RESULT] static: duration={elapsed:.3f} s, "
        f"lift heights={heights.tolist()} mm, "
        f"roof-to-head rod lengths={visible_lengths.tolist()} mm"
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
