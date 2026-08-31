# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--duration", type=float, default=None, help="Optional finite validation duration in seconds.")
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


def main():
    """Random actions agent with Isaac Lab environment."""
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
    neutral = float(raw_env.cfg.lift_neutral_height)
    print(
        f"[CHECK] lift travel={1000.0 * float(raw_env.cfg.lift_min_height):.1f}-"
        f"{1000.0 * float(raw_env.cfg.lift_max_height):.1f} mm, "
        f"neutral={1000.0 * neutral:.1f} mm"
    )
    step_dt = float(raw_env.cfg.sim.dt) * int(raw_env.cfg.decimation)
    elapsed = 0.0
    # simulate environment
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
    print(
        f"[RESULT] static: duration={elapsed:.3f} s, "
        f"lift heights={heights.tolist()} mm"
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
