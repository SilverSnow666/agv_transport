from __future__ import annotations

from isaaclab.utils import configclass

from .agv_level_carry_lift_env_cfg import AgvLevelCarryLiftVisualEnvCfg


@configclass
class AgvLevelResidualEnvCfg(AgvLevelCarryLiftVisualEnvCfg):
    """V7.6-A task where RL adds a small residual to the V7.5 controller."""

    action_space = 3
    observation_space = 33
    episode_length_s = 16.0

    enable_cargo = True
    enable_bumpy_support = True
    enable_virtual_friction_carry = True
    enable_front_position_guard = False
    enable_rear_lateral_guard = False
    leveling_controller_mode = "geometric_feedback"

    # The policy cannot change AGV motion. It only observes the resulting
    # disturbance and controls three bounded Lift residuals.
    residual_scripted_speed = 0.10
    residual_height_limit = 0.003

    # Independent per-environment reset randomization. The visual terrain mesh
    # is disabled because one cloned mesh cannot represent different analytical
    # terrain parameters in every vectorized environment.
    residual_domain_randomization = True
    residual_speed_range = (0.08, 0.18)
    residual_terrain_amplitude_range = (0.020, 0.050)
    residual_terrain_phase_x_range = (-3.141592653589793, 3.141592653589793)
    residual_terrain_phase_y_range = (-3.141592653589793, 3.141592653589793)
    residual_cargo_mass_range = (3.0, 8.0)
    residual_cargo_offset_x_range = (-0.030, 0.030)
    residual_cargo_offset_y_range = (-0.030, 0.030)
    enable_visual_terrain_mesh = False

    # Reward is deliberately restricted to Board/Cargo stability and control
    # effort. AGV progress is scripted and therefore is not rewarded.
    residual_alive_reward = 1.0
    residual_board_angle_penalty_scale = 120.0
    residual_board_angular_velocity_penalty_scale = 0.50
    residual_board_vertical_velocity_penalty_scale = 0.50
    residual_cargo_slip_penalty_scale = 12.0
    residual_cargo_velocity_penalty_scale = 1.0
    residual_cargo_tilt_penalty_scale = 20.0
    residual_cargo_angular_velocity_penalty_scale = 0.10
    residual_action_penalty_scale = 0.020
    residual_action_rate_penalty_scale = 0.010
    residual_lift_velocity_penalty_scale = 0.50
    residual_failure_penalty = 25.0
