"""V7.6-E1 reward refinement for smooth residual Lift control."""

from __future__ import annotations

from isaaclab.utils import configclass

from .agv_level_residual_env_cfg import AgvLevelResidualEnvCfg


@configclass
class AgvLevelResidualSmoothEnvCfg(AgvLevelResidualEnvCfg):
    """Versioned reward variant; dynamics, observations and actions are unchanged.

    V7.6-D showed that the baseline PPO reduced Board attitude error by driving
    frequent +/-3 mm residuals and substantially increasing Lift velocity. E1
    keeps the same physical task and strengthens only dynamic/control-effort
    costs so its result remains attributable to reward design.
    """

    residual_board_angular_velocity_reference = 0.003
    residual_lift_velocity_reference = 0.005

    residual_board_angle_penalty_scale = 0.75
    residual_board_angular_velocity_penalty_scale = 0.10
    residual_action_penalty_scale = 0.060
    residual_action_rate_penalty_scale = 0.030
    residual_lift_velocity_penalty_scale = 0.060
