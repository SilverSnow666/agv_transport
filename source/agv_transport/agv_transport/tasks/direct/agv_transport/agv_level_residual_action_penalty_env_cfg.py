"""V7.6-E2 single-factor residual action-magnitude refinement."""

from __future__ import annotations

from isaaclab.utils import configclass

from .agv_level_residual_smooth_env_cfg import AgvLevelResidualSmoothEnvCfg


@configclass
class AgvLevelResidualActionPenaltyEnvCfg(AgvLevelResidualSmoothEnvCfg):
    """Increase only the E1 residual action penalty from 0.060 to 0.100."""

    residual_action_penalty_scale = 0.100
