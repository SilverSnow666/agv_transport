"""V7.6-E3 targeted common-mode action regularization."""

from __future__ import annotations

from isaaclab.utils import configclass

from .agv_level_residual_smooth_env_cfg import AgvLevelResidualSmoothEnvCfg


@configclass
class AgvLevelResidualCommonModeEnvCfg(AgvLevelResidualSmoothEnvCfg):
    """Keep E1 unchanged and penalize only the residual common component."""

    residual_common_mode_action_penalty_scale = 0.040
