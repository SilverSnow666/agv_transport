"""V7.6-H2 Cargo-observable residual Lift task."""

from __future__ import annotations

from isaaclab.utils import configclass

from .agv_level_residual_smooth_env_cfg import AgvLevelResidualSmoothEnvCfg


@configclass
class AgvLevelResidualCargoObservableEnvCfg(AgvLevelResidualSmoothEnvCfg):
    """E1 with Cargo XY expressed as displacement from its randomized reset pose.

    Reward, dynamics, action space, network inputs and domain randomization stay
    unchanged. Only the semantics of observation channels 26-27 are corrected
    so the policy sees the same slip vector that the Cargo position reward uses.
    """

    residual_observe_cargo_slip_from_reset = True
