from __future__ import annotations

from isaaclab.utils import configclass

from .agv_level_carry_env_cfg import AgvLevelCarryEnvCfg


@configclass
class AgvLevelCarryLiftVisualEnvCfg(AgvLevelCarryEnvCfg):
    """V7 visual-only telescopic Lift overlay.

    The hidden 0.28 x 0.28 x 0.04 m kinematic Lift plates remain the only
    physical Board supports. These parameters only rebuild what the user sees:
    a mostly embedded base, two telescopic guide posts, and a compact moving
    head whose top surface follows the physical support surface.
    """

    lift_visual_base_size = (0.17, 0.15, 0.024)
    lift_visual_head_size = (0.18, 0.16, 0.015)
    # Unit-height cuboid. Runtime Z scale equals the exposed post length.
    lift_visual_column_size = (0.026, 0.026, 1.0)
    lift_visual_column_y_offset = 0.045
    # Put the base and posts slightly inside the roof so there is no floating gap.
    lift_visual_base_exposed_height = 0.004
    lift_visual_column_embed_depth = 0.002
    lift_visual_min_column_height = 0.002

    lift_visual_base_color = (0.30, 0.32, 0.35)
    lift_visual_column_color = (0.62, 0.64, 0.67)
    lift_visual_head_color = (0.24, 0.26, 0.29)
