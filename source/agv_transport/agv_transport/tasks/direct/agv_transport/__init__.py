# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

# Existing V5.x pushing / transport task. Keep this unchanged.
gym.register(
    id="Template-Agv-Transport-Direct-v0",
    entry_point=f"{__name__}.agv_transport_env:AgvTransportEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.agv_transport_env_cfg:AgvTransportEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# New V6.0 carrying task: three AGVs support a payload from below and transport it.
gym.register(
    id="Template-Agv-Carry-Direct-v0",
    entry_point=f"{__name__}.agv_carry_env:AgvCarryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.agv_carry_env_cfg:AgvCarryEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_carry_cfg.yaml",
    },
)

# V7 active-leveling carrying task. The physical environment remains the
# validated AgvLevelCarryEnv model; the subclass only rebuilds the visible Lift
# as an embedded base + telescopic guides + moving head synchronized with the
# hidden physical Lift plates.
gym.register(
    id="Template-Agv-Level-Carry-Direct-v0",
    entry_point=f"{__name__}.agv_level_carry_lift_env:AgvLevelCarryLiftVisualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.agv_level_carry_lift_env_cfg:AgvLevelCarryLiftVisualEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_carry_cfg.yaml",
    },
)

# V7.6-A: independent 3D residual-Lift task. AGV translation is scripted
# inside the environment; the policy cannot alter vehicle motion.
gym.register(
    id="Template-Agv-Level-Residual-Direct-v0",
    entry_point=f"{__name__}.agv_level_residual_env:AgvLevelResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.agv_level_residual_env_cfg:AgvLevelResidualEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_residual_cfg.yaml",
    },
)

# V7.6-E1: versioned reward-only refinement. The environment class, physical
# model, observations, action range and randomization remain the V7.6 baseline.
gym.register(
    id="Template-Agv-Level-Residual-Smooth-Direct-v0",
    entry_point=f"{__name__}.agv_level_residual_env:AgvLevelResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.agv_level_residual_smooth_env_cfg:AgvLevelResidualSmoothEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_residual_smooth_cfg.yaml",
    },
)

# V7.6-E2: single-factor follow-up to E1. Only the residual action-magnitude
# penalty changes; all other reward, controller and physical settings inherit E1.
gym.register(
    id="Template-Agv-Level-Residual-ActionPenalty-Direct-v0",
    entry_point=f"{__name__}.agv_level_residual_env:AgvLevelResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.agv_level_residual_action_penalty_env_cfg:"
            "AgvLevelResidualActionPenaltyEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_residual_action_penalty_cfg.yaml"
        ),
    },
)

# V7.6-E3: return to E1 and add only a soft common-mode action penalty.
gym.register(
    id="Template-Agv-Level-Residual-CommonMode-Direct-v0",
    entry_point=f"{__name__}.agv_level_residual_env:AgvLevelResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.agv_level_residual_common_mode_env_cfg:"
            "AgvLevelResidualCommonModeEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_residual_common_mode_cfg.yaml"
        ),
    },
)

# V7.6-H2: return to E1 and change only Cargo XY observation semantics from
# absolute Board-frame position to displacement from the randomized reset pose.
gym.register(
    id="Template-Agv-Level-Residual-CargoObservable-Direct-v0",
    entry_point=f"{__name__}.agv_level_residual_env:AgvLevelResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.agv_level_residual_cargo_observable_env_cfg:"
            "AgvLevelResidualCargoObservableEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_residual_cargo_observable_cfg.yaml"
        ),
    },
)
