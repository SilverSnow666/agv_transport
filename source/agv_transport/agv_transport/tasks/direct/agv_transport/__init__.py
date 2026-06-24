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
