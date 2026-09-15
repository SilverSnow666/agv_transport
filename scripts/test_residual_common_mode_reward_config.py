"""Static isolation tests for the V7.6-E3 one-factor reward variant."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/agv_transport/agv_transport/tasks/direct/agv_transport"
AGENT_DIR = TASK_DIR / "agents"
E3_CFG = TASK_DIR / "agv_level_residual_common_mode_env_cfg.py"


def class_assignments(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    assignments = {}
    for node in cls.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        try:
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return assignments


class CommonModeRewardConfigTests(unittest.TestCase):
    def test_e3_overrides_only_common_mode_penalty(self):
        self.assertEqual(
            class_assignments(E3_CFG, "AgvLevelResidualCommonModeEnvCfg"),
            {"residual_common_mode_action_penalty_scale": .040},
        )

    def test_base_default_is_zero_and_e1_is_unchanged(self):
        base = class_assignments(TASK_DIR / "agv_level_residual_env_cfg.py", "AgvLevelResidualEnvCfg")
        e1 = class_assignments(TASK_DIR / "agv_level_residual_smooth_env_cfg.py", "AgvLevelResidualSmoothEnvCfg")
        self.assertEqual(base["residual_common_mode_action_penalty_scale"], 0.)
        self.assertNotIn("residual_common_mode_action_penalty_scale", e1)

    def test_ppo_config_only_changes_identity_from_e1(self):
        e1 = yaml.safe_load((AGENT_DIR / "skrl_ppo_residual_smooth_cfg.yaml").read_text(encoding="utf-8"))
        e3 = yaml.safe_load((AGENT_DIR / "skrl_ppo_residual_common_mode_cfg.yaml").read_text(encoding="utf-8"))
        self.assertEqual(e3["agent"]["experiment"]["directory"], "agv_level_residual_common_mode_direct")
        self.assertEqual(e3["agent"]["experiment"]["experiment_name"], "v7_6_e3_residual_ppo_common_mode_reward")
        e3["agent"]["experiment"] = e1["agent"]["experiment"]
        self.assertEqual(e3, e1)

    def test_registration_and_reward_diagnostics(self):
        registration = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
        reward = (TASK_DIR / "agv_level_residual_env.py").read_text(encoding="utf-8")
        self.assertIn('id="Template-Agv-Level-Residual-CommonMode-Direct-v0"', registration)
        self.assertIn("agv_level_residual_common_mode_env_cfg:", registration)
        self.assertIn("skrl_ppo_residual_common_mode_cfg.yaml", registration)
        for key in ("RewardTerms/common_mode_action", "Metrics/common_mode_action_rms", "Metrics/differential_action_rms"):
            self.assertIn(f'"{key}"', reward)
        self.assertIn("+ penalty_common_mode_action", reward)

    def test_cost_decomposition_identity(self):
        import torch
        actions = torch.tensor([[1., .5, -1.], [.2, .4, .6]])
        common = actions.mean(1)
        total = actions.square().mean(1)
        differential = (actions - common[:, None]).square().mean(1)
        torch.testing.assert_close(total, differential + common.square())


if __name__ == "__main__":
    unittest.main()
