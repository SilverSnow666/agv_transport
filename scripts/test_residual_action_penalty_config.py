"""Static isolation checks for the V7.6-E2 single-factor reward variant."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/agv_transport/agv_transport/tasks/direct/agv_transport"
AGENT_DIR = TASK_DIR / "agents"
E2_CFG = TASK_DIR / "agv_level_residual_action_penalty_env_cfg.py"


def _class_assignments(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignments = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = ast.literal_eval(node.value)
    return assignments


class ActionPenaltyConfigTest(unittest.TestCase):
    def test_e2_overrides_only_action_magnitude_penalty(self):
        assignments = _class_assignments(
            E2_CFG, "AgvLevelResidualActionPenaltyEnvCfg"
        )
        self.assertEqual(assignments, {"residual_action_penalty_scale": 0.100})

    def test_ppo_config_only_changes_experiment_identity(self):
        e1 = yaml.safe_load(
            (AGENT_DIR / "skrl_ppo_residual_smooth_cfg.yaml").read_text(
                encoding="utf-8"
            )
        )
        e2 = yaml.safe_load(
            (AGENT_DIR / "skrl_ppo_residual_action_penalty_cfg.yaml").read_text(
                encoding="utf-8"
            )
        )
        e1_experiment = e1["agent"]["experiment"]
        e2_experiment = e2["agent"]["experiment"]
        self.assertEqual(
            e2_experiment["directory"], "agv_level_residual_action_penalty_direct"
        )
        self.assertEqual(
            e2_experiment["experiment_name"],
            "v7_6_e2_residual_ppo_action_penalty",
        )
        e2_experiment["directory"] = e1_experiment["directory"]
        e2_experiment["experiment_name"] = e1_experiment["experiment_name"]
        self.assertEqual(e2, e1)

    def test_task_registration_uses_versioned_entries(self):
        source = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
        self.assertIn(
            'id="Template-Agv-Level-Residual-ActionPenalty-Direct-v0"', source
        )
        self.assertIn("agv_level_residual_action_penalty_env_cfg:", source)
        self.assertIn("AgvLevelResidualActionPenaltyEnvCfg", source)
        self.assertIn("skrl_ppo_residual_action_penalty_cfg.yaml", source)


if __name__ == "__main__":
    unittest.main()
