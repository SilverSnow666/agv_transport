"""Static isolation checks for the V7.6-E1 reward-only variant."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/agv_transport/agv_transport/tasks/direct/agv_transport"
AGENT_DIR = TASK_DIR / "agents"
SMOOTH_CFG = TASK_DIR / "agv_level_residual_smooth_env_cfg.py"

EXPECTED_OVERRIDES = {
    "residual_board_angular_velocity_reference": 0.003,
    "residual_lift_velocity_reference": 0.005,
    "residual_board_angle_penalty_scale": 0.75,
    "residual_board_angular_velocity_penalty_scale": 0.10,
    "residual_action_penalty_scale": 0.060,
    "residual_action_rate_penalty_scale": 0.030,
    "residual_lift_velocity_penalty_scale": 0.060,
}


def _class_assignments(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignments = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = ast.literal_eval(node.value)
    return assignments


class SmoothRewardConfigTest(unittest.TestCase):
    def test_only_declared_reward_fields_are_overridden(self):
        assignments = _class_assignments(SMOOTH_CFG, "AgvLevelResidualSmoothEnvCfg")
        self.assertEqual(assignments, EXPECTED_OVERRIDES)

    def test_ppo_config_only_changes_experiment_identity(self):
        baseline = yaml.safe_load(
            (AGENT_DIR / "skrl_ppo_residual_cfg.yaml").read_text(encoding="utf-8")
        )
        smooth = yaml.safe_load(
            (AGENT_DIR / "skrl_ppo_residual_smooth_cfg.yaml").read_text(encoding="utf-8")
        )
        baseline_experiment = baseline["agent"]["experiment"]
        smooth_experiment = smooth["agent"]["experiment"]
        self.assertEqual(smooth_experiment["directory"], "agv_level_residual_smooth_direct")
        self.assertEqual(
            smooth_experiment["experiment_name"],
            "v7_6_e1_residual_ppo_smooth_reward",
        )
        smooth_experiment["directory"] = baseline_experiment["directory"]
        smooth_experiment["experiment_name"] = baseline_experiment["experiment_name"]
        self.assertEqual(smooth, baseline)

    def test_task_registration_uses_versioned_entries(self):
        source = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('id="Template-Agv-Level-Residual-Smooth-Direct-v0"', source)
        self.assertIn("agv_level_residual_smooth_env_cfg:AgvLevelResidualSmoothEnvCfg", source)
        self.assertIn("skrl_ppo_residual_smooth_cfg.yaml", source)


if __name__ == "__main__":
    unittest.main()
