"""Static isolation checks for the V7.6-H2 Cargo-observable task."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/agv_transport/agv_transport/tasks/direct/agv_transport"
AGENT_DIR = TASK_DIR / "agents"
H2_CFG = TASK_DIR / "agv_level_residual_cargo_observable_env_cfg.py"


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


class CargoObservableConfigTests(unittest.TestCase):
    def test_only_observation_semantics_flag_is_overridden(self):
        assignments = _class_assignments(
            H2_CFG, "AgvLevelResidualCargoObservableEnvCfg"
        )
        self.assertEqual(assignments, {"residual_observe_cargo_slip_from_reset": True})

    def test_ppo_config_only_changes_experiment_identity(self):
        smooth = yaml.safe_load(
            (AGENT_DIR / "skrl_ppo_residual_smooth_cfg.yaml").read_text(encoding="utf-8")
        )
        cargo = yaml.safe_load(
            (AGENT_DIR / "skrl_ppo_residual_cargo_observable_cfg.yaml").read_text(
                encoding="utf-8"
            )
        )
        smooth_experiment = smooth["agent"]["experiment"]
        cargo_experiment = cargo["agent"]["experiment"]
        self.assertEqual(
            cargo_experiment["directory"], "agv_level_residual_cargo_observable_direct"
        )
        self.assertEqual(
            cargo_experiment["experiment_name"],
            "v7_6_h2_residual_ppo_cargo_observable",
        )
        cargo_experiment["directory"] = smooth_experiment["directory"]
        cargo_experiment["experiment_name"] = smooth_experiment["experiment_name"]
        self.assertEqual(cargo, smooth)

    def test_task_registration_uses_versioned_entries(self):
        source = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
        self.assertIn(
            'id="Template-Agv-Level-Residual-CargoObservable-Direct-v0"', source
        )
        self.assertIn(
            "agv_level_residual_cargo_observable_env_cfg:", source
        )
        self.assertIn("skrl_ppo_residual_cargo_observable_cfg.yaml", source)

    def test_environment_switches_only_the_two_cargo_xy_channels(self):
        source = (TASK_DIR / "agv_level_residual_env.py").read_text(encoding="utf-8")
        helper = source.split("def _cargo_xy_observation", 1)[1].split(
            "def _get_observations", 1
        )[0]
        self.assertIn("residual_observe_cargo_slip_from_reset", helper)
        self.assertIn("cargo_xy - self.cargo_initial_relative_xy", helper)
        observation = source.split("def _get_observations", 1)[1].split(
            "# Stability reward and termination", 1
        )[0]
        self.assertEqual(observation.count("self._cargo_xy_observation("), 1)


if __name__ == "__main__":
    unittest.main()
