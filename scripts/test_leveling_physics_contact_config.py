"""Static isolation checks for the contact-only Board/Cargo task."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/agv_transport/agv_transport/tasks/direct/agv_transport"
CFG = TASK_DIR / "agv_level_physics_contact_env_cfg.py"


def _literal_assignments(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignments: dict[str, object] = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return assignments


class PhysicsContactConfigTest(unittest.TestCase):
    def test_all_virtual_assistance_is_disabled(self) -> None:
        assignments = _literal_assignments(CFG, "AgvLevelPhysicsContactEnvCfg")
        self.assertFalse(assignments["enable_virtual_friction_carry"])
        for field in (
            "virtual_friction_coupling",
            "slip_correction_gain",
            "payload_vertical_damping",
            "payload_roll_pitch_damping",
            "payload_yaw_damping",
            "payload_yaw_alignment_gain",
            "payload_yaw_alignment_coupling",
            "max_payload_yaw_rate",
        ):
            self.assertEqual(assignments[field], 0.0, field)

    def test_real_contact_reporting_and_failure_visibility(self) -> None:
        assignments = _literal_assignments(CFG, "AgvLevelPhysicsContactEnvCfg")
        self.assertTrue(assignments["enable_payload_contact_sensor"])
        self.assertFalse(assignments["residual_terminate_on_failure"])
        self.assertFalse(assignments["residual_require_geometric_feedback"])
        self.assertEqual(assignments["lift_drive_mode"], "dynamic_velocity")
        source = CFG.read_text(encoding="utf-8")
        self.assertIn("activate_contact_sensors=True", source)
        self.assertEqual(source.count("kinematic_enabled=False"), 1)
        self.assertIn("lift1_cfg = _BASE_CFG.lift1_cfg.replace", source)
        self.assertIn("lift2_cfg = _BASE_CFG.lift2_cfg.replace", source)
        self.assertIn("lift3_cfg = _BASE_CFG.lift3_cfg.replace", source)

    def test_task_registration_is_versioned(self) -> None:
        source = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('id="Template-Agv-Level-PhysicsContact-Direct-v0"', source)
        self.assertIn("agv_level_physics_contact_env_cfg:", source)

    def test_sensor_reports_real_filtered_forces(self) -> None:
        source = (TASK_DIR / "agv_level_carry_env.py").read_text(encoding="utf-8")
        self.assertIn('self.scene.sensors["payload_contact"]', source)
        self.assertIn("forces = self.payload_contact_sensor.data.force_matrix_w", source)
        self.assertIn("torch.linalg.norm(forces[:, 0], dim=-1)", source)
        self.assertIn('drive_mode == "dynamic_velocity"', source)
        self.assertIn("lift.write_root_velocity_to_sim", source)

    def test_demo_aligns_visual_and_prescribed_terrain(self) -> None:
        source = (ROOT / "scripts/leveling_physics_contact_demo.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("cfg.bump_amplitude = args_cli.terrain_amplitude", source)
        self.assertIn("cfg.bump_phase_x = args_cli.phase_x", source)
        self.assertIn("cfg.bump_phase_y = args_cli.phase_y", source)
        self.assertIn(
            "cfg.visual_terrain_ground_z = -(args_cli.terrain_amplitude + 0.03)",
            source,
        )
        self.assertNotIn("cfg.visual_terrain_ground_z = 0.0", source)


if __name__ == "__main__":
    unittest.main()
