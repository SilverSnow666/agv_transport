"""Tests for the V7.6-H.1 Cargo objective audit."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_residual_cargo_tradeoff.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_residual_cargo_tradeoff", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResidualCargoTradeoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_observation_layout_matches_residual_environment(self):
        self.assertEqual(len(self.module.OBSERVATION_LAYOUT), 33)
        self.assertEqual(self.module.OBSERVATION_LAYOUT[26:30], (
            "cargo_relative_x",
            "cargo_relative_y",
            "cargo_relative_vx",
            "cargo_relative_vy",
        ))
        self.assertNotIn("cargo_initial_relative_x", self.module.OBSERVATION_LAYOUT)
        self.assertNotIn("cargo_slip_from_reset_x", self.module.OBSERVATION_LAYOUT)
        source = (
            ROOT
            / "source/agv_transport/agv_transport/tasks/direct/agv_transport"
            / "agv_level_residual_env.py"
        ).read_text(encoding="utf-8")
        helper = source.split("def _cargo_xy_observation", 1)[1].split(
            "def _get_observations", 1
        )[0]
        self.assertIn("cargo_relative_position[:, 0:2]", helper)
        self.assertIn("cargo_initial_relative_xy", helper)
        base_cfg = (
            ROOT
            / "source/agv_transport/agv_transport/tasks/direct/agv_transport"
            / "agv_level_residual_env_cfg.py"
        ).read_text(encoding="utf-8")
        self.assertIn("residual_observe_cargo_slip_from_reset = False", base_cfg)

    def test_pearson_handles_linear_and_constant_inputs(self):
        self.assertAlmostEqual(self.module._pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertEqual(self.module._pearson([1, 1, 1], [2, 4, 6]), 0.0)

    def test_reward_term_signs_and_action_reconstruction(self):
        env_cfg = {
            "residual_height_limit": 0.003,
            "residual_board_angle_reference": np.deg2rad(0.2),
            "residual_board_angular_velocity_reference": 0.003,
            "residual_board_vertical_velocity_reference": 0.01,
            "residual_cargo_slip_reference": 0.005,
            "residual_cargo_velocity_reference": 0.02,
            "residual_cargo_tilt_reference": np.deg2rad(1.0),
            "residual_cargo_angular_velocity_reference": 0.05,
            "residual_lift_velocity_reference": 0.005,
            **{f"residual_{key}_penalty_scale": 1.0 for key in self.module.REWARD_KEYS},
        }
        row = {
            "board_roll_deg": "0.2",
            "board_pitch_deg": "0",
            "board_roll_rate_rad_s": "0",
            "board_pitch_rate_rad_s": "0",
            "board_vz_m_s": "0",
            "cargo_slip_m": "0.005",
            "cargo_relative_speed_m_s": "0",
            "cargo_relative_roll_deg": "0",
            "cargo_relative_pitch_deg": "0",
            "cargo_relative_angular_speed_rad_s": "0",
            "residual1_m": "0.003",
            "residual2_m": "0",
            "residual3_m": "-0.003",
            "lift1_velocity_m_s": "0",
            "lift2_velocity_m_s": "0",
            "lift3_velocity_m_s": "0",
        }
        terms, action = self.module._reward_terms(row, np.zeros(3), env_cfg)
        self.assertAlmostEqual(terms["board_angle"], -1.0)
        self.assertAlmostEqual(terms["cargo_slip"], -1.0)
        self.assertAlmostEqual(terms["action"], -2.0 / 3.0)
        np.testing.assert_allclose(action, [1.0, 0.0, -1.0])

    def test_full_audit_contains_three_independent_seeds(self):
        path = ROOT / "logs/v7_6_h/cargo_objective_audit/cargo_tradeoff_summary.csv"
        if not path.is_file():
            self.skipTest("Full V7.6-H.1 audit is not available")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual({int(row["training_seed"]) for row in rows}, {42, 43, 44})
        self.assertEqual({row["controller"] for row in rows}, {"E_zero", "F_ppo"})
        self.assertTrue(
            all(float(row["reward_reconstruction_max_abs_error"]) < 1.0e-4 for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
