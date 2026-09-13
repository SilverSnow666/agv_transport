"""Run with python -m unittest discover -s scripts/skrl -p test_residual_demo.py."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from residual_demo import configure_demo


class DemoConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            scene=SimpleNamespace(num_envs=128), viewer=SimpleNamespace(), max_agv_linear_speed=0.28,
            cargo_mass=4.0, bump_wavelength_x=1.4, bump_wavelength_y=1.1,
            residual_height_limit=0.003, leveling_controller_mode="geometric_feedback",
            enable_visual_terrain_mesh=False, residual_domain_randomization=True,
        )
        self.args = SimpleNamespace(
            task="Template-Agv-Level-Residual-Direct-v0", num_envs=None,
            ml_framework="torch", demo_speed=0.10, demo_amplitude=0.030,
            demo_phase_x=1.17, demo_phase_y=-2.03, zero_residual=False,
        )

    def test_fixed_ranges_match_visual_and_preserve_controller(self):
        original = deepcopy(self.cfg)
        configure_demo(self.cfg, self.args)
        self.assertEqual(self.cfg.scene.num_envs, 1)
        self.assertTrue(self.cfg.residual_domain_randomization)
        self.assertTrue(self.cfg.enable_visual_terrain_mesh)
        for base, sampled in (
            ("bump_amplitude", "residual_terrain_amplitude_range"),
            ("bump_phase_x", "residual_terrain_phase_x_range"),
            ("bump_phase_y", "residual_terrain_phase_y_range"),
            ("residual_scripted_speed", "residual_speed_range"),
        ):
            self.assertEqual(getattr(self.cfg, sampled), (getattr(self.cfg, base),) * 2)
        self.assertEqual(self.cfg.visual_terrain_ground_z, 0)
        self.assertEqual(self.cfg.visual_terrain_height_scale, 1)
        self.assertEqual(self.cfg.visual_terrain_z_offset, 0)
        self.assertEqual(self.cfg.residual_height_limit, original.residual_height_limit)
        self.assertEqual(self.cfg.leveling_controller_mode, original.leveling_controller_mode)
        self.assertEqual(original.scene.num_envs, 128)
        self.assertFalse(original.enable_visual_terrain_mesh)

    def test_invalid_options_rejected_before_mutation(self):
        for field, value in (
            ("num_envs", 2), ("task", "another-task"), ("ml_framework", "jax"),
            ("demo_speed", -0.1), ("demo_speed", 1.0),
            ("demo_amplitude", -0.001), ("demo_amplitude", 0.06),
            ("demo_phase_x", float("nan")), ("demo_phase_y", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                args = deepcopy(self.args)
                setattr(args, field, value)
                with self.assertRaises(ValueError):
                    configure_demo(self.cfg, args)
                self.assertEqual(self.cfg.scene.num_envs, 128)


if __name__ == "__main__":
    unittest.main()
