"""Static and numerical tests for the V7.6-F multi-phase evaluator."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/residual_multiphase_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("residual_multiphase_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResidualMultiphaseEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_phase_matrix_is_unique_and_spread_across_quadrants(self):
        phases = self.module.PHASES
        self.assertEqual(len(phases), 4)
        self.assertEqual(len({row["label"] for row in phases}), 4)
        self.assertEqual(len({row["seed"] for row in phases}), 4)
        self.assertEqual(
            {(row["phase_x"] > 0, row["phase_y"] > 0) for row in phases},
            {(False, False), (False, True), (True, False), (True, True)},
        )

    def test_default_suite_and_candidate_are_explicit(self):
        self.assertEqual(self.module.E1_TASK, "Template-Agv-Level-Residual-Smooth-Direct-v0")
        self.assertEqual(self.module.ALL_CASES, (
            "nominal", "fast", "rough", "heavy", "offset", "combined"
        ))
        self.assertEqual(set(self.module.METRICS), {
            "roll", "pitch", "board_rp_speed", "lift_speed", "cargo_slip",
            "cargo_post_warmup_slip",
        })
        self.assertEqual(self.module.E1_CHECKPOINT.name, "best_agent.pt")

    def test_e1_checkpoint_training_seed_is_read_from_saved_config(self):
        if not self.module.E1_CHECKPOINT.is_file():
            self.skipTest("E1 checkpoint is not available")
        self.assertEqual(self.module._checkpoint_training_seed(self.module.E1_CHECKPOINT), 42)

    def test_numeric_validation_rejects_nonfinite_values(self):
        self.module._validate_numeric_rows(
            [{"case": "rough", "controller": "F_ppo", "value": "1.0"}],
            {"case", "controller"},
        )
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    self.module._validate_numeric_rows(
                        [{"case": "rough", "controller": "F_ppo", "value": value}],
                        {"case", "controller"},
                    )

    def test_post_warmup_slip_excludes_settling_path(self):
        rows = [
            {"controller": "E_zero", "time_s": "0.1", "cargo_cumulative_slip_m": "0.003"},
            {"controller": "E_zero", "time_s": "0.5", "cargo_cumulative_slip_m": "0.0032"},
            {"controller": "E_zero", "time_s": "1.0", "cargo_cumulative_slip_m": "0.0037"},
        ]
        self.assertAlmostEqual(
            self.module._post_warmup_cumulative_slip(rows, "E_zero", 0.5), 0.5
        )
        with self.assertRaises(RuntimeError):
            self.module._post_warmup_cumulative_slip(rows, "F_ppo", 0.5)

    def test_manifest_and_outputs_from_full_run_are_consistent(self):
        output = ROOT / "logs/v7_6_f/multiphase_stress"
        if not output.is_dir():
            self.skipTest("Full V7.6-F outputs are not available")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        with (output / "multiphase_runs.csv").open(newline="", encoding="utf-8") as stream:
            runs = list(csv.DictReader(stream))
        self.assertEqual(manifest["suite"], "stress")
        self.assertEqual(manifest["cases"], ["rough", "combined"])
        self.assertEqual(len(runs), len(manifest["phases"]) * len(manifest["cases"]))
        self.assertTrue(all(float(row["roll_improvement_pct"]) > 0 for row in runs))
        self.assertTrue(all(float(row["pitch_improvement_pct"]) > 0 for row in runs))

    def test_csv_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            rows = [{"a": 1, "b": "rough"}, {"a": 2, "b": "combined"}]
            self.module._write(path, rows)
            self.assertEqual(self.module._read(path), [
                {"a": "1", "b": "rough"},
                {"a": "2", "b": "combined"},
            ])


if __name__ == "__main__":
    unittest.main()
