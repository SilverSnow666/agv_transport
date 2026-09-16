"""Tests for the V7.6-G independent-training-seed analysis."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/residual_multiseed_analysis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("residual_multiseed_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResidualMultiseedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_aggregate_reports_sample_statistics_and_positive_fraction(self):
        rows = []
        for seed, roll in ((42, 20.0), (43, -10.0), (44, 50.0)):
            row = {"training_seed": seed, "case": "rough"}
            for metric in self.module.METRICS:
                row[f"{metric}_improvement_pct"] = roll if metric == "roll" else 5.0
            rows.append(row)
        aggregate = self.module._aggregate(rows)
        roll = next(
            row for row in aggregate
            if row["training_seed"] == "all_seeds"
            and row["case"] == "all"
            and row["metric"] == "roll"
        )
        self.assertEqual(roll["samples"], 3)
        self.assertAlmostEqual(roll["mean_improvement_pct"], 20.0)
        self.assertAlmostEqual(roll["sample_std_pct"], 30.0)
        self.assertAlmostEqual(roll["positive_fraction"], 2.0 / 3.0)

    def test_saved_training_seeds_are_read_from_checkpoint_configs(self):
        run_dirs = (
            ROOT / "logs/v7_6_f/multiphase_stress",
            ROOT / "logs/v7_6_g/seed43_multiphase_stress",
            ROOT / "logs/v7_6_g/seed44_multiphase_stress",
        )
        if not all((directory / "manifest.json").is_file() for directory in run_dirs):
            self.skipTest("Full V7.6-G evaluation outputs are not available")
        seeds = []
        for directory in run_dirs:
            manifest, rows = self.module._load_run(directory)
            seeds.append(int(rows[0]["training_seed"]))
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                int(rows[0]["training_seed"]),
                self.module._saved_seed(Path(manifest["checkpoint"])),
            )
        self.assertEqual(seeds, [42, 43, 44])

    def test_full_aggregate_preserves_reproducible_benefits_and_costs(self):
        path = ROOT / "logs/v7_6_g/multiseed_aggregate/multiseed_runs.csv"
        if not path.is_file():
            self.skipTest("Full V7.6-G aggregate is not available")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        self.assertEqual({int(row["training_seed"]) for row in rows}, {42, 43, 44})
        self.assertTrue(all(float(row["roll_improvement_pct"]) > 0.0 for row in rows))
        self.assertTrue(all(float(row["pitch_improvement_pct"]) > 0.0 for row in rows))
        self.assertTrue(all(float(row["lift_speed_improvement_pct"]) < 0.0 for row in rows))
        p3_combined = [
            row for row in rows if row["phase"] == "p3" and row["case"] == "combined"
        ]
        self.assertEqual(len(p3_combined), 3)
        self.assertTrue(
            all(float(row["cargo_slip_improvement_pct"]) < 0.0 for row in p3_combined)
        )
        self.assertTrue(all(float(row[key]) == 0.0 for row in rows for key in self.module.SAFETY_KEYS))


if __name__ == "__main__":
    unittest.main()
