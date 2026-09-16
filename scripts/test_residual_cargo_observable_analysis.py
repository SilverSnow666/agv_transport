from __future__ import annotations

import unittest

from scripts.analyze_residual_cargo_observable import aggregate, compare


def _row(policy_scale: float) -> dict[str, str]:
    row = {"phase": "p0", "case": "rough"}
    for baseline, policy in (
        ("roll_e", "roll_e1"),
        ("pitch_e", "pitch_e1"),
        ("board_rp_speed_e", "board_rp_speed_e1"),
        ("lift_speed_e", "lift_speed_e1"),
        ("cargo_slip_e", "cargo_slip_e1"),
        ("cargo_post_warmup_slip_e", "cargo_post_warmup_slip_e1"),
    ):
        row[baseline] = "2.0"
        row[policy] = str(policy_scale)
    for metric in (
        "residual_rms_mm",
        "residual_common_mode_rms_mm",
        "residual_differential_rms_mm",
        "residual_at_limit_fraction",
        "residual_saturation_fraction",
        "support_loss_fraction_proxy",
        "early_termination",
        "board_dropped",
        "board_tipped",
        "cargo_dropped",
        "cargo_tipped",
    ):
        row[metric] = "0.0"
    return row


class CargoObservableAnalysisTest(unittest.TestCase):
    def test_direct_comparison_and_aggregate(self) -> None:
        rows = compare([_row(1.5)], [_row(1.0)])
        self.assertAlmostEqual(rows[0]["roll_e1_vs_e_pct"], 25.0)
        self.assertAlmostEqual(rows[0]["roll_h2_vs_e_pct"], 50.0)
        self.assertAlmostEqual(rows[0]["roll_h2_vs_e1_pct"], 100.0 / 3.0)
        summary = aggregate(rows)
        all_row = next(row for row in summary if row["scope"] == "all")
        self.assertAlmostEqual(all_row["mean_roll_h2_vs_e1_pct"], 100.0 / 3.0)

    def test_mismatched_baseline_is_rejected(self) -> None:
        h2 = _row(1.0)
        h2["roll_e"] = "2.1"
        with self.assertRaisesRegex(RuntimeError, "baseline mismatch"):
            compare([_row(1.5)], [h2])


if __name__ == "__main__":
    unittest.main()
