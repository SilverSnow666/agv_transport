"""Numerical edge cases for diagnostic intervals, switching and decomposition."""
import unittest
import numpy as np
from scripts.residual_control_metrics import plateau_events, sign_switches, correlation, analyze_group


class ControlMetricsTests(unittest.TestCase):
    def test_plateaus_split_polarity_and_keep_censoring(self):
        values = np.array([0.003, 0.003, -0.003, 0, -0.003])
        events = plateau_events(values, np.arange(5) * 0.1, 0.1)
        self.assertEqual([e["samples"] for e in events], [2, 1, 1])
        self.assertEqual([e["sign"] for e in events], [1, -1, -1])
        self.assertEqual(events[0]["left_censored"], 1)
        self.assertEqual(events[-1]["right_censored"], 1)
        self.assertAlmostEqual(sum(e["duration_s"] for e in events), 0.4)

    def test_hysteresis_ignores_near_zero_noise(self):
        self.assertEqual(sign_switches([.001, 1e-6, -1e-6, .002, -.001], .00005), 1)
        self.assertEqual(sign_switches([.003, -.003, .003, -.003], .00005), 3)
        self.assertEqual(sign_switches([0, 0, 0], .00005), 0)

    def test_constant_correlation_is_undefined(self):
        self.assertIsNone(correlation(np.zeros(5), np.arange(5)))
        self.assertAlmostEqual(correlation(np.arange(5), -np.arange(5)), -1)

    def test_group_rate_identity_excludes_reset_transition(self):
        rows = []
        for k in range(4):
            row = {"command_time_s": k * 0.1, "control_dt_s": 0.1}
            for j in (1, 2, 3):
                row.update({
                    f"residual{j}_m": .002 + k * .0001,
                    f"lift{j}_base_target_m": .03 - k * .00005,
                    f"lift{j}_final_target_m": .032 + k * .00005,
                    f"lift{j}_actual_height_pre_m": .03 + k * .00005,
                    f"lift{j}_height_m": .03 + (k + 1) * .00005,
                    f"lift{j}_actual_interval_velocity_m_s": .0005,
                    f"lift{j}_velocity_m_s": .0004,
                })
            rows.append(row)
        result = analyze_group(rows, .1)
        self.assertEqual(result["samples"], 3)
        # Startup action is 2 mm, but intra-window residual rate is only 1 mm/s.
        self.assertAlmostEqual(result["residual_rate_rms_mm_s"], 1)
        self.assertAlmostEqual(result["final_target_rate_rms_mm_s"], .5)
        self.assertAlmostEqual(result["rate_cross_term_mm2_s2"], -1)
        total = sum(result[k] for k in ("base_rate_mean_square_mm2_s2", "residual_rate_mean_square_mm2_s2", "rate_cross_term_mm2_s2"))
        self.assertAlmostEqual(total, result["final_rate_mean_square_mm2_s2"])
        self.assertAlmostEqual(result["residual_differential_rms_mm"], 0)
        self.assertAlmostEqual(result["mean_actual_height_change_mm"], .15)

    def test_empty_warmup_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_group([], 1)


if __name__ == "__main__":
    unittest.main()
