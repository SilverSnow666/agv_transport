# V7.6-E2.1: residual control decomposition

Simulation: 2026-09-14. Analysis and verification: 2026-09-15.
Starting HEAD: `a742773`, branch `v7.0-active-leveling`.

## Decision

Keep E1 best as the current candidate. Do not start E3 training merely to
reduce residual limit occupancy. Most observed limit time is sustained
single-polarity compensation, not frame-to-frame reversal between limits.
This does not establish that the compensation is necessary or optimal.

The additional E2 Lift motion has an important common-mode component:
the three lifts move up/down together more than with E1. E2's differential
Lift-velocity RMS is actually slightly lower than E1's in both cases. Thus
the net E2 increase in squared interval velocity is explained by the larger
common-mode component, not by faster residual changes alone.

No reward, policy, controller or physical parameters were changed here.
No new training was run.

## Protocol and provenance

- Compare E zero residual, E1 best and E2 best on rough and combined.
- Each policy has its own paired E run: four independent Isaac Sim processes,
  each running E then F. Eight runs, 720 control samples each, 5,760 rows total.
- One environment, 12 seconds, seed 137, phase `(1.17, -2.03)`, 60 Hz control,
  120 Hz physics. Deterministic policy mean and frozen normalization.
- Rough: 0.10 m/s, terrain amplitude 50 mm, Cargo 4 kg, no offset.
- Combined: 0.18 m/s, amplitude 50 mm, Cargo 8 kg, offset `(30, 0)` mm.
- Both E/F full initial-state signature differences are zero in each pair;
  cross-pair selected pre-state differences are also zero. Each percentage
  below uses its own paired E, not a historical baseline.
- E1 checkpoint: `logs/skrl/agv_level_residual_smooth_direct/2026-09-14_15-10-40_ppo_torch_v7_6_e1_residual_ppo_smooth_reward/checkpoints/best_agent.pt`.
- E2 checkpoint: `logs/skrl/agv_level_residual_action_penalty_direct/2026-09-14_16-21-35_ppo_torch_v7_6_e2_residual_ppo_action_penalty/checkpoints/best_agent.pt`.
- Results: `logs/v7_6_e2_1/decomposition_12s/`. Manifest records launch HEAD,
  checkpoint hashes and launch source hashes. Offline reducers/plots were
  subsequently extended without changing capture or simulation; their hashes
  are recorded separately in `analysis_source_sha256.json`.

## Timing and numerical definitions

`command_time_s` is the start of a control interval. Capture Board pre-state,
terrain attitude, support-relative heights, actual Lift heights and geometric
target before stepping. After stepping, read the command that was applied:

```text
base = clamp(geometric + filtered_feedback / axis_z)
final = clamp(base + residual)
```

Both reconstruction errors are exactly zero in the captured samples. The
largest apparent final clamp correction is 1.86e-6 mm (floating-point roundoff),
not meaningful clamping. Physical/final-target saturation is zero.

Existing `time_s` and physical state fields are at the END of the interval.
`lift*_actual_interval_velocity_m_s` is `(post_height - pre_height) / dt`.
Existing `lift*_velocity_m_s` is the actuator's final-substep reported velocity;
these are not interchangeable. Historical physical summaries keep the latter.

Command rate RMS uses differences within the selected window, excluding the
reset-to-first-command transition. Results are saved for the full window and
for command times >=1 s (660 intervals / 11 s). The diagnostic tables below use
the latter unless explicitly marked full-run. This removes startup effects
without deleting them from the raw data or full-window results.

Limit events are contiguous SAME-SIGN intervals with
`abs(residual) >= 0.003 - 1e-7 m`. Durations include each command's hold interval.
Boundary-truncated events retain left/right censoring flags; observed durations
are lower bounds when censored. Full and >=1 s event rows overlap and must not
be added together. The file has 63 event/window rows, not 63 distinct events.

Sign switches use +/-0.05 mm hysteresis, carrying the last nonzero sign through
the deadband. Switches are transitions, not oscillation cycles. Correlations
are channel-wise Pearson values; constant-series correlations are undefined
and written blank, not zero. Correlation alone does not establish causation.

## Sustained limit occupancy dominates

The percentage in the second column is the original full 12-second occupancy.
All remaining columns use the >=1 s diagnostic window. Long-run share is
weighted by time at the limit across all three channels, not event count.

| Case / policy | Full-run at limit | Mean event duration | Max event duration | Limit time in runs >=0.5 s | Sign switches per Lift in 11 s |
|---|---:|---:|---:|---:|---:|
| rough E1 | 20.65% | 1.239 s | 3.583 s | 94.84% | 3--4 |
| rough E2 | 22.73% | 1.617 s | 4.117 s | 94.23% | 2--5 |
| combined E1 | 25.00% | 0.818 s | 2.250 s | 87.96% | 9--12 |
| combined E2 | 25.46% | 0.963 s | 2.400 s | 88.85% | 2--5 |

No adjacent opposite-limit reversals occurred, including startup. Combined E1
does show localized small-scale command fluctuations and more zero crossings;
low overall switch counts must not be interpreted as absence of all jitter.
The evidence rejects rapid full-amplitude sign flipping as the main explanation
of the high occupancy in these runs. It does not prove the long bias is optimal.

## Base/residual interaction

Three-channel pooled RMS, mm/s; all command differences exclude startup.

| Case / policy | Base target rate | Residual rate | Final target rate | Actual interval velocity |
|---|---:|---:|---:|---:|
| rough E | 2.482 | 0 | 2.482 | 2.356 |
| rough E1 | 2.611 | 2.557 | 3.935 | 3.320 |
| rough E2 | 3.327 | 2.254 | 4.254 | 3.862 |
| combined E | 4.322 | 0 | 4.322 | 3.967 |
| combined E1 | 4.056 | 6.915 | 8.437 | 5.190 |
| combined E2 | 4.687 | 5.514 | 7.639 | 5.590 |

E2 residual-rate RMS is lower than E1 in both cases, yet actual Lift speed is
higher. In combined, even final-target rate RMS is lower with E2. Therefore
final-target derivative alone cannot predict actuator speed: tracking lag and
the distribution of common/differential motion matter.

With negligible clamping, the exported rate identity is
`mean(final_rate^2) = mean(base_rate^2) + mean(residual_rate^2) + 2mean(base_rate*residual_rate)`.
Cross terms for rough E1/E2 are +2.130/+1.940 mm^2/s^2, and for combined E1/E2
are +6.922/+5.985 mm^2/s^2. The two command components reinforce one another
on average; this is an algebraic observation, not proof of controller
competition. Per-channel correlations with base and Board pre-roll/pitch are
retained in `control_channel_summary.csv`.

## Common-mode mechanism

Define common velocity as the mean of three actual interval velocities and
differential velocity by subtracting that mean at every sample. This gives the
exact pooled mean-square identity `total^2 = common^2 + differential^2`.

| Case / policy | Common velocity RMS (mm/s) | Differential velocity RMS (mm/s) | Mean Lift-height range (mm) |
|---|---:|---:|---:|
| rough E | 0.002 | 2.356 | 0.008 |
| rough E1 | 1.492 | 2.967 | 9.988 |
| rough E2 | 2.514 | 2.932 | 14.659 |
| combined E | 0.003 | 3.967 | 0.010 |
| combined E1 | 1.558 | 4.951 | 6.142 |
| combined E2 | 2.743 | 4.870 | 9.381 |

These are local Lift-axis heights, not Board world Z. E2's differential speed
is slightly smaller, so its larger common-mode speed accounts for more than
the entire net increase in squared total velocity relative to E1.

Code inspection supplies a mechanism beyond correlation:
`_geometric_lift_targets()` adds mean-support-height error to CURRENT Lift
height. The feedback correction is zero-mean in world height before axis
projection. With nearly vertical axes and no clamps, mean(base) is therefore
approximately mean(actual height), not a fixed 30 mm reference. The three RL
actions are not constrained to zero mean. A persistent mean residual drives
common Lift motion through the position servo (`lift_position_kp=5/s`).

For two unclipped 1/120 s servo substeps, the interval gain from a held target
gap to mean velocity is approximately 4.8958/s. Thus even a sub-millimeter
common residual can drive several mm/s of shared Lift motion. This is a
moving-reference effect; +/-3 mm bounds instantaneous residual, not cumulative
departure from the initial mean height (absolute Lift limits still apply).

Measured common base-minus-pre-actual gap RMS is only 0.0027--0.0029 mm, whereas
common residual RMS is 0.304/0.513 mm for rough E1/E2 and 0.317/0.559 mm for
combined E1/E2. Common actual velocity versus common residual correlation is
0.99998 or higher. These data are consistent with the inspected mechanism.
They do not establish that removing common motion preserves cargo stability.

## Full-run physical cross-check

Positive improvement means lower metric than the paired E. Lift column is
explicitly an INCREASE, using the legacy final-substep velocity metric.

| Case / policy | Roll improvement | Pitch improvement | Board RP speed improvement | Lift speed increase | Cargo cumulative slip |
|---|---:|---:|---:|---:|---:|
| rough E1 | 54.04% | 26.39% | 24.07% | 40.65% | 1.622 mm |
| rough E2 | 48.87% | 14.33% | 27.62% | 69.62% | 1.891 mm |
| combined E1 | 51.87% | 33.67% | 42.85% | 31.73% | 3.145 mm |
| combined E2 | 48.25% | 24.30% | 59.27% | 43.76% | 3.504 mm |

Paired E Cargo slip: rough 1.603 mm, combined 3.259 mm. No run has early
termination, Board/Cargo drop/tip, support-proxy loss or final Lift target
saturation. Support count is 3 throughout, but all contact/support measurements
are analytical proxies, NOT PhysX contact sensors.

## Validation, files and reproduction

- 0.3 s rough diagnostic smoke completed for both policies and their E pairs.
- Repeated the legacy evaluator without the diagnostic flag at 0.3 s: all
  shared numeric trajectory fields matched diagnostic E1 rough exactly (36 rows).
- Full two-case matrix passed finite-value, 720-sample and initial-state checks.
- Five diagnostic numerical tests and six E1/E2 configuration tests passed.
- Python `py_compile` and `git diff --check` passed. Ruff is not installed.
- Both per-Lift figures and common-mode figure were rendered and visually checked.
- Runtime emitted existing Fabric point-instancer prototype warnings; no crash.

New files: `scripts/residual_control_decomposition.py`,
`scripts/residual_control_metrics.py`, `scripts/test_residual_control_metrics.py`,
and this record. The existing checkpoint evaluator only adds opt-in capture;
its default path is unchanged. Unrelated `scriptslift_motion_demo.py` is untouched.

Run from the repository root in `env_isaaclab`, with the checkpoint paths above
available. Choose a new output directory for a new simulation (nonempty output
directories are rejected). Default tasks/checkpoint paths can be inspected in
`--help`; checkpoint paths can be overridden with `--e1_checkpoint` and
`--e2_checkpoint`.

```bat
python scripts\residual_control_decomposition.py --duration 12 --log_dir logs\v7_6_e2_1\new_run --headless
python scripts\residual_control_decomposition.py --log_dir logs\v7_6_e2_1\decomposition_12s --analyze_only
python -m unittest scripts.test_residual_control_metrics scripts.test_residual_smooth_config scripts.test_residual_action_penalty_config
```

Outputs include paired raw trajectories/physical summaries, channel and group
summaries, interval-event CSV, manifest/definition JSON, and PNG/PDF figures:
`rough_control_decomposition`, `combined_control_decomposition`,
`common_mode_motion`. Logs are local artifacts excluded from Git.

## Limits and next experiment

This is two selected conditions, one phase pair and one training seed per
variant; it is not independent test-set certification. GPU/contact numerical
variation limits interpretation of small differences. Sustained saturation
cannot establish which residual would be optimal, nor infer required actuator
stroke. No real wheel-ground dynamics or contact-force sensor was added.
If a diagnostic run terminates, it fails explicitly rather than mixing its
auto-reset state into the preceding episode's control trace.

Next, independently test common-height management (for example a controlled
zero-common-residual ablation or anchored mean-height controller), keeping the
original E1 intact and comparing physical metrics. Such a change alters the
action/control mapping and is NOT a drop-in improvement certified for existing
checkpoints. A projected-policy test is only a diagnostic; retraining and
multi-phase/seed validation may be needed. Do not silently project old actions
or modify the controller as part of this logging task. Do not automatically
raise action magnitude penalty or shrink the residual bound.
