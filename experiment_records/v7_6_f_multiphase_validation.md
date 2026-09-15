# V7.6-F: E1 held-out multi-phase stress validation

Date: 2026-09-15. Starting HEAD `aa80ee7`, branch
`v7.0-active-leveling`.

## Outcome

The retained E1 best checkpoint improves both Board roll and pitch RMS in all
eight matched held-out stress pairs. Across four deterministic terrain phases
and the rough/combined conditions, mean roll/pitch improvements over E are
53.50%/32.95%; the worst observed improvements remain positive at
50.88%/18.04%. There are no early terminations, Board/Cargo drop or tip events,
support-proxy losses, or final Lift-target saturation events.

E1 is therefore robust as a Board-attitude controller across this phase set.
It is not yet a globally superior final policy: Lift velocity increases in all
eight pairs (28.27% on average), Board RP angular speed worsens in two combined
cases, and Cargo cumulative slip is phase-sensitive. The worst Cargo result is
a 53.53% increase in `p3/combined`, despite a small 2.67% mean increase across
all pairs. Retain E1 best as the posture-performance candidate and report these
tradeoffs; do not claim universal dynamic or Cargo improvement.

No controller, reward, physics, action mapping, checkpoint, or training setup
was changed and no new training was run.

## Evaluation design

Each child process runs a strictly matched pair:

- E: Geometric + Feedback with zero residual.
- E1: the deterministic clipped policy mean from the retained E1 best
  checkpoint.

The task is `Template-Agv-Level-Residual-Smooth-Direct-v0`. Each arm uses one
environment, a 12 s episode, 60 Hz control and 120 Hz physics. Within every
pair, speed, terrain amplitude and phase, Cargo properties and offset, seed,
duration, reset state and simulation configuration are identical. All eight
E/E1 initial-state signatures match exactly.

Checkpoint:

`logs/skrl/agv_level_residual_smooth_direct/2026-09-14_15-10-40_ppo_torch_v7_6_e1_residual_ppo_smooth_reward/checkpoints/best_agent.pt`

SHA-256:

`ebfd609defc438a155202a7606fd22e85ae67ff145143de09adf294e270b9817`

The four phase/seed pairs were fixed before execution and span all sign
quadrants:

| Label | Seed | Phase X (rad) | Phase Y (rad) |
|---|---:|---:|---:|
| p0 | 137 | 1.17 | -2.03 |
| p1 | 211 | -1.31 | 0.77 |
| p2 | 353 | 2.43 | 1.88 |
| p3 | 509 | -2.68 | -0.49 |

The stress cases are:

| Case | Speed | Terrain amplitude | Cargo mass | Cargo XY offset |
|---|---:|---:|---:|---:|
| rough | 0.10 m/s | 50 mm | 4 kg | 0, 0 mm |
| combined | 0.18 m/s | 50 mm | 8 kg | 30, 0 mm |

The phase is the meaningful physical variation here. Because each case pins
the environment randomization ranges to deterministic values, changing the
seed primarily checks reset reproducibility; these are not independent PPO
training seeds and must not be presented as such.

## Per-pair improvements

Positive values mean E1 is lower/better than E. Lift-speed increases are shown
as positive costs for readability, whereas the raw CSV stores them as negative
improvements.

| Phase/case | Roll | Pitch | Board RP speed | Lift speed increase | Cargo slip |
|---|---:|---:|---:|---:|---:|
| p0 rough | +54.04% | +26.39% | +24.07% | 40.65% | -1.19% |
| p0 combined | +51.87% | +33.67% | +42.85% | 31.73% | +3.49% |
| p1 rough | +51.65% | +18.04% | +12.39% | 14.84% | +0.04% |
| p1 combined | +50.88% | +25.16% | -9.38% | 24.15% | -0.42% |
| p2 rough | +61.51% | +44.78% | +34.75% | 27.66% | -1.57% |
| p2 combined | +53.95% | +48.50% | +9.97% | 27.41% | -0.49% |
| p3 rough | +52.17% | +32.63% | +8.14% | 27.83% | +32.31% |
| p3 combined | +51.97% | +34.45% | -12.06% | 31.91% | -53.53% |

Cargo slip is cumulative path length, not final displacement. Its percentage
is sensitive to the baseline denominator and the phase-dependent path. In
absolute terms, `p3/combined` changes from 3.0711 mm to 4.7149 mm. This is not a
drop/tip event, but it is a real 1.6439 mm increase and is retained as an
important limitation.

## Aggregate results

The table below averages the per-pair percentage improvements, rather than
computing a ratio of pooled raw samples. Standard deviations are sample
standard deviations across the four phases.

| Scope / metric | Mean | Std | Min | Max | Improved pairs |
|---|---:|---:|---:|---:|---:|
| rough roll | +54.84% | 4.56% | +51.65% | +61.51% | 4/4 |
| rough pitch | +30.46% | 11.26% | +18.04% | +44.78% | 4/4 |
| rough Board RP speed | +19.84% | 12.01% | +8.14% | +34.75% | 4/4 |
| rough Lift speed | -27.74% | 10.54% | -40.65% | -14.84% | 0/4 |
| rough Cargo slip | +7.40% | 16.62% | -1.57% | +32.31% | 2/4 |
| combined roll | +52.16% | 1.29% | +50.88% | +53.95% | 4/4 |
| combined pitch | +35.45% | 9.67% | +25.16% | +48.50% | 4/4 |
| combined Board RP speed | +7.84% | 25.32% | -12.06% | +42.85% | 2/4 |
| combined Lift speed | -28.80% | 3.73% | -31.91% | -24.15% | 0/4 |
| combined Cargo slip | -12.74% | 27.26% | -53.53% | +3.49% | 1/4 |
| all roll | +53.50% | 3.42% | +50.88% | +61.51% | 8/8 |
| all pitch | +32.95% | 10.08% | +18.04% | +48.50% | 8/8 |
| all Board RP speed | +13.84% | 19.43% | -12.06% | +42.85% | 6/8 |
| all Lift speed | -28.27% | 7.34% | -40.65% | -14.84% | 0/8 |
| all Cargo slip | -2.67% | 23.51% | -53.53% | +32.31% | 3/8 |

Across the eight raw runs, mean Board roll/pitch RMS changes from
0.23795/0.14672 deg for E to 0.11131/0.09685 deg for E1. Mean Board RP angular
speed changes from 0.0009992 to 0.0007510 rad/s. Mean Lift speed changes from
3.4710 to 4.4436 mm/s, and mean Cargo cumulative slip from 2.6733 to 2.7389 mm.

Residual RMS averages 1.8406 mm for rough and 1.8780 mm for combined. The
residual at-limit fraction averages 16.00%/19.51%, with a worst run of 25.00%.
Common residual RMS averages 0.2884/0.3258 mm. These values confirm that the
known action-usage tradeoff remains under held-out phases even though final
Lift target saturation is zero.

## Verification and artifacts

- 16 physical runs completed: four phases x two cases x E/E1.
- Each run contributes 720 intervals; 11,520 trajectory rows were validated.
- All numeric trajectory fields are finite and all expected rows are present.
- All eight paired initial-state maximum differences are exactly zero.
- No early termination, Board/Cargo drop or tip, support-proxy loss, or final
  Lift-target saturation occurred.
- The checkpoint SHA-256 was unchanged before and after evaluation.
- Analysis-only regeneration, Python compilation, unit tests and
  `git diff --check` passed. Ruff is not installed.
- The PNG/PDF comparison plot was rendered and visually checked.

Support/contact counts remain analytical geometry proxies, not PhysX contact
sensors. This evaluation validates the current model's closed-loop behavior;
it does not resolve the physical-model realism limitations documented by the
earlier Board-support ablation work.

Generated artifacts (Git-ignored):

- `logs/v7_6_f/multiphase_stress/multiphase_runs.csv`
- `logs/v7_6_f/multiphase_stress/multiphase_aggregate.csv`
- `logs/v7_6_f/multiphase_stress/multiphase_stress_comparison.png`
- `logs/v7_6_f/multiphase_stress/multiphase_stress_comparison.pdf`
- `logs/v7_6_f/multiphase_stress/manifest.json`
- Eight nested paired summary, improvement and trajectory CSV sets.

## Reproduction

From the repository root in `env_isaaclab`:

```bat
python scripts\residual_multiphase_eval.py ^
  --suite stress ^
  --duration 12 ^
  --output logs\v7_6_f\multiphase_stress ^
  --headless

python scripts\residual_multiphase_eval.py ^
  --output logs\v7_6_f\multiphase_stress ^
  --analyze_only

python -m unittest scripts.test_residual_multiphase_eval
```

A fresh physical run requires an empty output directory; analysis-only mode
reuses the manifest and raw CSV data. `--suite all` supports all six standard
conditions, but this stage deliberately uses the two stress cases where the
previous single-phase limitations were most visible.

## Decision

Freeze reward experiments E2 and E3 as negative results. Keep the E1 best
checkpoint as the current Residual PPO candidate for Board-attitude control.
The next useful experiment is not another one-factor reward change: either
validate the final candidate with multiple independently trained policies, or
address Cargo slip explicitly in the objective/observation and then retrain.
Until that is done, use E1 only when the Board-attitude benefit justifies the
measured Lift-activity and phase-dependent Cargo-slip costs.
