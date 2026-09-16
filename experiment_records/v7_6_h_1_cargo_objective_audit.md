# V7.6-H.1: Cargo objective and trajectory audit

Date: 2026-09-16. Starting HEAD `0af4c03`, branch
`v7.0-active-leveling`.

## Outcome

This stage audits the reproducible Cargo-slip limitation found in V7.6-G. It
does not change the controller, reward, observation, checkpoint or physics and
does not retrain PPO.

The main finding is not simply that the Cargo reward weight is too small. The
current task has an objective/observation mismatch:

```text
Cargo slip reward:
    cargo_relative_xy - cargo_initial_relative_xy

Policy observation:
    cargo_relative_xy
```

The randomized per-episode initial Cargo offset is not separately observed,
and the policy is not given the slip-from-reset vector. The same observed
Cargo position can therefore mean zero slip in one episode and a large error
in another. This makes the randomized slip objective partially observable.

The `p3/combined` traces also show that most of the reported E1 Cargo penalty
is created during the reset/contact-settling transient. Across all three
training seeds, 89-95% of the final E1-vs-E cumulative-slip difference is
already present at 0.5 s. Increasing a reward weight before correcting these
two issues would conflate objective strength, missing state information and
startup control behavior.

## Inputs

The audit reuses the frozen `p3/combined` paired trajectories from V7.6-F/G:

- E: Geometric + Feedback with zero residual.
- F: independently trained E1 policies with training seeds 42, 43 and 44.
- Speed 0.18 m/s, 50 mm terrain amplitude, 8 kg Cargo and 30 mm initial
  longitudinal Cargo offset.
- Phase `p3 = (-2.68, -0.49)`, evaluation seed 509, duration 12 s.

The analysis reads each checkpoint's saved `agent.yaml` and `env.yaml`, checks
the checkpoint SHA-256 and reconstructs every E1 reward term directly from the
logged state. Maximum reward reconstruction error is below `1.9e-6` per step,
which confirms that the reward-scale comparison matches the executed task.

## Reset transient versus transport slip

| Training seed / controller | First-step slip | Slip at 0.5 s | Slip after 0.5 s | Final cumulative slip |
|---|---:|---:|---:|---:|
| E zero residual | 2.773 mm | 2.800 mm | 0.271 mm | 3.071 mm |
| E1 seed 42 | 2.773 mm | 4.276 mm | 0.439 mm | 4.715 mm |
| E1 seed 43 | 2.773 mm | 4.537 mm | 0.365 mm | 4.903 mm |
| E1 seed 44 | 2.773 mm | 3.911 mm | 0.404 mm | 4.315 mm |

The first physics/control interval is identical and contributes 2.773 mm to
all arms. That is 90.3% of E's final cumulative-slip metric. In E1, the first
policy actions then create most of the additional displacement within the
first 50-500 ms. Relative to E, the extra final slip is 1.244-1.832 mm; the
extra slip already present at 0.5 s is 1.111-1.738 mm, or 89-95% of that total.

After the 0.5 s settling window E accumulates another 0.271 mm. E1 accumulates
0.365-0.439 mm, an absolute increase of 0.094-0.168 mm. E1 therefore still has
a smaller transport-phase disadvantage, but the previously reported 40-60%
final percentage degradation is dominated by startup behavior and should not
be interpreted as uniform slip throughout the 12 s trajectory.

The evaluation metric should retain full cumulative slip for transparency but
also report settling-window and post-warmup components in future experiments.

## Reward-scale audit

Mean penalty per control step after 0.5 s:

| Controller | Board angle | Cargo slip | Cargo velocity | Lift velocity | Action |
|---|---:|---:|---:|---:|---:|
| E | -1.590 | -0.077 | -0.0013 | -0.043 | 0.000 |
| E1 seed 42 | -0.492 | -0.178 | -0.0014 | -0.075 | -0.021 |
| E1 seed 43 | -0.405 | -0.204 | -0.0016 | -0.081 | -0.028 |
| E1 seed 44 | -0.387 | -0.150 | -0.0018 | -0.085 | -0.029 |

The Cargo slip term is not numerically invisible. In E1 it is 31-51% of the
Board-angle penalty magnitude, while the Cargo velocity term is two orders of
magnitude smaller. The policy can gain about 1.10-1.20 reward units per step by
reducing Board angle relative to E, while paying about 0.07-0.13 more Cargo
slip penalty plus the added action/Lift costs. The observed tradeoff is thus
consistent with the current reward, rather than evidence that the Cargo term
is missing from reward computation.

This also explains why blindly increasing the slip penalty is not the clean
next experiment: the policy is already reacting to a substantial term but
cannot unambiguously observe that term's episode-specific zero point.

## Observation audit

The 33-dimensional observation includes:

```text
indices 26-27: current Cargo position in the Board frame
indices 28-29: Cargo relative XY velocity
indices 30-32: Cargo relative roll, pitch and angular speed
```

It does not include `cargo_initial_relative_xy` or
`cargo_relative_xy - cargo_initial_relative_xy`. The saved running scaler does
contain plausible statistics for the current-position channels (roughly
17 mm standard deviation), so this is not a normalization failure. It is a
semantic observability failure: randomizing the desired Cargo offset changes
the reward target without exposing that target to the policy.

A clean H2 experiment should replace the two current-position observation
channels with slip-from-reset X/Y while retaining the same 33-dimensional
space, E1 reward, actions, network, randomization and training budget. This is
an isolated observation correction and avoids simultaneously tuning a reward
weight. The old E1 checkpoints must not be used with the changed observation
semantics; H2 must train from scratch.

## Dynamic associations

For E1, Pearson correlation between per-step Cargo path increment and absolute
Board vertical acceleration is 0.29-0.70 across the three seeds. Correlation
with residual-rate magnitude is 0.21-0.59. Correlation with Board RP speed is
only 0.07-0.21, while simultaneous Lift speed is weakly negative. These values
do not prove causation, but they support the trace-level observation that
Cargo movement is associated more strongly with vertical/startup excitation
and rapid residual changes than with the small remaining Board tilt itself.

## Verification and artifacts

- Six 720-sample controller trajectories were audited; no simulation was
  rerun and no checkpoint was modified.
- Checkpoint hashes, saved training seeds and saved reward configurations were
  verified.
- All reward terms were reconstructed with maximum absolute error below
  `1.9e-6`.
- Python compilation, unit tests, analysis regeneration and
  `git diff --check` pass. Ruff is not installed.
- The PNG/PDF audit figure was rendered and visually inspected.

Generated artifacts are Git-ignored:

- `logs/v7_6_h/cargo_objective_audit/cargo_tradeoff_summary.csv`
- `logs/v7_6_h/cargo_objective_audit/reward_term_trajectories.csv`
- `logs/v7_6_h/cargo_objective_audit/cargo_slip_correlations.csv`
- `logs/v7_6_h/cargo_objective_audit/observation_scalers.csv`
- `logs/v7_6_h/cargo_objective_audit/cargo_tradeoff_audit.png`
- `logs/v7_6_h/cargo_objective_audit/cargo_tradeoff_audit.pdf`
- `logs/v7_6_h/cargo_objective_audit/manifest.json`

Reproduction:

```bat
python scripts\analyze_residual_cargo_tradeoff.py ^
  --output logs\v7_6_h\cargo_objective_audit

python -m unittest scripts.test_residual_cargo_tradeoff
```

## Decision

Do not increase the Cargo-slip reward weight yet. Proceed to V7.6-H2 with one
change only: expose Cargo slip-from-reset X/Y instead of absolute Cargo
relative X/Y in the policy observation, then train from scratch under the
unchanged E1 reward. Evaluate H2 on the frozen 24-pair V7.6-G matrix and report
both full cumulative slip and post-0.5-s slip. A residual startup ramp can be
studied later as a separate ablation if the early transient remains after the
observation correction.
