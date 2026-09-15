# V7.6-E3 targeted common-mode reward

Date: 2026-09-15. Branch: `v7.0-active-leveling`. Starting HEAD:
`d33c1f8`.

## Outcome

E3 does not replace E1 best. A soft penalty applied only to the mean of the
three residual actions did not reduce the learned common residual in this
training run. Relative to E1 best, E3 used more common Lift motion, more total
Lift motion and generally more Cargo slip. It retained strong attitude
improvements and improved the six-case Board angular-speed mean, but did not
achieve the intended control-effort tradeoff.

This is a negative one-seed result, not proof that common-mode regularization
can never work. Keep E1 best as the current balanced candidate and do not
silently enable E3 for deployment.

## Isolated change

The new task is:

```text
Template-Agv-Level-Residual-CommonMode-Direct-v0
```

It inherits the complete V7.6-E1 task. E1 settings remain unchanged:

```text
residual action penalty:       0.060
residual action-rate penalty:  0.030
Lift velocity penalty:         0.060
residual action range:         +/-3 mm
```

E3 adds exactly one reward term:

```text
common = mean(action_1, action_2, action_3)
penalty_common = -0.040 * common^2
```

Because

```text
mean(action^2) = mean((action - common)^2) + common^2,
```

the effective common-action coefficient is 0.100 while the differential
coefficient remains 0.060. There is no zero-mean projection, change to the
action bound, common-mode action-rate term, physics change, observation
change or PPO hyperparameter change. The original base, E1 and E2 tasks have
a zero default for the new term.

New runtime diagnostics are:

```text
RewardTerms/common_mode_action
Metrics/common_mode_action_rms
Metrics/differential_action_rms
```

The checkpoint evaluator also reports common/differential residual RMS and
common/differential actual Lift-velocity RMS.

## Pre-training validation

The 64-environment, two-second randomized audit passed:

```text
Reset support gap:             3.000 mm in every environment
Total reward:                  +0.0375 / step
Board angle term:              -0.6669 / step
Board angular-rate term:       -0.0364 / step
Residual action term:          -0.0050 / step
Common-mode action term:       -0.0011 / step
Lift velocity term:            -0.0623 / step
Common/differential action RMS: 0.1641 / 0.2367
Action decomposition max error: 2.484e-8
```

The deterministic zero-residual 12-second regression passed:

```text
Board roll RMS:   0.1328 deg
Board pitch RMS:  0.1067 deg
```

The maximum trajectory differences from V7.5 remained within the existing
tolerances: 0.0002688 deg for angle and 0.037711 mm for linear quantities.

An additional 64-environment E1 compatibility audit reproduced its original
`+0.0386/step` total reward and `-0.0050/step` action term. The newly logged
common-mode term was exactly zero, confirming that the default-zero hook does
not change E1 reward behavior.

## PPO training

Command:

```bat
python -u scripts\skrl\train.py --task Template-Agv-Level-Residual-CommonMode-Direct-v0 --num_envs 64 --headless --max_iterations 300
```

Run:

```text
logs/skrl/agv_level_residual_common_mode_direct/
2026-09-15_10-48-39_ppo_torch_v7_6_e3_residual_ppo_common_mode_reward
```

Training completed from scratch with seed 42 in 1233.27 seconds. All 51
tensors in both evaluated checkpoints are finite.

| Diagnostic | First | Final | Best observed |
|---|---:|---:|---:|
| Instantaneous reward mean | -0.03367 | 0.56493 | 0.73717 |
| Complete-episode return mean | 117.65 | 617.55 | 672.67 |
| Policy standard deviation | 0.22344 | 0.15336 | 0.22344 maximum |
| Value loss | 2.43592 | 0.04161 | 0.00855 minimum |

`best_agent.pt` is tensor-identical to `agent_15360.pt` for the policy,
value and both running preprocessors (24 saved state tensors, maximum absolute
difference 0). It corresponds to approximately 240 iterations. The final
checkpoint is `agent_19200.pt`.

## Held-out six-condition evaluation

E3 best and E3 final were each evaluated against a freshly paired E
zero-residual baseline. Every pair uses one environment, 12 seconds, seed 137,
fixed unseen terrain phases `(1.17,-2.03)` and exactly matching initial
states. Percentages are the arithmetic mean of six per-case improvements;
positive means lower/better than E.

| Policy | Roll RMS | Pitch RMS | Board RP speed | Lift speed | Cargo slip | Common residual | Differential residual | Limit mean / worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 best | +65.85% | +24.18% | +10.42% | -46.28% | +0.57% | 0.246 mm | 1.649 mm | 10.68% / 25.00% |
| E3 best (~240) | +60.53% | +17.89% | +22.03% | -75.46% | -3.14% | 0.465 mm | 1.639 mm | 9.68% / 26.16% |
| E3 final (300) | +66.14% | +29.07% | +24.32% | -69.28% | -11.04% | 0.388 mm | 1.738 mm | 11.89% / 30.32% |

The E3 best common residual is 89.2% larger than E1 best; E3 final is 57.9%
larger. Mean common Lift-velocity RMS similarly changes from 1.180 mm/s for
E1 best to 2.228 mm/s for E3 best and 1.859 mm/s for E3 final. The targeted
penalty therefore did not produce the intended learned behavior in this run.

E3 best per-case results:

| Case | Roll | Pitch | Board RP speed | Lift speed | Cargo slip | Common residual | At limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | +66.16% | +12.21% | +8.86% | -87.73% | -11.42% | 0.381 mm | 2.41% |
| fast | +64.66% | +22.57% | +50.12% | -64.46% | +10.53% | 0.455 mm | 4.35% |
| rough | +49.18% | +23.59% | +37.02% | -69.68% | +0.20% | 0.546 mm | 20.56% |
| heavy | +67.58% | +5.59% | -20.65% | -93.44% | -7.50% | 0.390 mm | 2.31% |
| offset | +67.51% | +9.07% | +4.92% | -89.12% | -2.22% | 0.384 mm | 2.27% |
| combined | +48.07% | +34.34% | +51.94% | -48.33% | -8.43% | 0.633 mm | 26.16% |

E3 final improves the aggregate attitude metrics but transfers effort into
larger differential residual and Cargo motion. Rough/combined limit occupancy
is 25.60%/30.32%. Nominal/rough/combined Cargo cumulative slip is 32.56%,
16.74% and 7.75% worse than the paired E baseline.

## Safety, artifacts and limitations

Both evaluation directories contain 8,640 trajectory rows and 12 summaries.
All values are finite and all E/F initial-state differences are zero. Across
24 E/F runs there is no early termination, Board/Cargo drop or tip,
support-proxy loss, residual final-target saturation, or missing sample.
Support/contact is an analytical proxy, not a PhysX contact sensor.

Outputs:

```text
logs/v7_6_e3/reward_audit/
logs/v7_6_e3/e1_compatibility_audit/
logs/v7_6_e3/zero_12s/
logs/v7_6_e3/checkpoint_best/
logs/v7_6_e3/checkpoint_300/
logs/v7_6_e3/comparison/aggregate_comparison.csv
logs/v7_6_e3/comparison/per_case_comparison.csv
logs/v7_6_e3/comparison/common_mode_reward_comparison.png
logs/v7_6_e3/comparison/common_mode_reward_comparison.pdf
```

The comparison PNG/PDF was rendered and visually inspected. The result is
limited to one training seed and one held-out phase pair. Percent differences
with small denominators, especially Board angular speed, should not be treated
as statistically established. E3 may be sensitive to training variance; the
negative result does establish that adding this single penalty once is not a
reliable improvement over E1.

## Decision

Retain E1 best as the current balanced residual-policy candidate. Reject E3
best/final as replacements. Do not proceed directly to a stronger common-mode
coefficient: the observed response is not monotonic in the intended direction.
If common-mode control remains in scope, first repeat across training seeds or
test a reward tied to physically meaningful mean-height drift rather than raw
mean action. Otherwise proceed with multi-phase/multi-seed inference validation
of E1 best before declaring a final policy.
