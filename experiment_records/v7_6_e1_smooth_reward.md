# V7.6-E1 smooth residual reward refinement

## Objective

V7.6-D showed that the original residual PPO policy reduced static Board
roll/pitch error but used frequent +/-3 mm residual limits and greatly
increased Lift and Board angular motion. E1 tests whether reward shaping can
retain useful attitude correction while reducing this over-control.

The experiment is deliberately isolated as a new task:

```text
Template-Agv-Level-Residual-Smooth-Direct-v0
```

The environment implementation, physics, 33-dimensional observation, three
residual actions, +/-3 mm residual limit, base geometric-feedback controller
and domain randomization are unchanged. Only seven reward/reference values and
the experiment output identity differ from the V7.6 baseline.

## Reward-only change

| Setting | V7.6 baseline | V7.6-E1 |
|---|---:|---:|
| Board angle penalty scale | 1.00 | 0.75 |
| Board angular-velocity reference | 0.010 rad/s | 0.003 rad/s |
| Board angular-velocity penalty scale | 0.05 | 0.10 |
| Residual action penalty scale | 0.010 | 0.060 |
| Residual action-rate penalty scale | 0.005 | 0.030 |
| Lift velocity reference | 0.010 m/s | 0.005 m/s |
| Lift velocity penalty scale | 0.020 | 0.060 |

Cargo terms, failure penalties and the residual action range are unchanged.
`scripts/test_residual_smooth_config.py` statically verifies this isolation and
also verifies that the PPO YAML changes only the output directory/name.

## Pre-training validation

The 64-environment, two-second randomized reward audit passed:

```text
Reset support gap:        3.000 mm in every environment
Total reward:             +0.0386 / step
Board angle term:         -0.6669 / step
Board angular-rate term:  -0.0364 / step
Board vertical term:      -0.0986 / step
Cargo slip term:          -0.0641 / step
Residual action term:     -0.0050 / step
Lift velocity term:       -0.0623 / step
```

The deterministic zero-residual 12-second regression also passed:

```text
Board roll RMS:   0.1328 deg
Board pitch RMS:  0.1067 deg
```

The maximum difference from the V7.5 reference was 0.0002688 deg for angle and
0.037711 mm for linear quantities, within the existing regression tolerances.
This confirms that the versioned task does not alter the zero-residual physical
trajectory.

## PPO training

Command:

```bat
python scripts\skrl\train.py --task Template-Agv-Level-Residual-Smooth-Direct-v0 --num_envs 64 --headless --max_iterations 300
```

Run:

```text
logs/skrl/agv_level_residual_smooth_direct/
2026-09-14_15-10-40_ppo_torch_v7_6_e1_residual_ppo_smooth_reward
```

Training completed in 1138.29 seconds. All TensorBoard scalar samples and all
51 tensors in `agent_19200.pt` and `best_agent.pt` are finite.

| Diagnostic | First | Final | Best observed |
|---|---:|---:|---:|
| Instantaneous reward mean | -0.0333 | 0.5743 | 0.7439 |
| Complete-episode return mean | 119.22 | 626.21 | 681.57 |
| Policy standard deviation | 0.2234 | 0.1560 | 0.2234 maximum |
| Value loss | 2.4343 | 0.0363 | 0.00788 minimum |

The automatically selected `best_agent.pt` is inference-identical to
`agent_15360.pt`, corresponding to approximately 240 iterations: policy,
value, state-preprocessor and value-preprocessor tensor maximum differences
are all 0.0.

## Held-out six-condition evaluation

Both the final 300-iteration checkpoint and the automatic best checkpoint were
compared with E (geometric + feedback, zero residual) using the same seed 137,
unseen fixed terrain phases `(1.17, -2.03)`, initial state and 12-second
duration. Positive percentages mean F is better than E; negative percentages
mean F is worse.

### Automatic best checkpoint (approximately 240 iterations)

| Case | Roll RMS | Pitch RMS | Board RP angular speed | Lift velocity | Cargo cumulative slip | Residual RMS | At limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | +72.40% | +19.82% | -24.09% | -53.61% | -1.04% | 1.494 mm | 3.47% |
| fast | +69.68% | +25.40% | +3.08% | -45.38% | +2.15% | 1.560 mm | 5.46% |
| rough | +54.04% | +26.40% | +26.52% | -40.65% | -0.75% | 1.920 mm | 20.65% |
| heavy | +73.78% | +16.38% | -0.29% | -56.05% | +0.59% | 1.543 mm | 4.63% |
| offset | +73.35% | +23.43% | +14.45% | -50.24% | -1.05% | 1.511 mm | 4.86% |
| combined | +51.87% | +33.67% | +42.85% | -31.73% | +3.49% | 1.974 mm | 25.00% |
| **Mean** | **+65.85%** | **+24.18%** | **+10.42%** | **-46.28%** | **+0.57%** | **1.667 mm** | **10.68%** |

### Aggregate comparison

| Policy | Roll RMS | Pitch RMS | Board RP angular speed | Lift velocity | Cargo slip | Residual RMS | Mean at limit | Worst at limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 best (~240) | +65.85% | +24.18% | +10.42% | -46.28% | +0.57% | 1.67 mm | 10.68% | 25.00% |
| E1 final (300) | +70.40% | +28.41% | -1.49% | -49.04% | -0.87% | 1.79 mm | 15.69% | 29.77% |
| Old V7.6 checkpoint 300 | +67.45% | +44.58% | -14.84% | -86.71% | -3.13% | 1.91 mm | 21.92% | 36.30% |
| Old V7.6 checkpoint 600 | +73.35% | +56.40% | -38.68% | -143.48% | -1.37% | 2.11 mm | 30.89% | 46.44% |

The E1 best checkpoint is the better balanced E1 candidate. Compared with the
old 300-iteration policy, it changes mean Board angular speed from a 14.84%
regression to a 10.42% improvement, reduces Lift-velocity increase from 86.71%
to 46.28%, and reduces mean residual limiting from 21.92% to 10.68%. It also
keeps strong roll reduction. The tradeoff is lower pitch improvement.

## Decision and limitation

E1 validates the reward-refinement direction, but is not yet accepted as the
final residual policy. Its mean at-limit rate narrowly misses the 10% target,
and rough/combined cases still reach 20.65%/25.00%. Mean pitch improvement is
24.18%, just below the provisional 25-35% range, while the nominal case still
shows a 24.09% Board angular-speed regression.

A follow-up E2 experiment should make a small, versioned adjustment rather
than overwrite E1. The main target is residual limiting on high-amplitude
terrain without returning to the V7.6 baseline's excessive Lift motion. E1
should remain the documented evidence that reward shaping substantially
improves the control-effort tradeoff.

Each evaluation directory contains 8,640 trajectory samples, 12 summaries and
72 metric comparisons; every E/F run contains 720 samples. There are no
initial-state mismatches, non-finite values, early terminations, Board/Cargo
drops or tips, or analytical support-proxy losses. Final Lift target
saturation is zero. Support/contact values are analytical proxies, not PhysX
contact-sensor measurements.

Evaluation outputs:

```text
logs/v7_6_e1/checkpoint_300/
logs/v7_6_e1/checkpoint_best/
```
