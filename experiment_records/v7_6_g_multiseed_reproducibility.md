# V7.6-G: E1 independent-training-seed reproducibility

Date: 2026-09-16. Starting HEAD `593e54c`, branch
`v7.0-active-leveling`.

## Outcome

The retained E1 reward and PPO setup were reproduced with three independent
training seeds (42, 43 and 44). Each policy was evaluated on the same four
held-out terrain phases and the same rough/combined stress cases. Across all
24 strictly paired E/F comparisons, Residual PPO lowers both Board roll and
pitch RMS in every run:

| Metric | Mean improvement | Sample std | Worst result | Improved pairs |
|---|---:|---:|---:|---:|
| Board roll RMS | +54.29% | 5.33% | +45.46% | 24/24 |
| Board pitch RMS | +40.95% | 9.20% | +18.04% | 24/24 |
| Board RP angular speed | +6.84% | 30.99% | -48.90% | 16/24 |
| Lift speed | -35.28% | 10.45% | -59.74% | 0/24 |
| Cargo cumulative slip | -2.39% | 23.33% | -59.64% | 7/24 |

Positive values mean the PPO policy is lower/better than its matched E zero
residual baseline. Negative Lift-speed results therefore mean that actuator
activity increased. The conclusion is two-sided:

- E1's Board roll/pitch benefit is reproducible across training seeds and
  held-out terrain phases.
- Increased Lift activity is also reproducible: all 24 policies/cases have a
  higher Lift-speed RMS than E.
- Board angular-speed behavior is training-seed sensitive and is not a
  universal improvement.
- Cargo slip is not generally improved. In particular, `p3/combined` worsens
  for all three independently trained policies, so that limitation is not a
  seed-42 accident.

E1 remains the most reliable Board-attitude candidate, but it is not a final
comprehensive transport policy.

## Independent training runs

All runs use `Template-Agv-Level-Residual-Smooth-Direct-v0`, the unchanged E1
reward, network, domain randomization and training budget. Seed 42 is the
previous E1 run; seeds 43 and 44 were trained from scratch, not resumed from a
checkpoint. Each policy is the automatically saved `best_agent.pt`; no
checkpoint was selected after inspecting evaluation results.

| Seed | Environments | Iterations | Training time | Episode return first -> last | Policy std first -> last |
|---:|---:|---:|---:|---:|---:|
| 42 | 64 | 300 | 1138.29 s | 119.22 -> 626.21 | 0.2234 -> 0.1560 |
| 43 | 64 | 300 | 1047.96 s | 162.53 -> 681.75 | 0.2232 -> 0.1601 |
| 44 | 64 | 300 | 1044.62 s | 205.01 -> 697.88 | 0.2233 -> 0.1618 |

New runs:

- Seed 43:
  `logs/skrl/agv_level_residual_smooth_direct/2026-09-15_17-30-16_ppo_torch_v7_6_e1_residual_ppo_smooth_reward`
- Seed 44:
  `logs/skrl/agv_level_residual_smooth_direct/2026-09-15_17-48-55_ppo_torch_v7_6_e1_residual_ppo_smooth_reward`

The saved `agent.yaml` and `env.yaml` both contain the requested training seed.
Both new best checkpoints contain 51 finite tensors. The seed is now copied
into every multi-phase manifest and independently recovered from the saved
training configuration by the aggregate analysis.

## Evaluation design

For each policy, V7.6-F's frozen design is reused without modification:

- Four held-out phases: `p0`, `p1`, `p2`, `p3`.
- Two cases: `rough` and `combined`.
- E arm: Geometric + Feedback with zero residual.
- F arm: deterministic clipped policy mean from that seed's E1 best model.
- Duration: 12 s, one environment, 60 Hz control and 120 Hz physics.

This produces:

```text
3 independent PPO training seeds
x 4 held-out terrain phases
x 2 stress cases
= 24 matched E/F comparisons
= 48 physical trajectories
```

Seed 42's already validated V7.6-F trajectories were reused. Seeds 43 and 44
were evaluated in 32 new physical runs. Every trajectory contains 720 logged
intervals: 34,560 rows are represented in the full three-seed analysis, of
which 23,040 rows are new in this stage.

Within every E/F pair, speed, terrain amplitude and phase, Cargo mass and
offset, evaluation seed, duration, reset state and simulation configuration
are identical. All initial-state comparisons are exactly zero.

## Results by training seed

| Training seed | Roll | Pitch | Board RP speed | Lift speed | Cargo slip |
|---:|---:|---:|---:|---:|---:|
| 42 | +53.50% | +32.95% | +13.84% | -28.27% | -2.67% |
| 43 | +51.09% | +46.82% | -4.68% | -39.87% | -1.78% |
| 44 | +58.28% | +43.06% | +11.37% | -37.71% | -2.72% |

The roll/pitch conclusion is consistent for every policy. Board RP angular
speed is much less stable across training seeds: seed 43 has a 4.68% mean
degradation and individual comparisons range from a 48.90% degradation to an
89.90% improvement. This variance must be reported rather than hidden behind
the positive overall mean.

## Results by stress case

| Case | Roll | Pitch | Board RP speed | Lift speed | Cargo slip |
|---|---:|---:|---:|---:|---:|
| rough | +55.50 +/- 5.26% | +39.69 +/- 10.02% | +14.49 +/- 33.24% | -37.26 +/- 13.83% | +7.90 +/- 18.29% |
| combined | +53.08 +/- 5.35% | +42.21 +/- 8.57% | -0.80 +/- 27.85% | -33.31 +/- 5.34% | -12.68 +/- 23.92% |

Values are the mean and sample standard deviation of the paired percentage
improvements across three training seeds and four phases.

## Reproducible Cargo limitation

The most important Cargo result is the predeclared `p3/combined` condition:

| Seed | Roll | Pitch | Board RP speed | Lift speed | Cargo slip E -> F | Cargo improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +51.97% | +34.45% | -12.06% | -31.91% | 3.071 -> 4.715 mm | -53.53% |
| 43 | +51.16% | +47.73% | -48.90% | -36.18% | 3.071 -> 4.903 mm | -59.64% |
| 44 | +53.24% | +47.61% | -3.39% | -38.86% | 3.071 -> 4.315 mm | -40.50% |

All three policies preserve strong Board attitude improvements but increase
Cargo cumulative slip by 40.50-59.64%. The absolute slip remains below 5 mm
and no Cargo drop or tip occurs, but the repeated direction and phase/case
specificity make this a credible objective-design limitation. The next reward
study should therefore be explicitly Cargo-aware and should use this frozen
condition as a required regression case.

## Verification

- All 24 E/F initial-state pairs match exactly.
- All 34,560 trajectory rows are present and all numeric values are finite.
- No early termination, Board/Cargo drop or tip, support-proxy loss, residual
  target saturation or final Lift-target saturation occurred.
- Checkpoint SHA-256 hashes match their evaluation manifests.
- The aggregate script rejects duplicate training seeds, mismatched evaluation
  designs, invalid checkpoint hashes, non-finite data and safety events.
- Python compilation, unit tests, analysis regeneration and
  `git diff --check` pass. Ruff is not installed.
- The PNG/PDF multi-seed comparison figure was rendered and visually checked.

Support/contact remains an analytical geometry proxy, not a PhysX contact
sensor. Three training seeds are enough to establish reproducibility evidence,
but not to characterize the full stochastic distribution of PPO training.

## Artifacts and reproduction

Generated artifacts are Git-ignored:

- `logs/v7_6_g/seed43_multiphase_stress/`
- `logs/v7_6_g/seed44_multiphase_stress/`
- `logs/v7_6_g/multiseed_aggregate/multiseed_runs.csv`
- `logs/v7_6_g/multiseed_aggregate/multiseed_aggregate.csv`
- `logs/v7_6_g/multiseed_aggregate/multiseed_reproducibility.png`
- `logs/v7_6_g/multiseed_aggregate/multiseed_reproducibility.pdf`
- `logs/v7_6_g/multiseed_aggregate/manifest.json`

The two new training commands were:

```bat
python scripts\skrl\train.py ^
  --task=Template-Agv-Level-Residual-Smooth-Direct-v0 ^
  --num_envs=64 --seed=43 --headless --max_iterations=300

python scripts\skrl\train.py ^
  --task=Template-Agv-Level-Residual-Smooth-Direct-v0 ^
  --num_envs=64 --seed=44 --headless --max_iterations=300
```

The new evaluations used the same command with the corresponding checkpoint:

```bat
python scripts\residual_multiphase_eval.py ^
  --suite stress --duration 12 ^
  --checkpoint "<seed checkpoint>" ^
  --output "<seed output directory>" ^
  --headless
```

Aggregate analysis:

```bat
python scripts\residual_multiseed_analysis.py ^
  --input logs\v7_6_f\multiphase_stress ^
  --input logs\v7_6_g\seed43_multiphase_stress ^
  --input logs\v7_6_g\seed44_multiphase_stress ^
  --output logs\v7_6_g\multiseed_aggregate
```

## Decision

Freeze E1 as a reproducible Board-attitude policy candidate. Do not claim that
it improves actuator economy, Board angular dynamics or Cargo stability in all
conditions. The next justified development stage is V7.6-H: first audit the
current Cargo-slip reward/observation scales against the `p3/combined` traces,
then make one isolated Cargo-aware objective change and retrain. Preserve E1
and all 24 paired results as the required comparison baseline.
