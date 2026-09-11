# V7.6-C residual PPO checkpoint evaluation

## Purpose

This stage evaluates whether the 100-iteration V7.6-B PPO smoke checkpoint
learned a useful Lift residual or merely increased its training reward. The
comparison is:

```text
E = V7.5 geometric + Board feedback + zero residual
F = V7.5 geometric + Board feedback + deterministic PPO residual
```

The evaluated checkpoint is:

```text
logs/skrl/agv_level_residual_direct/
2026-09-10_17-49-26_ppo_torch_v7_6_b_residual_ppo_baseline/
checkpoints/best_agent.pt
```

## Evaluation protocol

The new `scripts/residual_checkpoint_eval.py` utility runs E and F sequentially
inside one simulator process. Before each reset it locks every randomized range
to the selected case's constant value. Both controllers use seed 137, terrain
phase `(1.17, -2.03)`, the same 12-second duration, and the same terrain-aware
support-stack initialization.

The script compares a concatenated initial state containing Board, Cargo, all
three AGVs, all three Lift rigid bodies, Lift height, and Lift velocity. The
maximum E/F initial-state difference was exactly zero in all six cases.

| Case | Speed | Terrain amplitude | Cargo mass | Cargo X offset |
|---|---:|---:|---:|---:|
| nominal | 0.10 m/s | 30 mm | 4 kg | 0 mm |
| fast | 0.18 m/s | 30 mm | 4 kg | 0 mm |
| rough | 0.10 m/s | 50 mm | 4 kg | 0 mm |
| heavy | 0.10 m/s | 30 mm | 8 kg | 0 mm |
| offset | 0.10 m/s | 30 mm | 6 kg | 30 mm |
| combined | 0.18 m/s | 50 mm | 8 kg | 30 mm |

Command:

```bat
python scripts\residual_checkpoint_eval.py --case all --duration 12 --headless --log_dir logs\v7_6_c\heldout_12s
```

The script accepts an explicit `--checkpoint`. When omitted, it selects the
most recently modified `best_agent.pt` under
`logs/skrl/agv_level_residual_direct` and writes the resolved absolute path to
the summary and improvement CSVs.

## Board attitude results

| Case | E roll RMS | F roll RMS | Roll improvement | E pitch RMS | F pitch RMS | Pitch improvement |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 0.1546 deg | 0.0868 deg | 43.82% | 0.0738 deg | 0.0622 deg | 15.77% |
| fast | 0.1561 deg | 0.0896 deg | 42.64% | 0.0823 deg | 0.0696 deg | 15.42% |
| rough | 0.2558 deg | 0.1589 deg | 37.87% | 0.1224 deg | 0.1095 deg | 10.49% |
| heavy | 0.1547 deg | 0.0838 deg | 45.85% | 0.0735 deg | 0.0655 deg | 10.94% |
| offset | 0.1547 deg | 0.0886 deg | 42.70% | 0.0738 deg | 0.0612 deg | 17.06% |
| combined | 0.2576 deg | 0.1631 deg | 36.68% | 0.1369 deg | 0.1222 deg | 10.72% |
| Mean | - | - | 41.59% | - | - | 13.40% |

F reduced both roll and pitch RMS in all six cases. Mean per-step reward also
improved in all six cases. The strongest absolute attitude reduction occurred
under the rough and combined terrain conditions, so the policy is not simply
adding a nominal constant offset.

## Dynamic, Cargo, and effort trade-offs

The attitude improvement is not free:

- Board roll/pitch angular-velocity RMS improved in nominal, rough, and
  combined, but worsened in fast, heavy, and offset. The mean change was a
  6.24% degradation, driven mainly by a 45.26% increase in the heavy case.
- Board vertical-acceleration RMS changed by only -0.009% on average. It is
  effectively unchanged at this precision.
- Cargo relative-XY RMS improved by 0.81% on average. Cargo cumulative slip
  improved by 0.76% on average, but nominal, rough, heavy, and offset each had
  small individual regressions of at most 1.71%.
- Lift-velocity RMS increased in all six cases, by 14.54% to 33.37% and 22.17%
  on average. Its largest absolute value was still only 4.46 mm/s.
- PPO residual RMS ranged from 0.85 to 1.27 mm. The combined case reached the
  3 mm residual command limit for 10 of 2,160 channel samples (0.463%). All
  other cases had 0% residual-limit occupancy.
- No final Lift target hit the mechanical height bounds in any case.

## Safety and data checks

All 12 runs completed the requested 720 control steps. There were no early
terminations, Board/Cargo drops, Board/Cargo tips, analytical support losses, or
Lift target saturations. Mean analytical support count remained 3.0.

Output files:

```text
logs/v7_6_c/heldout_12s/residual_checkpoint_trajectories.csv
logs/v7_6_c/heldout_12s/residual_checkpoint_summary.csv
logs/v7_6_c/heldout_12s/residual_checkpoint_improvements.csv
```

The final files contain 8,640 trajectory rows, 12 controller summaries, and 72
finite E-to-F metric comparisons. Every E/F group contains exactly 720 samples;
there are no missing cells or non-finite summary values. Support/contact values
remain analytical proxies rather than PhysX contact-sensor measurements.

## Conclusion and next step

The short PPO checkpoint has learned a meaningful residual direction: it
consistently reduces both Board attitude RMS measures across all six fixed
evaluation samples without compromising support or Lift travel safety. This is
enough evidence to proceed beyond the smoke model.

The checkpoint is not yet a final result. The evaluation uses one deterministic
seed and one held-out phase pair, both inside the training distribution. Longer
training should retain the present reward and log every checkpoint, followed by
multi-seed evaluation. The final model must be selected using Board attitude,
angular velocity, Cargo behavior, Lift effort, and residual-limit occupancy,
not training return alone.
