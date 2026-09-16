# V7.6-H2: Cargo-target-observable residual policy

Date: 2026-09-16. Starting HEAD `c9008b6`; implementation commits
`199a1b5` and `b3c3306`; branch `v7.0-active-leveling`.

## Outcome

H2 fixes the partial-observability defect found in H1 by replacing the two
Cargo position observation channels with displacement from the randomized
reset target. Observation dimensionality remains 33 and the E1 reward,
action mapping, network, domain randomization and training budget are
unchanged.

The change is useful but does not produce a uniformly superior final policy.
Across the frozen four-phase rough/combined matrix, H2 improves post-settling
Cargo path length by 9.82% relative to E1 on average and in seven of eight
runs. It also reduces Lift velocity by 1.04% relative to E1. The tradeoff is a
0.94%/4.20% mean loss in roll/pitch performance relative to E1, with strong
phase-dependent Board angular-speed behavior. In `p3/combined`, H2 reduces
E1's post-0.5 s Cargo path from 0.4391 to 0.3217 mm, but Board RP angular
speed rises from 0.0002968 to 0.0004748 rad/s.

Keep E1 as the best established Board-attitude candidate. Keep H2 as the
better-founded Cargo-observable candidate, but do not call it the final
transport policy until independent H2 training seeds establish whether the
dynamic variance is reproducible.

## Change isolation

The base task retains its prior behavior:

```python
residual_observe_cargo_slip_from_reset = False
```

The versioned H2 task sets only:

```python
residual_observe_cargo_slip_from_reset = True
```

The observation helper therefore returns:

\[
o_{cargo,xy}=p_{cargo,xy}-p_{cargo,xy}^{reset}
\]

instead of the previous absolute Board-frame Cargo XY. This matches the
existing reward target without adding channels or exposing privileged state.
The task is:

`Template-Agv-Level-Residual-CargoObservable-Direct-v0`

The evaluation also separates the unavoidable initial contact-settling path
from transport slip. Full cumulative path remains reported, while the new
primary diagnostic subtracts cumulative path present at 0.5 s.

## Regression and reset validation

- Eight randomized environments ran finite for 0.5 s.
- Reset Lift-to-Board support gap was exactly 3.000 mm in all eight envs.
- Cargo face gap was effectively zero.
- Reward decomposition error was `1.818e-08`.
- Cargo mass restored deterministically to 4 kg.
- The 12 s zero-residual trajectory retained V7.5 equivalence:
  roll/pitch RMS `0.1328/0.1067 deg`, maximum angle difference
  `0.0002688 deg`, maximum linear-state difference `0.037711 mm`.
- The default E1 task remains backward compatible because the new flag is
  false unless the H2 configuration is selected.

## Training

H2 was trained from scratch; no E1 checkpoint was resumed.

```bat
python scripts\skrl\train.py ^
  --task=Template-Agv-Level-Residual-CargoObservable-Direct-v0 ^
  --num_envs=64 ^
  --seed=42 ^
  --headless ^
  --max_iterations=300
```

Run directory:

`logs/skrl/agv_level_residual_cargo_observable_direct/2026-09-16_10-52-43_ppo_torch_v7_6_h2_residual_ppo_cargo_observable`

- Training time: 1293.18 s.
- Instantaneous reward mean: `-0.03467 -> 0.55017`; maximum `0.72780`.
- Episode return mean: `120.27 -> 611.07`; maximum `667.31`.
- Policy standard deviation: `0.22343 -> 0.15665`.
- The automatically selected `best_agent.pt` was saved at approximately
  iteration 240. Its policy, value and both normalizer state dictionaries are
  tensor-identical to `agent_15360.pt`.
- All 51 checkpoint tensors inspected for best/final checkpoints were finite.
- Best checkpoint SHA-256:
  `ecbf07349698a94730fb29a8466e430118192a092e22809a88ffd8d228bb9a67`.

## Frozen multi-phase evaluation

The retained best checkpoint was evaluated without hand selection against E
zero residual in the same four phase/seed pairs and two stress cases used by
V7.6-F. Every pair used identical speed, terrain, Cargo state, seed, duration,
reset and simulation configuration. All eight initial-state signatures match
exactly.

```bat
python scripts\residual_multiphase_eval.py ^
  --task Template-Agv-Level-Residual-CargoObservable-Direct-v0 ^
  --checkpoint "<H2 run>\checkpoints\best_agent.pt" ^
  --suite stress ^
  --duration 12 ^
  --slip_warmup 0.5 ^
  --output logs\v7_6_h2\multiphase_stress ^
  --headless
```

### Mean improvement relative to paired E

Positive values mean lower/better than E.

| Metric | E1 | H2 | H2 direct vs E1 |
|---|---:|---:|---:|
| Board roll RMS | +53.50% | +52.95% | -0.94% |
| Board pitch RMS | +32.95% | +30.20% | -4.20% |
| Board RP angular speed | +13.84% | +10.84% | -0.03% |
| Lift velocity RMS | -28.27% | -26.89% | +1.04% |
| Full Cargo cumulative path | -2.67% | -3.37% | -2.50% |
| Cargo path after 0.5 s | -20.16% | -6.92% | +9.82% |

The direct H2/E1 column averages per-run percentage changes. It is not the
difference between the two preceding aggregate percentages.

H2 improves roll and pitch over E in 8/8 runs. Board RP speed improves in 6/8,
full Cargo path in 5/8, and post-0.5 s Cargo path in 4/8. Directly relative to
E1, H2 improves post-settling Cargo path in 7/8 runs. Residual at-limit
fraction averages 20.54% for H2 versus 17.75% for E1; final Lift-target
saturation remains zero.

### Important `p3/combined` result

| Metric | E | E1 | H2 |
|---|---:|---:|---:|
| Roll RMS (deg) | 0.23161 | 0.11125 | 0.12226 |
| Pitch RMS (deg) | 0.17559 | 0.11509 | 0.11821 |
| Board RP speed (rad/s) | 0.0002649 | 0.0002968 | 0.0004748 |
| Full Cargo path (mm) | 3.0711 | 4.7149 | 4.3409 |
| Cargo path after 0.5 s (mm) | 0.2712 | 0.4391 | 0.3217 |
| Lift velocity RMS (m/s) | 0.004262 | 0.005622 | 0.005619 |

H2 therefore removes 26.73% of E1's post-settling Cargo path in this worst
Cargo case, but it does not beat E and it substantially worsens Board angular
speed. The observation correction addresses the H1 information mismatch; it
does not by itself solve the multi-objective control tradeoff.

## Verification and artifacts

- 16 physical runs completed: four phases x two cases x E/H2.
- 11,520 expected trajectory rows are present and finite.
- Eight paired initial states match exactly.
- No early termination, Board/Cargo drop or tip, support-proxy loss, or final
  Lift-target saturation occurred.
- The checkpoint hash remained unchanged.
- 40 residual task/evaluation and H2 comparison tests passed.
- Python compilation and `git diff --check` passed.
- Ruff is not installed.
- The PNG/PDF comparison figure was rendered and visually inspected.

Generated Git-ignored artifacts:

- `logs/v7_6_h2/multiphase_stress/`
- `logs/v7_6_h2/comparison/per_run_comparison.csv`
- `logs/v7_6_h2/comparison/aggregate_comparison.csv`
- `logs/v7_6_h2/comparison/cargo_observable_comparison.png`
- `logs/v7_6_h2/comparison/cargo_observable_comparison.pdf`

Support/contact metrics remain analytical geometry proxies, not PhysX contact
sensors. The result is specific to the current idealized contact and virtual
carry model.

## Decision

Do not replace E1 with H2 as a universal final policy, and do not add another
reward weight based on this single training seed. The next controlled step is
H3 reproducibility: train H2 from scratch with two additional fixed seeds and
run the same frozen eight-pair evaluation. If H2 repeatedly reduces
post-settling Cargo path while retaining the same phase-specific angular-speed
failure, the next change should target dynamic smoothness explicitly; if that
failure does not reproduce, it is primarily training variance and model
selection rather than a structural observation defect.
