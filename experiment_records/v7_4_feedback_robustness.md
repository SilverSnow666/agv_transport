# V7.4 Board-feedback robustness validation

## Purpose

V7.3 validated geometric feedforward plus Board-local roll/pitch PD feedback in
one 12-second terrain run. V7.4 checks whether that incremental benefit remains
under changes in speed, analytical terrain shape, and Cargo mass.

Changing only the reset seed is not a terrain robustness test here: the road is
an analytical sinusoidal function, not a randomly generated height field.
Therefore the suite fixes seed 0 and directly changes the parameters that
define the excitation.

## Conditions

Each condition runs D (`geometric`) and E (`geometric + feedback`) for 12 s.
Controller gains, support offsets, AGV kinematics, friction, Board geometry,
and virtual-carry settings are identical between D and E.

| Condition | Speed (m/s) | Bump amplitude (mm) | Phase x/y | Cargo mass (kg) |
|---|---:|---:|---:|---:|
| baseline | 0.10 | 30 | 0.25 / 0.45 | 4 |
| slow | 0.05 | 30 | 0.25 / 0.45 | 4 |
| fast | 0.15 | 30 | 0.25 / 0.45 | 4 |
| rough | 0.10 | 45 | 0.25 / 0.45 | 4 |
| phase_shift | 0.10 | 30 | 1.10 / -0.35 | 4 |
| heavy_cargo | 0.10 | 30 | 0.25 / 0.45 | 8 |

When Cargo mass changes, its inertia tensor is scaled by the same mass ratio.
The terrain visualization is regenerated after amplitude or phase changes so
that it remains consistent with the analytical support surface.

## Command

```bat
python scripts\leveling_terrain_test.py --suite robustness --duration 12 --log_dir logs\v7_4\robustness_12s --headless
```

Per-run CSV files are written below each condition directory. The combined
machine-readable result is `logs/v7_4/robustness_12s/robustness_summary.csv`.

## Primary results

| Condition | D roll RMS | E roll RMS | Improvement | D pitch RMS | E pitch RMS | Improvement |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.2369° | 0.1328° | 43.94% | 0.1848° | 0.1067° | 42.25% |
| slow | 0.2510° | 0.1300° | 48.21% | 0.1980° | 0.1099° | 44.47% |
| fast | 0.2355° | 0.1335° | 43.33% | 0.1874° | 0.1120° | 40.23% |
| rough | 0.3541° | 0.1985° | 43.95% | 0.2763° | 0.1596° | 42.25% |
| phase_shift | 0.1923° | 0.1076° | 44.06% | 0.1869° | 0.1046° | 44.03% |
| heavy_cargo | 0.2368° | 0.1327° | 43.95% | 0.1848° | 0.1067° | 42.26% |

Feedback improves both roll and pitch RMS in all 6 conditions. Mean roll/pitch
improvement is 44.57% / 42.58%; the worst result remains 43.33% / 40.23%.

All 12 runs have:

- 0% Lift saturation;
- 100% Cargo support/contact proxy;
- no Cargo drop or tip;
- measured AGV speed within the existing tolerance.

## Costs and non-monotonic metrics

The feedback controller adds Lift motion and intentionally permits unequal
support heights to cancel residual Board attitude. Support-height RMS therefore
increases and is not a suitable standalone objective for E.

The result is not uniformly better on every dynamic metric:

- In `rough`, Board roll/pitch angular-speed RMS is 16.60% higher under E even
  though roll/pitch angle RMS improves by about 44%/42%.
- E increases Lift velocity RMS in every condition (roughly 4% to 13%).
- Extra corrective motion increases the already-small Cargo slip in several
  cases. The largest E cumulative slip is 1.031 mm in `heavy_cargo`, and the
  largest E relative XY displacement is 0.932 mm; both remain small compared
  with the 300 mm Cargo footprint.

These tradeoffs should be retained rather than tuning the benchmark to make
every reported quantity improve.

## Limitations

- Cargo support/contact remains an analytical geometry proxy, not a PhysX
  contact-sensor measurement.
- AGVs and Lift plates are still kinematic proxies; this does not validate a
  complete wheel-ground-suspension model.
- Terrain variation covers two phases and two amplitudes, not a statistical
  distribution of random height fields.
- The doubled Cargo mass checks load sensitivity, but Board mass and Cargo
  center-of-mass offsets remain fixed.
