# V7.5 reusable environment-level Board controller

## Purpose

V7.3 and V7.4 validated Board-local roll/pitch PD feedback, but the controller
lived only in `scripts/leveling_terrain_test.py`. V7.5 promotes the same logic
into `AgvLevelCarryEnv` so training, demos, and future tests can select it
without duplicating the benchmark implementation.

The compatibility default remains `external`: existing policies and scripts
may continue writing `lift_target_height` directly. The additional modes are:

- `neutral`: fixed neutral Lift height;
- `geometric`: equalize the three physical Lift support-top world heights;
- `geometric_feedback`: geometric feedforward plus filtered, bounded,
  zero-mean Board-local roll/pitch PD correction.

The environment computes targets once per control step in `_pre_physics_step`.
The existing finite-speed Lift actuator still tracks those targets during the
physics substeps.

## Numerical parity check

Both implementations were run from seed 0 for 2 seconds at 0.10 m/s:

```bat
python scripts\leveling_terrain_test.py --mode feedback --controller_source script --duration 2 --target_speed 0.10 --log_dir logs\v7_5\smoke_script --headless
python scripts\leveling_terrain_test.py --mode feedback --controller_source environment --duration 2 --target_speed 0.10 --log_dir logs\v7_5\smoke_environment --headless
```

The two CSV files each contain 121 samples. Across every numeric CSV field,
the maximum absolute sample difference is `0.0`.

## Full environment-controller regression

```bat
python scripts\leveling_terrain_test.py --mode all --controller_source environment --duration 12 --target_speed 0.10 --log_dir logs\v7_5\environment_12s --headless
```

| Metric | Neutral (C) | Geometric (D) | Geometric + feedback (E) |
|---|---:|---:|---:|
| Board roll RMS | 0.4076 deg | 0.2369 deg | 0.1328 deg |
| Board pitch RMS | 0.2518 deg | 0.1848 deg | 0.1067 deg |
| Max abs roll | 0.8425 deg | 0.4917 deg | 0.2561 deg |
| Max abs pitch | 0.5979 deg | 0.3704 deg | 0.2157 deg |
| Board roll/pitch angular-speed RMS | 0.001218 rad/s | 0.000131 rad/s | 0.000050 rad/s |
| Board vertical-acceleration RMS | 0.004224 m/s^2 | 0.002423 m/s^2 | 0.000667 m/s^2 |
| Support-height RMS | 2.550 mm | 0.394 mm | 0.957 mm |
| Lift velocity RMS | 0.000 mm/s | 1.997 mm/s | 2.241 mm/s |
| Feedback-height RMS | 0.000 mm | 0.000 mm | 0.925 mm |
| Cargo contact proxy | 100% | 100% | 100% |
| Cargo dropped / tipped | no / no | no / no | no / no |
| Lift saturation | 0% | 0% | 0% |

C to D improves roll/pitch RMS by 41.89% / 26.59%. D to E adds a further
43.94% / 42.25% improvement. The higher support-height RMS under E is expected:
feedback intentionally offsets the support tops to cancel residual Board
attitude rather than minimizing support-height mismatch alone.

## Limitations

- Cargo support/contact is an analytical geometry proxy, not a PhysX contact
  sensor measurement.
- AGVs and Lift plates remain kinematic proxies.
- The full V7.4 robustness matrix was not repeated because the script and
  environment implementations matched exactly sample-for-sample; the prior
  six-condition results remain the robustness evidence.

## Compatibility checks

- Python `py_compile` passed for the environment, both related config files,
  and the terrain benchmark.
- `git diff --check` passed.
- `scripts/leveling_static_test.py --duration 1 --headless` exited successfully
  with the default `external` mode, confirming that existing scripts can still
  write Lift targets directly.
- `ruff` is not installed in the active Isaac Lab environment.
