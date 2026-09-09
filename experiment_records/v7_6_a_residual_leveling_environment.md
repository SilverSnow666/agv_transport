# V7.6-A residual Lift RL environment

## Purpose

V7.5 integrated the validated geometric + Board-attitude feedback controller
into the environment. V7.6-A adds a separate task for residual learning without
changing the existing six-action AGV task:

```text
Template-Agv-Level-Residual-Direct-v0
```

The three policy actions are normalized residuals for Lift1, Lift2, and Lift3.
AGV translation is scripted inside the environment, so a future policy cannot
improve the score by changing vehicle motion. The applied command is

```text
h_final = clamp(h_geometric+feedback + 0.003 * action, h_min, h_max)
```

Thus each residual is bounded to +/-3 mm and the V7.5 controller remains the
interpretable baseline.

## Observation and reward

The policy observation has 33 values:

- Board roll, pitch, local roll/pitch angular velocity, and vertical velocity;
- three Lift heights, velocities, V7.5 base targets, feedback corrections, and
  support-top relative heights;
- roll and pitch of all three analytical AGV terrain planes;
- Cargo Board-frame XY position and velocity, relative roll/pitch, and relative
  angular-speed magnitude.

The reward only concerns Board stability, Cargo stability, and residual control
effort. It includes Board attitude/angular/vertical-velocity costs, Cargo slip,
relative velocity, tilt and angular-velocity costs, action/action-rate and Lift
velocity costs, plus a terminal failure penalty. There is no transport-progress
reward because vehicle motion is not controlled by the policy.

## Reset domain randomization

Each vectorized environment independently samples:

| Quantity | Range |
|---|---:|
| Scripted speed | 0.08 to 0.18 m/s |
| Terrain amplitude | 20 to 50 mm |
| Terrain X/Y phase | -pi to +pi |
| Cargo mass | 3 to 8 kg |
| Cargo initial X/Y offset | -30 to +30 mm |

Cargo inertia is scaled by the same factor as mass. The analytical terrain is
evaluated per environment; the single shared visual terrain mesh is disabled by
default for vectorized training because it cannot show different terrain
parameters for every cloned environment.

## Validation

### Action mapping

```bat
python scripts\residual_leveling_test.py --case mapping --duration 0.5 --num_envs 1 --headless
```

All three one-hot checks passed: action channels 1, 2, and 3 changed only
Lift1, Lift2, and Lift3 respectively, by exactly +3.000 mm.

### Vectorized randomization smoke test

```bat
python scripts\residual_leveling_test.py --case random --duration 1 --num_envs 8 --headless
```

The observation shape was `(8, 33)`. Observations and rewards remained finite.
With seed 7, this reset sampled speed 0.101 to 0.175 m/s, terrain amplitude
25.7 to 47.9 mm, and Cargo mass 3.71 to 6.20 kg.

### Zero-residual V7.5 equivalence

```bat
python scripts\residual_leveling_test.py --case zero --duration 12 --num_envs 1 --headless
```

At every control step, both the residual and the difference between the final
Lift target and the V7.5 base target were below `1e-8 m`. The resulting metrics
match the V7.5 E baseline:

| Metric | V7.5 E | V7.6-A zero residual |
|---|---:|---:|
| Board roll RMS | 0.1328 deg | 0.1328 deg |
| Board pitch RMS | 0.1067 deg | 0.1067 deg |

The archived V7.5 CSV and the new run also passed a field-specific trajectory
comparison. Across separate GPU PhysX processes, the maximum Board angle
difference was 0.000269 deg and the maximum linear difference was 0.0377 mm.
These are below the declared 0.0005 deg and 0.05 mm tolerances. Cross-process
GPU simulation is not treated as bitwise deterministic.

### Existing task regression

```bat
python scripts\leveling_terrain_test.py --mode geometric --controller_source environment --duration 12 --target_speed 0.10 --log_dir logs\v7_6_a\terrain_regression --headless
```

The existing task remained unchanged: Board roll/pitch RMS was
0.2369/0.1848 deg, support-height RMS was 0.394 mm, all Lifts had 0% saturation,
and Cargo remained supported without drop or tip for the full run.

Python `py_compile` and `git diff --check` passed. Ruff is not installed in the
active Isaac Lab environment.

## Limitations and next step

- AGVs, Lift plates, and terrain interaction remain the established kinematic
  and analytical proxies.
- Cargo/support contact diagnostics are analytical geometry proxies, not PhysX
  contact-sensor measurements.
- Reward scales and PPO hyperparameters are an initial baseline only.
- No PPO training result is claimed in V7.6-A.

The next stage is V7.6-B: train the three-action PPO baseline, compare it with a
forced zero-residual policy on held-out randomized conditions, and report both
stability gains and residual effort/saturation.
