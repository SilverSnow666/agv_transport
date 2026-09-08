# V7.3 Board-attitude feedback leveling

## Purpose

V7.2/V7.1.1 showed that geometric support-height feedforward substantially
reduces Board roll and pitch, but it does not directly regulate the residual
Board attitude. V7.3 adds a bounded roll/pitch PD correction and evaluates its
incremental contribution without changing the terrain, AGV motion, Lift
geometry, Cargo, or virtual-carry settings.

## Compared cases

All cases use one environment, the same deterministic terrain and reset, the
same three support offsets, `target_speed=0.10 m/s`, and `duration=12 s`.

- C / `no_leveling`: all Lift heights fixed at the 30 mm neutral position.
- D / `geometric`: equalize the three physical support-top world heights.
- E / `feedback`: D plus Board-local roll/pitch PD feedback.

These C/D/E labels extend the earlier A/B/C/D Board-support ablation. They do
not replace or redefine its four cases.

## Controller

The Board quaternion transforms angular velocity from world coordinates into
the Board frame. The controller computes

```text
roll_cmd  = -(Kp_roll  * board_roll  + Kd_roll  * board_local_wx)
pitch_cmd = -(Kp_pitch * board_pitch + Kd_pitch * board_local_wy)

delta_h_i = -support_x_i * tan(pitch_cmd)
            + support_y_i * tan(roll_cmd)
```

The three `delta_h_i` values are made zero-mean, uniformly limited, low-pass
filtered, converted from world vertical displacement to each AGV's local Lift
axis, and added to the geometric target.

Validated defaults:

```text
roll/pitch Kp       = 0.80
roll/pitch Kd       = 0.08 s
maximum correction = 8 mm per support
filter alpha        = 0.25
```

Zero-mean correction preserves average Board height. Uniform limiting
preserves the commanded support-plane orientation.

## Reproduction

```bat
python scripts\leveling_terrain_test.py --mode all --duration 12 --target_speed 0.10 --log_dir logs\v7_3\baseline_default --headless
```

Individual gains can be overridden using `--feedback_roll_kp`,
`--feedback_pitch_kp`, `--feedback_roll_kd`, `--feedback_pitch_kd`,
`--feedback_max_correction_mm`, and `--feedback_filter_alpha`.

## Results

| Metric | C fixed Lift | D geometric | E geometric + feedback |
|---|---:|---:|---:|
| Board roll RMS (deg) | 0.4076 | 0.2369 | 0.1328 |
| Board pitch RMS (deg) | 0.2518 | 0.1848 | 0.1067 |
| Max abs roll (deg) | 0.8425 | 0.4917 | 0.2561 |
| Max abs pitch (deg) | 0.5979 | 0.3704 | 0.2157 |
| Board roll/pitch angular-speed RMS (rad/s) | 0.001218 | 0.000131 | 0.000050 |
| Board vertical-acceleration RMS (m/s^2) | 0.004224 | 0.002423 | 0.000667 |
| Support-height RMS (mm) | 2.550 | 0.394 | 0.957 |
| Lift velocity RMS (mm/s) | 0.000 | 1.997 | 2.241 |
| Feedback correction RMS (mm) | 0.000 | 0.000 | 0.925 |
| Cargo relative XY RMS (mm) | 0.00148 | 0.00145 | 0.00139 |
| Cargo maximum relative XY (mm) | 0.00479 | 0.00502 | 0.00724 |
| Cargo cumulative slip (mm) | 0.07321 | 0.07430 | 0.07747 |
| Cargo angular-speed RMS (rad/s) | 0.01160 | 0.01210 | 0.01263 |
| Cargo support/contact proxy | 100% | 100% | 100% |
| Lift saturation | 0% | 0% | 0% |
| Cargo dropped / tipped | false / false | false / false | false / false |

Relative to D, E improves roll RMS by 43.94%, pitch RMS by 42.25%, maximum
roll by 47.91%, maximum pitch by 41.78%, roll/pitch angular-speed RMS by
61.82%, and vertical-acceleration RMS by 72.47%.

Support-height RMS rises from 0.394 mm to 0.957 mm because E intentionally
commands unequal support heights to counter the measured Board attitude. It
is therefore not, by itself, a failure of feedback leveling. The incremental
Lift velocity cost is 0.244 mm/s (12.20%).

## Limitations

- Cargo support/contact is an analytical center/footprint/face-gap proxy, not a
  PhysX contact-sensor measurement.
- The result is a deterministic one-environment baseline on one terrain seed;
  robustness across seeds, payload masses, speeds, and disturbances remains to
  be evaluated.
- `board_roll_rate` and `board_pitch_rate` retain the legacy world-frame
  components for CSV compatibility. Both the PD derivative term and the new
  `board_rp_angular_speed` aggregate use Board-local angular velocity.
