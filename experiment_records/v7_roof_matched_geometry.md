# V7 AGV roof-matched collision geometry — 2026-09-08

The AGV collision box now measures **0.55 × 0.42 × 0.10 m**. Its flat-ground
center is at 0.05 m and its top is at 0.10 m, within 0.061 mm of the previously
measured 0.100061 m roof of the unmodified, scaled iwhub visual asset.
The code that raised the native visual lift by 59.939 mm has been removed.
The visual root translation compensates the lower rigid-body origin, preserving
the original vehicle height on flat ground. No internal vehicle mesh is moved.

All four cases use this geometry. Lift and Board initial positions follow the
new AGV top; their flat-ground heights are 60 mm lower. Cargo initialization
and the Board drop threshold follow the same height change. Lift stroke,
support footprint, terrain sampling, speed and virtual-assistance settings are
unchanged. This supersedes the original experiment's 160 mm proxy height;
results from different geometries must not be combined as one ablation.

## Reproduce

Run from the project directory using the Isaac Lab Python environment:

```powershell
python scripts/leveling_ablation_test.py --case all --duration 12 --target_speed 0.10 --log_dir logs/v7_ablation/roof_matched_100mm --headless
python scripts/leveling_terrain_test.py --mode geometric --duration 12 --target_speed 0.10 --log_dir logs/v7_2/roof_matched_100mm --headless
```

## Board-only ablation, seed 0, 12 seconds, 0.10 m/s

| Case | Roll RMS (deg) | Pitch RMS (deg) | Vertical acceleration RMS (m/s²) | Z range (mm) | Forward displacement (m) | Contact-loss proxy fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: direct AGV support, no assistance | 0.9678 | 5.8176 | 0.4467 | 83.611 | 0.006 | 80% |
| B: fixed Lift, no assistance | 0.5377 | 9.0570 | 0.4772 | 120.480 | 0.016 | 80% |
| C: fixed Lift, virtual carry/damping | 0.4066 | 0.2506 | 0.000429 | 7.557 | 1.114 | 0% |
| D: C plus geometric leveling | 0.2419 | 0.1846 | 0.000392 | 6.648 | 1.114 | 0% |

A/B largely fail to travel with the kinematic AGVs. Their full-run attitude
statistics include support loss and falling, so A→B does not measure only
steady supported vibration. B→C includes the assistance's transport effect
as well as damping. C→D reduces roll RMS by 40.52% and pitch RMS by 26.32%;
Z jerk RMS increases from 0.00411 to 0.00714 m/s³.

Contact/support fields remain analytical proxies, not force-sensor readings.
The collider remains a rectangular approximation with its existing footprint;
this change aligns its roof height, not every contour of the visual mesh.
AGV terrain following remains kinematic, and C/D retain explicit virtual
carry/damping. Visual alignment does not establish full physical realism.

## Validation

- Existing 12 s geometric terrain test: Board roll/pitch RMS 0.2369/0.1848 deg,
  support-height RMS 0.396 mm. Cargo contact proxy 100%, no drop or tip.
- Existing 8 s disturbance test: support RMS 11.641 mm before compensation,
  0.000 mm after; Board tilt 1.680 deg before, 0.000 deg after.
- Static checks at 0 and 100 mm stroke completed with finite states and
  29/129 mm roof-to-head visual rod lengths. Neutral 30 mm was exercised in
  the four-case runs.
- A/B viewport images inspected: original vehicle shape restored, direct
  support at the original roof, and the external Lift connected to that roof.
  Vulkan viewport startup stalled on this host; a Direct3D retry succeeded
  using `--kit_args='--/app/vulkan=false'`.
- Python compilation and `git diff --check` passed. Ruff was not installed
  in the Isaac Lab environment.

The simplified collider height change also shifts absolute observations;
old trained checkpoints should be evaluated again before reuse.
