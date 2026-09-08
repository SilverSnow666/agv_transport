# V7 native iwhub Lift visual

## Change

- Removed the temporary visual-only cylinder actuator and cuboid lift head.
- Reused the iwhub asset's native visible Lift branch at
  `/Visual/lift/Lift`.
- Kept the sibling `/Visual/lift/Collision` branch and the V7 hidden physical
  Lift plates untouched.
- Each native Lift visual is translated from its measured source top height
  by the corresponding `lift_height`, so its top meets the nominal Board
  underside.
- Case A hides the native Lift visual together with parking the physical Lift;
  Cases B/C/D show and independently drive all three native Lift visuals.

The source asset exposes the complete visible Lift as one mesh, not as
separate kinematic links.  At the normal 30 mm working height the bottom of
that mesh still overlaps the vehicle body.  Large diagnostic strokes near the
100 mm physical limit can visually separate the rigid mesh from the body; no
fake linkage or hidden damping was added to conceal this source-asset limit.

## Validation

Commands:

```powershell
python scripts/leveling_static_test.py --duration 2 --lift_height_mm 30 \
  --screenshot_path outputs/native_iwhub_lift_30mm_low.png \
  --kit_args='--/app/vulkan=false'

python scripts/leveling_ablation_test.py --case all --duration 0.5 \
  --settle_duration 0.1 --target_speed 0.10 \
  --log_dir logs/v7_ablation/native_iwhub_smoke --headless

python scripts/leveling_terrain_test.py --mode geometric --duration 12 \
  --target_speed 0.10 --log_dir logs/v7_2/native_iwhub_lift --headless
```

Terrain regression (720 samples):

- Board roll RMS: 0.236868 deg
- Board pitch RMS: 0.184836 deg
- Support-height RMS: 0.395629 mm
- Cargo contact: 100%
- Cargo dropped/tipped: false/false

These values match the preceding roof-matched physical baseline, confirming
that the change is visual-only.
