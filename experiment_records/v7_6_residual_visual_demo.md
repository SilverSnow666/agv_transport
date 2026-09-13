# V7.6 single-environment visible-terrain PPO demo

## Scope

The latest discussion (2026-09-13) concerned a trained PPO policy appearing to
drive over a flat floor. Residual training intentionally disables the terrain
mesh, while AGV poses follow per-environment analytical terrain. This change
adds an **opt-in playback mode**, not a new training task or a physics upgrade.

Changed files: `scripts/skrl/play.py`, `scripts/skrl/residual_demo.py`, and
`scripts/skrl/test_residual_demo.py`. No environment, reward, observation,
network, checkpoint or training configuration is changed.

## Usage

From `D:\Omniverse\Project\agv_transport`, in the Isaac Lab Python environment:

```bat
python scripts\skrl\play.py --task Template-Agv-Level-Residual-Direct-v0 --num_envs 1 --checkpoint "D:\Omniverse\Project\agv_transport\logs\skrl\agv_level_residual_direct\2026-09-11_16-02-27_ppo_torch_v7_6_b_residual_ppo_baseline\checkpoints\best_agent.pt" --residual-demo --real-time
```

- Omit `--headless` to open the GUI. Close the window to stop, or add
  `--duration 12` for a bounded run in simulation seconds.
- Add `--zero-residual` for E (same fixed scenario, zero Lift residual).
  Without it, F uses the loaded PPO policy's deterministic mean action.
- Defaults: speed 0.10 m/s, amplitude 30 mm, phases 1.17 / -2.03 radians,
  current configured wavelengths (1.40 / 1.10 m), Cargo 4 kg at zero XY offset.
- Optional `--demo-speed`, `--demo-amplitude` (meters, 0..0.050),
  `--demo-phase-x`, `--demo-phase-y`. These change the actual analytical
  disturbance, not merely the picture. They are not training randomization.
- Add `--video --video_length 720` to record 12 seconds. Demo videos go into
  a timestamped `videos/residual_demo/` subfolder of the selected checkpoint's
  run, leaving previous play videos untouched.
- Demo accepts only the residual task, one environment, and the torch backend.
  Normal playback without `--residual-demo` retains its existing configuration.

## Consistency safeguards

The scalar configuration used to build USD vertices and the tensor parameters
used for AGV terrain sampling share amplitude, phases and wavelengths. Visual
height scale is exactly 1; vertical visual offset is zero. Existing height
colors distinguish crests and troughs. Renderer subdivision is disabled to
retain the sampled triangle surface.

Instead of switching `residual_domain_randomization=False`, the demo freezes
all randomization ranges to single values. This retains the A.1 terrain-aware
reset and Cargo mass/inertia path; support clearance remains 3 mm. Every reset
checks actual USD vertices against the runtime analytical function and checks
support geometry. Initial and post-reset observations/actions must be finite.

The legacy visible-terrain option moves the physical ground plane to -80 mm.
This demo overrides that displacement to zero and hides only the ground's USD
visibility. Runtime checks confirm the ground collision is still enabled and
its root remains at the training height. The terrain mesh adds no collision.

## Verification (2026-09-13)

- Loaded the explicitly selected `2026-09-11_16-02-27.../best_agent.pt`.
- PPO headless 18 seconds: 1,080 steps, exit 0, including automatic episode
  reset. Both reset checks passed. 6,720 vertices, maximum height discrepancy
  0.000028 mm; support clearance 3.000 mm.
- PPO rendered 2 seconds: 120 steps, exit 0; ground collider enabled check
  passed. Inspected a frame from the recorded MP4: visible colored hills and
  troughs, AGVs, yellow Board and blue Cargo all present.
- Standard-library config tests cover matching fixed ranges, controller/limit
  preservation, and rejection of invalid task, environment count, backend,
  amplitude, speed and non-finite phases.
- Zero-residual E demo, 50 mm amplitude and phases 0.25 / 0.45: 120 steps,
  exit 0, vertex discrepancy 0.000054 mm and reset clearance 3.000 mm.
- Ordinary playback without the demo flag: two-second regression, exit 0.
- Python syntax and `git diff --check` passed. Ruff is not installed.

Local artifacts (ignored by git): `logs/residual_demo_smoke.log`,
`logs/residual_demo_render.log`, `logs/residual_demo_preview.png`; recorded MP4
under the selected run's `videos/residual_demo/20260913_212644/`.

The initial sandbox run could not load Isaac Sim's official ground USD asset;
the runtime validations were rerun with asset/cache access. Existing Isaac Sim
warnings about Fabric point instancers and GPU memory interfaces remain.

## Limitations

This is still kinematic AGV motion over an analytical disturbance, not physical
wheel/terrain collision. The finite-resolution visual mesh approximates the
continuous surface between vertices. The unchanged flat catch plane can catch
fallen objects; the visible terrain is not their collision surface. Residual
motion remains bounded at +/-3 mm and is intentionally subtle. Visual playback
does not establish PPO superiority: use the E/F quantitative evaluator for that.
