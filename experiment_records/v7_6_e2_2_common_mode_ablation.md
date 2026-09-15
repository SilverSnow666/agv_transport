# V7.6-E2.2: inference-only common-mode constraint ablation

Date: 2026-09-15. Starting HEAD `94127f9`, branch `v7.0-active-leveling`.

## Outcome

The bounded zero-mean E1 policy nearly eliminates common Lift motion and
reduces total Lift velocity. It does not retain all original E1 performance:
roll RMS increases by about 9--11%, and Board RP angular-velocity RMS worsens
in both tested cases relative to E1. Pitch and Cargo slip change much less.

Keep the original E1 best checkpoint as the current candidate. Do not silently
enable the projection in training or deployment. No reward, environment,
action-space, physical parameters or checkpoint weights were changed; no
training was run. Given the mixed result, stop at the requested initial two
conditions instead of automatically extending to all six or starting training.

## Three arms and bound handling

All arms use the E1 task `Template-Agv-Level-Residual-Smooth-Direct-v0`:

- E: Geometric + Feedback, zero residual.
- E1: the original E1 best checkpoint, deterministic clipped policy mean.
- E1-ZM: same checkpoint; inference-only centered and bounded action mapping.

The baseline policy already clips each mean action to [-1,1]. Denote these
three actions by `a`. The applied E1-ZM mapping is:

```text
c = a - mean(a)
s = 1 / max(1, max(abs(c)))
a_applied = s * c
residual = 3 mm * a_applied
```

Subtracting the mean alone can exceed the original +/-3 mm bound. For example,
`[3,3,-3] mm` centers to `[2,2,-4] mm`. Independent clipping would reintroduce
a nonzero mean. Uniform radial scaling instead produces `[1.5,1.5,-3] mm`,
preserving zero mean and differential direction while reducing differential
amplitude. This is NOT an exact pure-common-component removal at those steps.
The choice is explicit and the scale and pre-projection clipped policy actions
are saved every step; no hidden environment clipping is used to enforce it.

This diagnostic changes the closed-loop observation trajectory as well as
the applied action mapping. Unchanged weights do not mean unchanged future
policy outputs. The result cannot separate common-mode removal from radial
attenuation and closed-loop adaptation; nor can it establish whether a policy
trained with a zero-mean mapping would perform better.

## Reproducible setup

Checkpoint (unchanged SHA-256 recorded in the manifest):

`logs/skrl/agv_level_residual_smooth_direct/2026-09-14_15-10-40_ppo_torch_v7_6_e1_residual_ppo_smooth_reward/checkpoints/best_agent.pt`

| Case | Speed | Terrain amplitude | Cargo mass | Cargo XY offset |
|---|---:|---:|---:|---:|
| rough | 0.10 m/s | 50 mm | 4 kg | 0, 0 mm |
| combined | 0.18 m/s | 50 mm | 8 kg | 30, 0 mm |

One environment; 12 s; seed 137; phase `(1.17,-2.03)`; 60 Hz control and
120 Hz physics. Each case uses one independent Isaac Sim process running
E, E1 and E1-ZM in sequence, with full matched resets and frozen inference
normalizers. All three initial-state signatures match exactly for both cases.
Six runs, 720 intervals each, 4,320 trajectory rows. No nonfinite values or
missing intervals. Existing physical auxiliary coupling is identical in all
three arms; this is not a new physical-model validation.

Results directory: `logs/v7_6_e2_2/stress_12s/`.

## Full 12-second results

Lift speed is the existing final-physics-substep velocity metric. New interval
displacement velocities are saved separately and show the same relative
reduction here. Board angular speed is the RMS of the logged RP angular-velocity
vector, not a finite difference of Euler angles.

| Case / arm | Roll RMS (deg) | Pitch RMS (deg) | Board RP speed (rad/s) | Lift speed (mm/s) | Cargo cumulative slip (mm) |
|---|---:|---:|---:|---:|---:|
| rough E | 0.255776 | 0.122367 | 0.00017230 | 2.33293 | 1.60303 |
| rough E1 | 0.117557 | 0.090069 | 0.00013084 | 3.28120 | 1.62218 |
| rough E1-ZM | 0.129997 | 0.090377 | 0.00019769 | 2.92799 | 1.61754 |
| combined E | 0.257607 | 0.136885 | 0.00135316 | 3.89241 | 3.25902 |
| combined E1 | 0.123993 | 0.090799 | 0.00077329 | 5.12763 | 3.14514 |
| combined E1-ZM | 0.135389 | 0.092213 | 0.00083325 | 4.83563 | 3.10608 |

Relative to E1 (not relative to E):

| Case | Roll RMS increase | Pitch RMS increase | Board RP speed increase | Lift speed reduction | Cargo slip reduction |
|---|---:|---:|---:|---:|---:|
| rough | 10.58% | 0.34% | 51.10% | 10.76% | 0.29% |
| combined | 9.19% | 1.56% | 7.75% | 5.69% | 1.24% |

The rough RP speed percentage has a small denominator: the absolute increase
is about 0.0000669 rad/s. Do not imply that it is a large visible oscillation,
or treat its exact percentage as statistically established from one run.
The increase still remains after omitting startup (see CSV).

Compared to E, E1-ZM still improves roll/pitch RMS by 49.18%/26.14% on rough
and 47.44%/32.64% on combined. It has not lost the entire PPO benefit; it is
a tradeoff, not an outright unsafe controller or a demonstrated superior one.

## Common motion and projection activity

Full-window common velocity is the mean of the three interval displacement
velocities. Mean Lift height is a local actuator-height measure, not Board
world Z.

| Case / arm | Common residual RMS (mm) | Common Lift velocity RMS (mm/s) | Mean Lift-height range (mm) | Residual at limit |
|---|---:|---:|---:|---:|
| rough E1 | 0.29281 | 1.43751 | 9.98814 | 20.65% |
| rough E1-ZM | <0.000001 | 0.01254 | 0.12178 | 13.10% |
| combined E1 | 0.30803 | 1.51315 | 6.14217 | 25.00% |
| combined E1-ZM | <0.000001 | 0.01266 | 0.11788 | 14.91% |

Maximum absolute applied common residual is below 2e-7 mm; each applied
component stays within +/-3 mm, allowing float rounding. Common Lift velocity
is reduced by 99.13%/99.16%; total velocity falls much less because differential
motion remains. Remaining small common motion is compatible with tilted
actuator axes and the base controller's world-height versus local-axis mapping;
zero local residual mean does not impose an exact world-height constraint.

Radial scaling occurs on 39.31% of rough steps and 44.72% of combined steps.
Minimum scale is 0.81597/0.80185 (at most about 18.4%/19.8% attenuation at a
given step); whole-window mean scale is 0.96645/0.95537. This confound is not
negligible. Differential residual RMS is about 6.85%/7.17% smaller than E1's,
but that difference also includes the changed observation trajectory.

At-limit percentages under the new mapping have a different interpretation:
uniform scaling can place a channel exactly on its bound. Do not compare the
occupancies alone as evidence of learned action restraint. Final Lift-target
clamping/saturation remains zero in all six runs.

## Startup-sensitive vertical metrics

Both full and command-time >=1 s windows are exported. Rate RMS excludes the
first transition inside each window; Cargo slip for the latter is accumulated
only over that window. The full vertical-acceleration RMS is roughly
0.486--0.492 m/s^2 for all arms and is dominated by startup.

After excluding the first second:

| Case / arm | Board vertical velocity RMS (m/s) | Vertical acceleration RMS (m/s^2) |
|---|---:|---:|
| rough E | 0.00021863 | 0.00080734 |
| rough E1 | 0.00015070 | 0.00050405 |
| rough E1-ZM | 0.00015479 | 0.00063756 |
| combined E | 0.00064266 | 0.00721095 |
| combined E1 | 0.00028024 | 0.00482864 |
| combined E1-ZM | 0.00029893 | 0.00467999 |

Thus suppressing common Lift motion does not guarantee lower logged Board
vertical velocity or acceleration. Retain the mixed measurements rather than
inferring a benefit solely from the almost-flat mean Lift-height trace.

## Verification and files

- Projection unit tests: zero mean, bound handling, differential preservation
  when feasible, radial scaling when not, no input mutation, idempotence,
  invalid input rejection, and a 10,000-row random batch.
- 15 total projection/diagnostic/E1/E2 configuration tests passed.
- 0.3 s three-arm rough smoke passed. The original E/E1 arms match all shared
  numeric fields of the E2.1 smoke exactly (36 rows, maximum difference 0).
- Full six-run finite-value, sample-count and initial-state checks passed.
- No Board/Cargo drop/tip, early termination, support-proxy loss or final
  target saturation. Contact/support remains an analytical proxy, not a sensor.
- Python compilation and `git diff --check` passed. Ruff is not installed.
- PNG/PDF comparison figure rendered and visually checked.

Added `scripts/residual_common_mode.py`,
`scripts/residual_common_mode_ablation.py`,
`scripts/test_residual_common_mode.py`, and this record. The existing evaluator
has an opt-in `--common_mode_ablation` flag; without it its E/F workflow remains
unchanged. The flag automatically enables control-decomposition capture and
adds `F_ppo_zm`. No new environment/task is registered.

Use `ablation_summary.csv` and `ablation_comparisons.csv` for the three-arm
comparison. The nested legacy `residual_checkpoint_improvements.csv` continues
to describe E versus original F only; nested raw trajectories and physical
summaries include all three arms. Manifest and source/checkpoint SHA-256 files
record provenance. Logs are local, Git-ignored artifacts. Unrelated
`scriptslift_motion_demo.py` is untouched.

## Reproduction and next decision

From the repository root in `env_isaaclab`:

```bat
python scripts\residual_common_mode_ablation.py --duration 12 --log_dir logs\v7_6_e2_2\new_stress_run --headless
python scripts\residual_common_mode_ablation.py --log_dir logs\v7_6_e2_2\stress_12s --analyze_only
python -m unittest scripts.test_residual_common_mode scripts.test_residual_control_metrics scripts.test_residual_smooth_config scripts.test_residual_action_penalty_config
```

Default `--case stress` runs rough/combined; `--case all` supports all six but
was deliberately NOT run here. `--checkpoint`, `--seed`, `--phase_x/y`,
`--duration`, `--warmup` are configurable. Use an empty output directory for a
new simulation; existing outputs are rejected unless `--analyze_only` is used.

This single-seed, single-phase projected-policy test supports controlling
unintended common-height motion as a useful design direction, but does not
justify replacing E1 immediately or concluding the common mode is entirely
unnecessary. Before training changes, a separate soft common-mode or
mean-height-control ablation could test a less restrictive compromise; its
bound handling must also be explicit. Alternatively, compare projection
schemes to separate attenuation from common-mode removal. Confirm any candidate
across phases/seeds before treating small dynamic differences as reliable.
