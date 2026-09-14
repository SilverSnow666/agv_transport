# V7.6-E2 residual action magnitude experiment

## Scope and implementation

Starting revision: `d6f7e2e`, branch `v7.0-active-leveling`.
E2 inherits E1 and changes only `residual_action_penalty_scale` from 0.060
to 0.100. The new task is
`Template-Agv-Level-Residual-ActionPenalty-Direct-v0`.
Its environment class, physics, observations, base controller, +/-3 mm
residual limit, domain randomization and other reward terms are inherited.
Its PPO configuration differs from E1 only in experiment directory/name.
E1 and the original V7.6 tasks remain available.

The action term is `-scale * mean(actions**2)` in the existing environment.
E2 tests whether increasing its weight reduces residual limit occupancy
without losing attitude accuracy or increasing actuator motion.

## Validation and training

Six E1/E2 static configuration tests passed. Python compilation and
`git diff --check` passed. Ruff is not installed in `env_isaaclab`.

The 64-environment randomized two-second audit passed, with support gaps
3.000 mm and Lift heights 21.683--41.641 mm. Compared with E1, the action
reward changed from approximately -0.0050 to -0.0083 per step; the other
reported components were unchanged. Total reward was +0.0353 per step.

Zero-residual 12-second regression passed: roll/pitch RMS 0.1328/0.1067 deg.
Maximum differences from the V7.5 reference were 0.000268794596 deg and
0.037711 mm, within the existing tolerances.

Training started from scratch with seed 42, 64 environments and 300 PPO
iterations (19,200 vector steps), using the same PPO settings as E1.
It completed in 1132.23 seconds with exit code zero.

| Training diagnostic | First | Final |
|---|---:|---:|
| Instantaneous reward mean | -0.03537 | 0.53994 |
| Complete-episode return mean | 115.10 | 606.59 |
| Policy standard deviation | 0.22344 | 0.15972 |
| Value loss | 2.43479 | 0.03906 |

The four diagnostic series above are finite. Both evaluated checkpoints
contain 51 finite tensors. The best checkpoint has exactly the same policy,
value and state/value preprocessor tensors as `agent_15360.pt` (240 iterations).
The final checkpoint is `agent_19200.pt` (300 iterations).

Run directory:

```text
logs/skrl/agv_level_residual_action_penalty_direct/
2026-09-14_16-21-35_ppo_torch_v7_6_e2_residual_ppo_action_penalty
```

## Six-condition evaluation

Each checkpoint was evaluated against a fresh E zero-residual baseline for
nominal, fast, rough, heavy, offset and combined. Each pair uses seed 137,
terrain phases (1.17, -2.03), the same initial state and 12 seconds.
All E/F initial-state maximum differences are zero.

Percentages below are arithmetic means of the six per-case percentage
improvements: positive means lower/better than E; negative means higher/worse.
These are not pooled trajectory RMS improvements. Limit occupancy is the
fraction of residual channel samples within 0.0001 mm of +/-3 mm, distinct
from final target clamping at the Lift travel bounds.

| Policy | Roll RMS | Pitch RMS | Board RP angular speed | Lift velocity | Cargo cumulative slip | Mean limit | Worst limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 best (240) | +65.85% | +24.18% | +10.42% | -46.28% | +0.57% | 10.68% | 25.00% |
| E1 final (300) | +70.40% | +28.41% | -1.49% | -49.04% | -0.87% | 15.69% | 29.77% |
| E2 best (240) | +58.54% | +5.81% | +14.72% | -78.10% | -7.55% | 10.34% | 25.46% |
| E2 final (300) | +64.26% | +20.98% | +18.62% | -72.79% | -6.53% | 12.75% | 31.53% |

E2 final per-case results:

| Case | Roll RMS | Pitch RMS | Board RP angular speed | Lift velocity | Cargo cumulative slip | Limit occupancy |
|---|---:|---:|---:|---:|---:|---:|
| nominal | +70.80% | +17.55% | -1.99% | -83.74% | -10.35% | 3.52% |
| fast | +69.26% | +25.68% | +55.49% | -59.51% | +3.26% | 5.74% |
| rough | +53.09% | +23.96% | +24.27% | -67.19% | -14.84% | 28.70% |
| heavy | +68.84% | +8.45% | -18.80% | -99.43% | -5.67% | 3.89% |
| offset | +71.48% | +18.32% | -0.75% | -83.17% | -1.34% | 3.15% |
| combined | +52.11% | +31.89% | +53.51% | -43.69% | -10.26% | 31.53% |

Both E2 candidates fail the intended balanced-policy criteria. Final E2
passes the mean roll >=55% and pitch >=20% targets, but its Lift-velocity
increase is 72.79% (target below 40--45%), mean limit occupancy is 12.75%
(target below 8--10%), and worst rough/combined limit occupancy is 31.53%
(target below 15%). Heavy-case Board angular speed worsens by 18.80%, so a
positive six-case mean does not imply uniformly improved dynamics.

The final E2 Cargo slip increases are small in absolute terms but consistent
enough to retain: nominal 1.595 to 1.760 mm, rough 1.627 to 1.869 mm, combined
3.259 to 3.594 mm. There were no drops or tips.

## Interpretation and decision

Retain E1 best as the current balanced candidate; reject E2 best/final as
replacements. Increasing action penalty to 0.100 did not produce the intended
improvement on high-amplitude terrain. Even average residual RMS did not
decrease: E1 best 1.667 mm versus E2 best 1.736 mm and E2 final 1.796 mm.

Do not infer that E2 best moves the residual faster from its increased Lift
velocity. Its mean residual-rate RMS is 4.886 mm/s versus E1 best 5.431 mm/s;
final E2 is 5.990 mm/s. Actual Lift motion depends on the base feedback,
residual and contact dynamics together. Determining the cause of the larger
Lift motion requires component-wise trajectory analysis.

This is one training seed and one evaluation phase pair per variant. The
experiment holds configuration fixed, but does not prove a general monotonic
effect of action penalty. Small E baseline differences across separate runs
are present despite matching initial states (for example heavy angular-speed
RMS varies around 0.00044--0.00047 rad/s). Each reported improvement therefore
uses its own paired E run. GPU/contact numerical variation and training
variance limit claims about small percentage differences.

Before escalating to E3 or long training, inspect base target, feedback and
residual trajectories on rough/combined and check whether the observed
tradeoff reproduces across training seeds. E3 with a still larger magnitude
penalty is a hypothesis, not an established remedy from these results.

Each E2 evaluation directory contains 8,640 trajectory rows, 12 summary rows
and 72 metric comparisons. Every run has 720 samples, with no missing or
non-finite trajectory values, early termination, Board/Cargo drop/tip,
support-proxy loss or final target saturation. Support/contact metrics remain
analytical proxies, not PhysX contact-sensor measurements.

## Reproduction

Run from the repository root in `env_isaaclab`:

```bat
python -u scripts\residual_leveling_test.py --task Template-Agv-Level-Residual-ActionPenalty-Direct-v0 --case random --duration 2 --num_envs 64 --log_dir logs\v7_6_e2\reward_audit --headless
python -u scripts\residual_leveling_test.py --task Template-Agv-Level-Residual-ActionPenalty-Direct-v0 --case zero --duration 12 --num_envs 1 --log_dir logs\v7_6_e2\zero_12s --headless
python -u scripts\skrl\train.py --task Template-Agv-Level-Residual-ActionPenalty-Direct-v0 --num_envs 64 --headless --max_iterations 300
```

Evaluation commands use the recorded training directory (a rerun creates a
new timestamped directory):

```bat
set E2_RUN=logs\skrl\agv_level_residual_action_penalty_direct\2026-09-14_16-21-35_ppo_torch_v7_6_e2_residual_ppo_action_penalty
python -u scripts\residual_checkpoint_eval.py --task Template-Agv-Level-Residual-ActionPenalty-Direct-v0 --checkpoint "%E2_RUN%\checkpoints\best_agent.pt" --case all --duration 12 --log_dir logs\v7_6_e2\checkpoint_best --headless
python -u scripts\residual_checkpoint_eval.py --task Template-Agv-Level-Residual-ActionPenalty-Direct-v0 --checkpoint "%E2_RUN%\checkpoints\agent_19200.pt" --case all --duration 12 --log_dir logs\v7_6_e2\checkpoint_300 --headless
```
