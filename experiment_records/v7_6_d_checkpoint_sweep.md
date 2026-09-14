# V7.6-D.1 formal PPO checkpoint sweep

## Scope

The 128-environment, 1500-iteration training run produced checkpoints every
150 iterations. The first sweep compares iterations 300, 450 and 600 because
the formal `best_agent.pt` was written near iteration 600. Every checkpoint is
evaluated against E (geometric + feedback, zero residual) with the same six
conditions, seed, terrain phases and 12-second duration used by V7.6-C.

Training run:

```text
logs/skrl/agv_level_residual_direct/
2026-09-11_16-02-27_ppo_torch_v7_6_b_residual_ppo_baseline
```

Commands:

```bat
python scripts\residual_checkpoint_eval.py --checkpoint "...\checkpoints\agent_19200.pt" --case all --duration 12 --log_dir logs\v7_6_d\checkpoint_300 --headless
python scripts\residual_checkpoint_eval.py --checkpoint "...\checkpoints\agent_28800.pt" --case all --duration 12 --log_dir logs\v7_6_d\checkpoint_450 --headless
python scripts\residual_checkpoint_eval.py --checkpoint "...\checkpoints\agent_38400.pt" --case all --duration 12 --log_dir logs\v7_6_d\checkpoint_600 --headless
```

## Six-condition mean results

Positive percentages mean an improvement over E; negative percentages mean
that F is worse. `At limit` counts PPO residual channel samples at +/-3 mm,
not final Lift travel saturation.

| Checkpoint | Roll RMS | Pitch RMS | Board RP angular speed | Lift velocity | Cargo cumulative slip | Residual RMS | Mean at limit | Worst case at limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 (`agent_19200`) | +67.45% | +44.58% | -14.84% | -86.71% | -3.13% | 1.91 mm | 21.92% | 36.30% |
| 450 (`agent_28800`) | +73.49% | +48.60% | -44.03% | -145.29% | -3.26% | 2.11 mm | 30.12% | 45.14% |
| 600 (`agent_38400`) | +73.35% | +56.40% | -38.68% | -143.48% | -1.37% | 2.11 mm | 30.89% | 46.44% |
| `best_agent` | +73.35% | +56.40% | -38.68% | -143.48% | -1.37% | 2.11 mm | 30.89% | 46.44% |

All three periodic checkpoints strongly improve static Board attitude. That
improvement is purchased with substantially more Lift motion and frequent
residual limiting. The dynamic tradeoff is already present at iteration 300,
then becomes more pronounced at 450/600. Cargo cumulative slip changes remain
small on average, but do not compensate for the control-activity regression.

The formal `best_agent.pt` and `agent_38400.pt` files have different whole-file
SHA-256 hashes because checkpoint serialization includes more than inference
weights. Direct tensor comparison shows exact equality (maximum absolute
difference 0.0) for policy, value, state preprocessor and value preprocessor.
Their six-condition aggregate metrics are consequently identical. There is no
reason to evaluate both as independent candidates.

## Selection decision

The provisional balanced-model criteria from the previous stage were:

```text
roll improvement > 50%
pitch improvement > 30%
Board angular-speed degradation < approximately 10%
Lift-velocity increase < 50%
residual at-limit fraction < 5-10%
no drop/tip/support loss
```

All candidates pass attitude and safety criteria. None passes the dynamic or
control-effort criteria. Iteration 300 is the least aggressive candidate but
still increases mean Lift velocity by 86.71%, worsens mean Board roll/pitch
angular speed by 14.84%, and spends 21.92% of residual channel samples at the
limit. Therefore no checkpoint in the 300-600 interval is accepted as the
balanced final policy.

This trend begins early enough that it should not be described solely as late
training overfit. The current reward prioritizes Board angle much more strongly
than angular velocity, Lift velocity, action magnitude and action rate. The
next controlled experiment should be V7.6-E reward refinement, preserving the
environment, observations, action limit, domain randomization and base E
controller while increasing smoothness/control-effort penalties. A new task or
versioned reward configuration should preserve the V7.6 baseline rather than
silently replacing it.

## Validation

Each output directory contains 8,640 trajectory rows, 12 summary rows and 72
metric comparisons. Every E/F run contains exactly 720 samples. Across all 36
runs there are no missing or non-finite values, initial-state mismatches, early
terminations, Board/Cargo drops or tips, or analytical support-proxy losses.
Final Lift target saturation is zero. Support/contact values remain analytical
proxies, not PhysX contact-sensor measurements.

CSV outputs are under:

```text
logs/v7_6_d/checkpoint_300/
logs/v7_6_d/checkpoint_450/
logs/v7_6_d/checkpoint_600/
```
