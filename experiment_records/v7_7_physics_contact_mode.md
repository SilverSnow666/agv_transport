# V7.7 Contact-only Board physics validation mode

## Motivation

The existing V7 residual-RL tasks intentionally use virtual planar carry and
payload damping to compensate for pose-written kinematic supports. Those terms
are useful for controller development, but they can keep the Board horizontal
after a physical support has departed. This versioned task provides a separate
physics-validation mode and does not change any existing training task.

Task ID:

```text
Template-Agv-Level-PhysicsContact-Direct-v0
```

## Isolation guarantees

The task sets all of the following to false or zero:

```text
enable_virtual_friction_carry
virtual_friction_coupling
slip_correction_gain
payload_vertical_damping
payload_roll_pitch_damping
payload_yaw_damping
payload_yaw_alignment_gain
payload_yaw_alignment_coupling
max_payload_yaw_rate
```

Board and Cargo motion is therefore produced only by gravity and PhysX
contacts. The Payload has a real filtered `ContactSensor` for Lift1, Lift2,
Lift3 and Cargo. Analytical support counts remain logged separately and are
explicitly named as proxies.

The three Lift plates are dynamic rigid bodies with gravity disabled and a
bounded velocity servo that tracks the prescribed AGV/Lift pose. This is an
ideal actuator model, but unlike the old pose-written kinematic plates it
transmits tangential motion through PhysX friction. No Board velocity, pose or
attitude is overwritten.

## Verification runs

### Contact-only transport

```bat
python scripts\leveling_physics_contact_demo.py ^
  --duration 3 ^
  --settle_duration 1 ^
  --target_speed 0.10 ^
  --terrain_amplitude 0.030 ^
  --controller geometric_feedback ^
  --log_path logs\v7_physics_contact\dynamic_transport_3s.csv ^
  --headless
```

Result:

| Metric | Result |
|---|---:|
| Initial settled Lift contacts | 3 |
| Minimum real Lift contacts | 3 |
| Board forward displacement | 0.3008 m |
| Expected carrier displacement | about 0.300 m |
| Maximum Board roll/pitch magnitude | 0.2237 deg |
| Virtual carry/stabilization active | never |

This confirms that the moving support plates carry the Board through actual
contact friction rather than the virtual velocity rewrite.

### Front support departure

The diagnostic fault moves AGV1 two metres forward after two seconds while
leaving AGV2, AGV3, Board and Cargo untouched. Controller targets are frozen
after settling so the two remaining supports cannot compensate for the removed
front support.

```bat
python scripts\leveling_physics_contact_demo.py ^
  --duration 7 ^
  --settle_duration 1 ^
  --target_speed 0 ^
  --terrain_amplitude 0.050 ^
  --controller geometric_feedback ^
  --force_front_support_loss_at 2 ^
  --log_path logs\v7_physics_contact\dynamic_support_loss_7s.csv ^
  --headless
```

Result:

| Metric | Before loss | After loss |
|---|---:|---:|
| Real Lift contacts | 3 | 2 |
| Board pitch | about 0.23 deg | 7.74 deg final |
| Maximum Board tilt | - | 7.78 deg |
| Maximum Cargo relative XY displacement | - | 0.057 mm |

The Board tips forward until its front edge reaches the flat ground plane. The
Cargo does not slide appreciably because the configured static friction is
0.80 and a 7.7 degree slope is well below its static-friction angle. This is an
expected physical outcome, not hidden stabilization.

## Visual commands

Normal contact-only transport:

```bat
python scripts\leveling_physics_contact_demo.py --duration 20 --target_speed 0.10 --terrain_amplitude 0.050 --real_time
```

Visible support-loss demonstration:

```bat
python scripts\leveling_physics_contact_demo.py --duration 10 --target_speed 0 --terrain_amplitude 0.050 --force_front_support_loss_at 3 --real_time
```

## Known modelling boundary

AGV terrain pose is still prescribed analytically and the white sinusoidal
terrain mesh is visual-only. This mode makes the Board, Cargo and Lift contact
chain physical; it is not yet a full wheel-terrain vehicle dynamics model. A
future full-physics vehicle task would require articulated/dynamic AGVs and a
collidable terrain mesh.
