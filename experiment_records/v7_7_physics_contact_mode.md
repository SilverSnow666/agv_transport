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

The contact-validation task also replaces the controller-development task's
historical `0.28 x 0.28 m` Lift proxy with a `0.18 x 0.16 x 0.015 m` contact
pad matching the configured visible Lift head. Its static/dynamic friction is
`0.80 / 0.65`. This removes the hidden oversized platform that previously let
the Board bridge widely tilted supports. There is no ball joint, fixed joint,
attachment, suction or other Board constraint: gravity, collision and friction
are the complete mechanical coupling between each Lift and the Board.

Because each Lift head is mechanically mounted to an AGV in the real system,
the contact task uses a stiff bounded velocity servo between the prescribed AGV
pose and its Lift rigid body. This is an AGV-to-Lift actuator model, not a
Lift-to-Board constraint. It reduced the normal-run mean AGV/Lift attitude
tracking error to about `0.65 deg` without preventing Board tip or support loss.

## Verification runs

### Contact-only transport

```bat
python scripts\leveling_physics_contact_demo.py ^
  --duration 5 ^
  --settle_duration 1 ^
  --target_speed 0.10 ^
  --terrain_amplitude 0.050 ^
  --controller geometric_feedback ^
  --log_path logs\v7_physics_contact\final_feedback_5s.csv ^
  --headless
```

Result:

| Metric | Neutral | Geometric + feedback |
|---|---:|---:|
| Roll RMS | 0.45 deg | 0.14 deg |
| Pitch RMS | 0.44 deg | 0.10 deg |
| Maximum Board roll/pitch magnitude | 0.68 deg | 0.23 deg |
| Minimum real Lift contacts | 3 | 3 |
| Board forward displacement | 0.497 m | 0.497 m |
| Mean AGV/Lift attitude tracking error | 0.67 deg | 0.65 deg |
| Virtual carry/stabilization active | never | never |

This confirms that the moving support plates carry the Board through actual
contact friction rather than the virtual velocity rewrite. It also produces a
clear same-plant distinction between fixed-height and active-leveling modes.

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
  --controller neutral ^
  --force_front_support_loss_at 2 ^
  --log_path logs\v7_physics_contact\matched_pad_support_loss_7s.csv ^
  --headless
```

Result:

| Metric | Before loss | After loss |
|---|---:|---:|
| Real Lift contacts | 3 | 1 |
| Board pitch | about 0.34 deg | 12.26 deg final |
| Maximum Board tilt | - | 12.45 deg |
| Maximum Cargo relative XY displacement | - | about 1.5 mm |

The Board tips forward until its front edge reaches the flat ground plane. The
Cargo does not slide appreciably because the configured static friction is
0.80 and a 12.3 degree slope is below its static-friction angle. This is an
expected physical outcome, not hidden stabilization.

## Visual commands

Normal contact-only transport:

```bat
python scripts\leveling_physics_contact_demo.py --duration 20 --target_speed 0.10 --terrain_amplitude 0.050 --controller neutral --real_time
```

`neutral` is the visual default so terrain-induced Board motion is easy to
see. Use `--controller geometric_feedback` to enable the active leveling
controller in the same contact-only plant.

To reveal the real collision pads and compare their orientation with the
visible Lift heads and AGVs:

```bat
python scripts\leveling_physics_contact_demo.py --duration 20 --controller neutral --show_lift_collision_proxies --real_time
```

The CSV records the actual simulated roll/pitch of all three AGVs and all
three Lift rigid bodies. This makes it possible to verify numerically that the
collision pads tilt with the carriers; the debug boxes are not visual-only
decorations.

Visible support-loss demonstration:

```bat
python scripts\leveling_physics_contact_demo.py --duration 10 --target_speed 0 --terrain_amplitude 0.050 --force_front_support_loss_at 3 --real_time
```

The demo also copies the requested amplitude and phase into the visual mesh
configuration before scene construction. Its fallback plane is placed 30 mm
below the deepest sinusoidal valley, so the negative terrain half is no longer
hidden by a coincident flat ground plane.

## Known modelling boundary

AGV terrain pose is still prescribed analytically and the white sinusoidal
terrain mesh is visual-only. This mode makes the Board, Cargo and Lift contact
chain physical; it is not yet a full wheel-terrain vehicle dynamics model. A
future full-physics vehicle task would require articulated/dynamic AGVs and a
collidable terrain mesh.
