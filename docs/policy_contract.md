# Policy contract (version 0.1.0)

Every faultwalk policy, whatever produced it, is two files:

- `<name>.onnx`: the network, with observation normalisation inside the graph.
- `<name>.json`: metadata saying how to feed it and how to use its output.

The CPU evaluator, the browser demo and the ROS 2 node read only these two files. The
reference implementation is `src/faultwalk/contract.py` (loading, validation, ONNX signature
check). `src/faultwalk/cpu_sim.py` is the reference consumer, and `src/faultwalk/export.py`
writes the files from a trained mjlab / RSL-RL policy.

## ONNX graph

| Tensor | Name (default) | Shape | Meaning |
|---|---|---|---|
| input | `obs` | `[B, observation.size]` | float32 observation vector, built as below |
| output | `actions` | `[B, 12]` | raw action `a`, in policy joint order |
| output (reserved) | `diagnosis` | `[B, 12, 4]` | per-joint fault logits; see below |

`B` is 1 or dynamic. No other inputs or outputs are allowed. A `diagnosis` output must be
declared in the metadata, and a declared one must exist. `faultwalk.contract.check_onnx` checks
all of this.

The graph is the deterministic policy (the mean action). RSL-RL's ONNX module is
`obs_normalizer -> mlp -> deterministic output`, so normalisation statistics travel with the
weights. Consumers never normalise.

## Metadata JSON

```jsonc
{
  "contract_version": "0.1.0",          // major version must match the reader
  "robot": "unitree_go2",
  "joint_names": ["FL_hip_joint", ...],  // 12 names: the policy joint order
  "default_joint_pos": [...],           // rad, policy order
  "action": {
    "size": 12,
    "scale": [...],                     // per joint
    "clip": null                        // or [lo, hi], applied to the raw action
  },
  "pd": {
    "stiffness": [...],                 // N m / rad
    "damping": [...],                   // N m s / rad
    "effort_limit": [...],              // N m, or null
    "armature": [...]                   // kg m^2, or null
  },
  "control": {"physics_dt": 0.005, "decimation": 4, "control_dt": 0.02},
  "observation": {
    "history_length": 1,
    "history_layout": "term_major_oldest_first",
    "size": 47,
    "terms": [
      {"name": "base_ang_vel", "size": 3, "scale": 1.0, "clip": null, "params": {}},
      ...
    ]
  },
  "onnx": {"input_name": "obs", "action_output": "actions"},
  "diagnosis": null,                    // or a DiagnosisSpec, see below
  "source": {...}                       // provenance, free-form
}
```

`control_dt` and `observation.size` are derived. If present they must agree with the primary
fields.

### Joint order

`joint_names` is the order of every per-joint vector in the file and in the network's input and
output. It is the order the training framework used (mjlab: the order of joints in the robot's
MJCF; unitree_rl_mjlab's Go2 is FL, FR, RL, RR x hip, thigh, calf). Consumers map these names
to their own model by name. Menagerie's Go2 happens to use the same order today; nothing may
rely on that.

### Action to joint target

```
a      = clip(actions, *action.clip)            # if clip is set
target = default_joint_pos + action.scale * a   # rad, per joint
```

The target goes to a position servo with the training gains, as in mjlab's
`BuiltinPositionActuator`:

```
torque = clip(stiffness * (target - q) - damping * qdot, -effort_limit, effort_limit)
```

The target is held for `decimation` physics steps of `physics_dt`, so the policy runs every
`control_dt` seconds. `armature` is the reflected rotor inertia added to each joint in training.

### Observation vector

Each term is computed, clipped (if `clip` is set), then multiplied by `scale` (mjlab's order:
compute, noise, clip, scale). No noise is added at evaluation time.

Known terms (`contract.KNOWN_TERMS`):

| Term | Size | Definition |
|---|---|---|
| `base_lin_vel` | 3 | linear velocity of the IMU site in the IMU frame (velocimeter) |
| `base_ang_vel` | 3 | angular velocity of the IMU site in the IMU frame (gyro) |
| `projected_gravity` | 3 | `R_base^T [0, 0, -1]` |
| `command` | 3 | commanded `(vx, vy, wz)` in the base frame |
| `phase` | 2 | `(sin 2 pi p, cos 2 pi p)`, `p = (k * control_dt mod period) / period`, `k` = policy steps since reset; zero when the command norm is below `params.standing_threshold` (0.1) |
| `joint_pos` | 12 | `q - default_joint_pos`, policy order |
| `joint_vel` | 12 | `qdot`, policy order |
| `actions` | 12 | previous raw action `a` (before scale and offset); zero after reset |

A term not in this table fails validation unless the reader passes `allow_unknown_terms=True`.
Adding a term means adding it here, to `KNOWN_TERMS` and to `cpu_sim.Go2Sim.term_value`.

### History

With `history_length = H`, each term keeps its last `H` frames, oldest first, and the full
vector is the terms concatenated in order:

```
[term1(t-H+1) ... term1(t), term2(t-H+1) ... term2(t), ...]
```

This is mjlab's layout (per-term circular buffer flattened, then concatenation). After a reset
the history is filled with copies of the first frame, as mjlab does. Contract 0.1 has one `H`
for all terms. The nominal policy uses `H = 1`; the adaptive policy (M3) will use `H > 1`.

### Diagnosis output (reserved)

The adaptive policy will also report which motor it thinks has failed. Contract 0.1 reserves:

```json
"diagnosis": {
  "output_name": "diagnosis",
  "shape": [12, 4],
  "classes": ["healthy", "locked", "limp", "weak"],
  "kind": "logits",
  "status": "reserved"
}
```

Meaning: for each joint in `joint_names` order, unnormalised logits over the classes. Shape,
classes and whether severity is also reported are to be finalised in M3; until then nothing
should depend on the values, and stand-in policies omit the output (or emit zeros if a consumer
needs the tensor to exist). Changing it will bump the contract version.

## Versioning

`contract_version` is semver. Readers reject a different major version. Adding optional fields
is a minor bump; changing the meaning, order or shape of anything above is a major bump.

## Stock reference: Unitree-Go2-Flat

`faultwalk.export.reference_metadata()` returns the contract for unitree_rl_mjlab's
`Unitree-Go2-Flat` task, read from its source (commit `1425b15`, 2026-04-13):

| Field | Value |
|---|---|
| joint order | FL, FR, RL, RR x hip, thigh, calf |
| default pose | hip -0.1 (left) / +0.1 (right), thigh 0.9, calf -1.8 |
| stiffness / damping | hip, thigh 20 / 1; calf 40 / 2 |
| effort limit / armature | hip, thigh 23.5 N m / 0.01; calf 45 N m / 0.02 |
| action scale | 0.25 for all joints |
| timing | physics 0.005 s, decimation 4, control 50 Hz |
| observation (47) | base_ang_vel 3, projected_gravity 3, command 3, phase 2 (period 0.6 s), joint_pos 12, joint_vel 12, actions 12 |

Sources: `src/assets/robots/unitree_go2/go2_constants.py`,
`src/tasks/velocity/velocity_env_cfg.py`, `src/tasks/velocity/config/go2/env_cfgs.py` and
`src/tasks/velocity/mdp/observations.py` in
<https://github.com/unitreerobotics/unitree_rl_mjlab>. A policy exported from an actual run
uses values read from the live environment (`export.metadata_from_env`), not these constants.

## CPU MuJoCo notes (`cpu_sim.py`)

- Robot: MuJoCo Menagerie `unitree_go2/go2.xml` at a pinned commit, downloaded and cached under
  `~/.cache/faultwalk` (override with `FAULTWALK_CACHE`, or point `FAULTWALK_GO2_XML` at a local
  file). Visual meshes are not downloaded; the mesh geoms are removed, which leaves the physics
  unchanged (explicit inertials, primitive collision geoms).
- Menagerie's torque motors are reconfigured as position servos with the contract gains, and its
  passive joint damping (2) and friction loss (0.2) are removed to match training
  (`--menagerie-dynamics` keeps them). Contact and solver settings stay Menagerie's (elliptic
  cone, impratio 100, foot friction 0.8): that is the deliberate sim-to-sim difference.
- Falling = base tilt above 70 degrees (the training `fell_over` termination) or base height
  below 0.12 m.
