"""Turn a trained mjlab / RSL-RL policy into a contract-form ONNX + JSON pair.

Used inside the training notebook (GPU machine with mjlab installed):

    from faultwalk.export import export_policy
    onnx_path, json_path = export_policy(runner, env, "/kaggle/working/export", "go2_flat",
                                         task_id="Unitree-Go2-Flat")

Observation normalisation: RSL-RL (>= 5.0) builds its ONNX module from
``obs_normalizer -> mlp -> deterministic output`` (``_OnnxMLPModel`` in
``rsl_rl/models/mlp_model.py``), so the normaliser is already part of the
graph. :func:`export_policy` checks that it is present (not an Identity) when
the actor was trained with ``obs_normalization=True`` and that ONNX and PyTorch
agree on random inputs.

Heavy imports (torch, onnx, onnxruntime, mjlab) happen inside functions, so the
module itself imports with the standard library and :mod:`faultwalk.contract`.

:func:`reference_metadata` gives the contract for a stock task from values read
in the task's source, without a checkpoint. It is the stand-in contract used
by tests and by tools developed before a trained policy exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faultwalk.contract import (
    ActionSpec,
    ControlSpec,
    ObservationSpec,
    ObservationTerm,
    OnnxIO,
    PDGains,
    PolicyMetadata,
    check_onnx,
)

GO2_TRAINING_JOINT_ORDER = [
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")
]


def reference_metadata(task_id: str = "Unitree-Go2-Flat") -> PolicyMetadata:
    """Contract for a stock task, from values read in its source code.

    Unitree-Go2-Flat (unitreerobotics/unitree_rl_mjlab @ 1425b15, 2026-04-13):
      - src/assets/robots/unitree_go2/go2_constants.py: kp 20/20/40, kd 1/1/2,
        effort 23.5/23.5/45, armature 0.01/0.01/0.02 (hip/thigh/calf); init pose
        thigh 0.9, calf -1.8, hip +0.1 on right legs, -0.1 on left legs.
      - src/tasks/velocity/velocity_env_cfg.py: actor terms base_ang_vel,
        projected_gravity, command, phase(period 0.6), joint_pos, joint_vel,
        actions; no scales; action scale 0.25; timestep 0.005; decimation 4.
      - xmls/go2.xml joint order: FL, FR, RL, RR x hip, thigh, calf.
    """
    if task_id != "Unitree-Go2-Flat":
        raise KeyError(f"no reference metadata for {task_id!r}")
    names = GO2_TRAINING_JOINT_ORDER
    part = [n.split("_")[1] for n in names]
    hip_sign = [(-0.1 if n[1] == "L" else 0.1) for n in names]
    default = [
        hip_sign[i] if p == "hip" else (0.9 if p == "thigh" else -1.8) for i, p in enumerate(part)
    ]
    n = len(names)
    return PolicyMetadata(
        joint_names=list(names),
        default_joint_pos=default,
        action=ActionSpec(size=n, scale=[0.25] * n),
        pd=PDGains(
            stiffness=[40.0 if p == "calf" else 20.0 for p in part],
            damping=[2.0 if p == "calf" else 1.0 for p in part],
            effort_limit=[45.0 if p == "calf" else 23.5 for p in part],
            armature=[0.02 if p == "calf" else 0.01 for p in part],
        ),
        control=ControlSpec(physics_dt=0.005, decimation=4),
        observation=ObservationSpec(
            terms=[
                ObservationTerm("base_ang_vel", 3),
                ObservationTerm("projected_gravity", 3),
                ObservationTerm("command", 3),
                ObservationTerm("phase", 2, params={"period": 0.6, "standing_threshold": 0.1}),
                ObservationTerm("joint_pos", n),
                ObservationTerm("joint_vel", n),
                ObservationTerm("actions", n),
            ],
            history_length=1,
        ),
        onnx=OnnxIO(),
        source={
            "framework": "mjlab",
            "repo": "https://github.com/unitreerobotics/unitree_rl_mjlab",
            "commit": "1425b15f73bd4095f0df53709d7c389c3eb9e790",
            "task": task_id,
            "kind": "reference (from source, no checkpoint)",
        },
    )


# ------------------------------------------------------------ from live env
def metadata_from_env(
    env: Any, task_id: str, source: dict | None = None, clip_actions: float | None = None
) -> PolicyMetadata:
    """Read the contract from a live mjlab ``ManagerBasedRlEnv`` (unwrapped).

    Every value comes from the env, not from constants, so it reflects any CLI
    overrides used for training. Joint order is the action term's target order,
    which must equal the robot's joint order (checked). ``clip_actions`` is the
    RSL-RL runner's ``clip_actions`` (applied by ``RslRlVecEnvWrapper.step``).
    """
    import numpy as np

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    term = env.action_manager.get_term("joint_pos")
    # VERIFY: `_target_names` is private in mjlab 1.2-1.6; seen in
    # https://github.com/mujocolab/mjlab/blob/v1.6.1/src/mjlab/envs/mdp/actions/actions.py
    target_names = list(getattr(term, "_target_names", joint_names))
    if target_names != joint_names:
        raise RuntimeError(f"action order {target_names} != joint order {joint_names}")
    n = len(joint_names)
    # VERIFY: `_scale` is a float or a (num_envs, n) tensor (same file as above).
    scale = term._scale
    scale = [float(scale)] * n if isinstance(scale, (int, float)) else scale[0].cpu().tolist()
    default = robot.data.default_joint_pos[0].cpu().tolist()

    m = env.sim.mj_model
    import mujoco

    def jid(name: str) -> int:
        for cand in (name, f"robot/{name}"):  # mjlab prefixes entity element names
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, cand)
            if i >= 0:
                return i
        raise KeyError(name)

    jids = [jid(nm) for nm in joint_names]
    act = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    aids = [act[j] for j in jids]
    kp = m.actuator_gainprm[aids, 0]
    kd = -m.actuator_biasprm[aids, 2]
    if not np.allclose(-m.actuator_biasprm[aids, 1], kp):
        raise RuntimeError("actuators are not position servos (biasprm[1] != -kp)")
    effort = (
        [float(x) for x in m.actuator_forcerange[aids, 1]]
        if m.actuator_forcelimited[aids].all()
        else None
    )
    armature = [float(x) for x in m.dof_armature[m.jnt_dofadr[jids]]]

    om = env.observation_manager
    names = list(om.active_terms["actor"])
    dims = om.group_obs_term_dim["actor"]
    terms: list[ObservationTerm] = []
    hist_set: set[int] = set()
    for name, dim in zip(names, dims, strict=True):
        cfg = om.get_term_cfg("actor", name)
        h = max(1, int(getattr(cfg, "history_length", 0) or 0))
        hist_set.add(h)
        total = int(np.prod(dim))
        sc = cfg.scale
        if sc is None:
            sc = 1.0
        elif hasattr(sc, "cpu"):
            sc = sc.cpu().tolist()
        params = {}
        if name == "phase":
            # 0.1 is hard-coded in unitree_rl_mjlab src/tasks/velocity/mdp/observations.py
            params = {"period": float(cfg.params["period"]), "standing_threshold": 0.1}
        clip = list(cfg.clip) if getattr(cfg, "clip", None) is not None else None
        terms.append(ObservationTerm(name, total // h, sc, clip, params))
    if len(hist_set) != 1:
        raise RuntimeError(f"per-term history lengths differ {hist_set}; contract 0.1 needs one")

    meta = PolicyMetadata(
        joint_names=joint_names,
        default_joint_pos=default,
        action=ActionSpec(
            size=n,
            scale=scale,
            clip=None if clip_actions is None else [-clip_actions, clip_actions],
        ),
        pd=PDGains(kp.tolist(), kd.tolist(), effort, armature),
        control=ControlSpec(physics_dt=float(env.physics_dt), decimation=int(env.cfg.decimation)),
        observation=ObservationSpec(terms=terms, history_length=hist_set.pop()),
        source={"framework": "mjlab", "task": task_id, **(source or {})},
    )
    meta.validate()
    return meta


# ------------------------------------------------------------------- export
def export_policy(
    runner: Any,
    env: Any,
    out_dir: str | Path,
    name: str,
    task_id: str,
    source: dict | None = None,
    clip_actions: float | None = None,
    parity_samples: int = 64,
    atol: float = 1e-4,
) -> tuple[Path, Path]:
    """Write ``<name>.onnx`` and ``<name>.json`` (contract form) and verify them.

    ``runner`` is an ``MjlabOnPolicyRunner`` (RSL-RL ``OnPolicyRunner``), ``env``
    the unwrapped mjlab env (or the RSL-RL wrapper; ``.unwrapped`` is used).
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    env = getattr(env, "unwrapped", env)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = metadata_from_env(env, task_id, source, clip_actions)

    # VERIFY: rsl-rl 5.0.1 / 5.4.2 API, read in rsl_rl/algorithms/ppo.py (get_policy) and
    # rsl_rl/models/mlp_model.py (as_onnx(verbose)); https://github.com/leggedrobotics/rsl_rl
    policy = runner.alg.get_policy()
    module = policy.as_onnx(verbose=False)
    module.to("cpu").eval()
    if getattr(policy, "obs_normalization", False) and isinstance(
        module.obs_normalizer, torch.nn.Identity
    ):
        raise RuntimeError("actor uses obs normalisation but the ONNX module has none")

    onnx_path = out / f"{name}.onnx"
    dummy = torch.zeros(1, meta.observation.size)
    torch.onnx.export(
        module,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=[meta.onnx.input_name],
        output_names=[meta.onnx.action_output],
        dynamic_axes={},
        dynamo=False,  # VERIFY: same choice as mjlab's export_policy_to_onnx (torch>=2.9)
    )
    # Inline any external-data file so a policy is always a single .onnx file,
    # and embed the contract for tools that only get the ONNX.
    model = onnx.load(str(onnx_path))
    entry = model.metadata_props.add()
    entry.key = "faultwalk_contract"
    entry.value = json.dumps(meta.to_dict())
    onnx.save(model, str(onnx_path), save_as_external_data=False)
    stray = out / f"{name}.onnx.data"
    if stray.exists():
        stray.unlink()

    json_path = out / f"{name}.json"
    meta.save(json_path)
    check_onnx(meta, onnx_path)

    # Parity: ONNX vs the PyTorch module on observation-scale random inputs.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    with torch.no_grad():
        for _ in range(parity_samples):
            x = torch.randn(1, meta.observation.size, generator=gen)
            ref = module(x).numpy()
            got = sess.run(None, {meta.onnx.input_name: x.numpy()})[0]
            worst = max(worst, float(np.abs(ref - got).max()))
    if worst > atol:
        raise RuntimeError(f"ONNX/PyTorch mismatch {worst:.2e} > {atol}")
    return onnx_path, json_path
