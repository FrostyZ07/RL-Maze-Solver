"""Policy contract: metadata validation and ONNX signature checks."""

from __future__ import annotations

import copy
import json

import pytest

from faultwalk.contract import (
    CONTRACT_VERSION,
    ContractError,
    DiagnosisSpec,
    PolicyMetadata,
    check_onnx,
)
from faultwalk.export import reference_metadata


def meta_dict() -> dict:
    return reference_metadata().to_dict()


def make_onnx(path, obs_dim: int, act_dim: int = 12, diag_shape=None, batch=1, in_name="obs"):
    """Tiny linear ONNX policy with all-zero weights (actions = 0)."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    inits = [numpy_helper.from_array(np.zeros((obs_dim, act_dim), np.float32), "W")]
    nodes = [helper.make_node("MatMul", [in_name, "W"], ["actions"])]
    outputs = [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [batch, act_dim])]
    if diag_shape is not None:
        k = int(np.prod(diag_shape))
        inits += [
            numpy_helper.from_array(np.zeros((obs_dim, k), np.float32), "Wd"),
            numpy_helper.from_array(np.array([-1, *diag_shape], np.int64), "shape"),
        ]
        nodes += [
            helper.make_node("MatMul", [in_name, "Wd"], ["d_flat"]),
            helper.make_node("Reshape", ["d_flat", "shape"], ["diagnosis"]),
        ]
        outputs.append(
            helper.make_tensor_value_info("diagnosis", TensorProto.FLOAT, [batch, *diag_shape])
        )
    graph = helper.make_graph(
        nodes,
        "dummy",
        [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [batch, obs_dim])],
        outputs,
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


# ---------------------------------------------------------------- metadata
def test_reference_metadata_is_valid():
    m = reference_metadata()
    m.validate()
    assert m.observation.size == 47  # 3+3+3+2+12+12+12
    assert m.control.control_dt == pytest.approx(0.02)
    assert m.contract_version == CONTRACT_VERSION
    assert m.diagnosis is None


def test_roundtrip(tmp_path):
    m = reference_metadata()
    p = tmp_path / "m.json"
    m.save(p)
    m2 = PolicyMetadata.load(p)
    assert m2 == m
    raw = json.loads(p.read_text())
    assert raw["observation"]["size"] == 47
    assert raw["control"]["control_dt"] == pytest.approx(0.02)


def test_slices_term_major_with_history():
    d = meta_dict()
    d["observation"]["history_length"] = 3
    d["observation"].pop("size")
    m = PolicyMetadata.from_dict(d)
    assert m.observation.size == 47 * 3
    s = m.observation.slices()
    assert s["base_ang_vel"] == slice(0, 9)
    assert s["projected_gravity"] == slice(9, 18)
    assert s["actions"].stop == 141


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["joint_names"].pop(), "expected 12 joints"),
        (lambda d: d["joint_names"].__setitem__(1, d["joint_names"][0]), "duplicates"),
        (lambda d: d["joint_names"].__setitem__(0, "FL_knee"), "not in the Go2 model"),
        (lambda d: d["pd"]["stiffness"].__setitem__(0, 0.0), "stiffness"),
        (lambda d: d["pd"]["damping"].append(1.0), "pd.damping has length 13"),
        (lambda d: d["action"].__setitem__("scale", [0.25] * 11), "action.scale"),
        (lambda d: d["control"].__setitem__("decimation", 0), "decimation"),
        (lambda d: d["control"].__setitem__("control_dt", 0.05), "control_dt"),
        (lambda d: d["observation"]["terms"][0].__setitem__("size", 4), "base_ang_vel"),
        (lambda d: d["observation"]["terms"].append({"name": "foo", "size": 1}), "unknown"),
        (lambda d: d["observation"]["terms"][3]["params"].clear(), "phase"),
        (lambda d: d["observation"].__setitem__("history_length", 0), "history_length"),
        (lambda d: d["observation"].__setitem__("size", 48), "observation.size"),
        (lambda d: d.__setitem__("contract_version", "9.0.0"), "contract_version"),
        (lambda d: d.pop("pd"), "malformed"),
    ],
)
def test_invalid_metadata_rejected(mutate, match):
    d = meta_dict()
    mutate(d)
    with pytest.raises(ContractError, match=match):
        PolicyMetadata.from_dict(d)


def test_unknown_term_can_be_allowed():
    m = reference_metadata()
    m2 = copy.deepcopy(m)
    from faultwalk.contract import ObservationTerm

    m2.observation.terms.append(ObservationTerm("height_scan", 187))
    with pytest.raises(ContractError):
        m2.validate()
    m2.validate(allow_unknown_terms=True)


def test_joint_index_by_name():
    m = reference_metadata()
    assert m.joint_index(["FR_hip_joint", "FL_hip_joint"]) == [3, 0]
    with pytest.raises(ContractError):
        m.joint_index(["FL_knee_joint"])


def test_diagnosis_spec_validation():
    m = reference_metadata()
    m.diagnosis = DiagnosisSpec()
    m.validate()
    assert m.diagnosis.shape == [12, 4]
    m.diagnosis.shape = [12, 3]
    with pytest.raises(ContractError, match="classes"):
        m.validate()


# -------------------------------------------------------------------- onnx
def test_check_onnx_accepts_matching_model(tmp_path):
    m = reference_metadata()
    sig = check_onnx(m, make_onnx(tmp_path / "p.onnx", 47))
    assert sig.inputs == {"obs": [1, 47]}
    assert sig.outputs == {"actions": [1, 12]}


def test_check_onnx_dynamic_batch(tmp_path):
    check_onnx(reference_metadata(), make_onnx(tmp_path / "p.onnx", 47, batch="N"))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"obs_dim": 48}, "input shape"),
        ({"obs_dim": 47, "act_dim": 11}, "action output"),
        ({"obs_dim": 47, "batch": 8}, "batch dim"),
        ({"obs_dim": 47, "in_name": "observations"}, "inputs"),
        ({"obs_dim": 47, "diag_shape": [12, 4]}, "unexpected outputs"),
    ],
)
def test_check_onnx_rejects_mismatch(tmp_path, kwargs, match):
    path = make_onnx(tmp_path / "p.onnx", **kwargs)
    with pytest.raises(ContractError, match=match):
        check_onnx(reference_metadata(), path)


def test_check_onnx_diagnosis_output(tmp_path):
    m = reference_metadata()
    m.diagnosis = DiagnosisSpec()
    check_onnx(m, make_onnx(tmp_path / "d.onnx", 47, diag_shape=[12, 4]))
    with pytest.raises(ContractError, match="diagnosis output"):
        check_onnx(m, make_onnx(tmp_path / "d2.onnx", 47, diag_shape=[12, 3]))
    with pytest.raises(ContractError, match="lacks it"):
        check_onnx(m, make_onnx(tmp_path / "d3.onnx", 47))


# ------------------------------------------------------------------ export
def _fake_mjlab(tmp_path):
    """Minimal stand-ins for an mjlab env + RSL-RL runner (no mjlab, no GPU)."""
    torch = pytest.importorskip("torch")
    mujoco = pytest.importorskip("mujoco")
    from types import SimpleNamespace

    ref = reference_metadata()
    names = ref.joint_names
    bodies = "".join(
        f'<body name="b{i}" pos="{i} 0 0"><joint name="robot/{n}" axis="0 1 0" '
        f'armature="{ref.pd.armature[i]}"/><geom size="0.05" mass="1"/></body>'
        for i, n in enumerate(names)
    )
    # Actuators deliberately listed in reverse order: export must map by name.
    acts = "".join(
        f'<position joint="robot/{n}" kp="{ref.pd.stiffness[i]}" kv="{ref.pd.damping[i]}" '
        f'forcerange="-{ref.pd.effort_limit[i]} {ref.pd.effort_limit[i]}"/>'
        for i, n in reversed(list(enumerate(names)))
    )
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody><actuator>{acts}</actuator></mujoco>"
    )
    terms = ref.observation.terms
    cfgs = {
        t.name: SimpleNamespace(
            scale=None, clip=None, history_length=0, params=dict(t.params) or {}
        )
        for t in terms
    }
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                joint_names=names,
                data=SimpleNamespace(default_joint_pos=torch.tensor([ref.default_joint_pos])),
            )
        },
        action_manager=SimpleNamespace(
            get_term=lambda _: SimpleNamespace(_target_names=names, _scale=0.25)
        ),
        sim=SimpleNamespace(mj_model=model),
        observation_manager=SimpleNamespace(
            active_terms={"actor": [t.name for t in terms]},
            group_obs_term_dim={"actor": [(t.size,) for t in terms]},
            get_term_cfg=lambda g, n: cfgs[n],
        ),
        physics_dt=0.005,
        cfg=SimpleNamespace(decimation=4),
    )

    class Normalizer(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.register_buffer("_mean", torch.randn(1, n))
            self.register_buffer("_std", torch.rand(1, n) + 0.5)

        def forward(self, x):
            return (x - self._mean) / (self._std + 0.01)

    class OnnxModule(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.obs_normalizer = Normalizer(n)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(n, 32), torch.nn.ELU(), torch.nn.Linear(32, 12)
            )

        def forward(self, x):
            return self.mlp(self.obs_normalizer(x))

    policy = SimpleNamespace(
        obs_normalization=True, as_onnx=lambda verbose: OnnxModule(ref.observation.size)
    )
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: policy))
    return runner, env, ref


def test_export_policy_contract_form(tmp_path):
    pytest.importorskip("onnxruntime")
    from faultwalk.export import export_policy

    runner, env, ref = _fake_mjlab(tmp_path)
    onnx_path, json_path = export_policy(runner, env, tmp_path, "pol", task_id="Unitree-Go2-Flat")
    meta = PolicyMetadata.load(json_path)
    check_onnx(meta, onnx_path)
    assert meta.joint_names == ref.joint_names
    assert meta.pd == ref.pd
    assert meta.action.scale == ref.action.scale
    assert meta.default_joint_pos == pytest.approx(ref.default_joint_pos)
    assert meta.observation.size == ref.observation.size
    assert [t.name for t in meta.observation.terms] == [t.name for t in ref.observation.terms]
    assert meta.control.control_dt == pytest.approx(0.02)
    assert not (tmp_path / "pol.onnx.data").exists()
