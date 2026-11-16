"""CPU MuJoCo runner: smoke tests with a dummy zero-action ONNX policy.

Tests that need the Menagerie Go2 model skip when it is not cached and cannot
be downloaded (offline).
"""

from __future__ import annotations

import urllib.request

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from faultwalk import cpu_sim  # noqa: E402
from faultwalk.contract import PolicyMetadata  # noqa: E402
from faultwalk.export import reference_metadata  # noqa: E402


@pytest.fixture(scope="module")
def go2_xml():
    try:
        return cpu_sim.fetch_menagerie_go2(timeout=10)
    except cpu_sim.ModelUnavailable as e:
        pytest.skip(f"Go2 model unavailable (offline?): {e}")


def zero_policy_onnx(path, obs_dim: int, act_dim: int = 12):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.zeros((obs_dim, act_dim), np.float32), "W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["actions"])],
        "zero_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, act_dim])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_fetch_offline_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("FAULTWALK_CACHE", str(tmp_path))
    monkeypatch.delenv("FAULTWALK_GO2_XML", raising=False)

    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(cpu_sim.ModelUnavailable):
        cpu_sim.fetch_menagerie_go2()


def test_joint_mapping_by_name(go2_xml):
    meta = reference_metadata()
    model, jmap = cpu_sim.build_model(meta, go2_xml)
    for i, name in enumerate(meta.joint_names):
        j = model.joint(name)
        assert jmap.qpos_adr[i] == model.jnt_qposadr[j.id]
        assert model.actuator_trnid[jmap.actuator_id[i], 0] == j.id
        assert model.actuator_gainprm[jmap.actuator_id[i], 0] == meta.pd.stiffness[i]
    assert model.opt.timestep == pytest.approx(0.005)


def test_zero_policy_stands(tmp_path, go2_xml):
    """Zero action = hold the default pose: the robot must not fall for 3 s."""
    meta = reference_metadata()
    path = zero_policy_onnx(tmp_path / "zero.onnx", meta.observation.size)
    sim = cpu_sim.Go2Sim(meta, cpu_sim.OnnxPolicy(path, meta), go2_xml)
    s = sim.run(seconds=3.0)
    assert not s.fell
    assert s.steps == 150
    assert s.mean_base_height > 0.2
    assert s.distance_xy < 0.1
    obs = sim.observe()
    sl = meta.observation.slices()
    np.testing.assert_allclose(obs[sl["projected_gravity"]], [0, 0, -1], atol=0.05)
    assert np.all(obs[sl["phase"]] == 0)  # zero command -> phase zeroed
    # Low training gains (kp 20/40) let the joints sag ~0.1-0.25 rad under body
    # weight; a trained policy offsets this, a zero policy does not.
    assert np.abs(obs[sl["joint_pos"]]).max() < 0.3


def test_shuffled_joint_order_still_stands(tmp_path, go2_xml):
    """A policy trained with another joint order must map by name, not index."""
    d = reference_metadata().to_dict()
    perm = np.random.default_rng(0).permutation(12)
    for key in ("joint_names", "default_joint_pos"):
        d[key] = [d[key][i] for i in perm]
    for key in ("stiffness", "damping", "effort_limit", "armature"):
        d["pd"][key] = [d["pd"][key][i] for i in perm]
    d["action"]["scale"] = [d["action"]["scale"][i] for i in perm]
    meta = PolicyMetadata.from_dict(d)
    path = zero_policy_onnx(tmp_path / "zero.onnx", meta.observation.size)
    s = cpu_sim.Go2Sim(meta, cpu_sim.OnnxPolicy(path, meta), go2_xml).run(seconds=2.0)
    assert not s.fell and s.mean_base_height > 0.2


def test_history_layout_and_phase(go2_xml):
    d = reference_metadata().to_dict()
    d["observation"]["history_length"] = 3
    d["observation"].pop("size")
    meta = PolicyMetadata.from_dict(d)
    sim = cpu_sim.Go2Sim(meta, lambda obs: np.zeros(12), go2_xml)
    sim.set_command(0.5, 0.0, 0.0)
    obs = sim.reset()
    assert obs.shape == (47 * 3,)
    ph = meta.observation.slices()["phase"]
    np.testing.assert_allclose(obs[ph], [0, 1] * 3, atol=1e-6)  # back-filled with t=0 frame
    for _ in range(2):
        obs = sim.step(obs)
    frames = obs[ph].reshape(3, 2)  # oldest -> newest
    t = np.array([0, 1, 2]) * meta.control.control_dt / 0.6
    np.testing.assert_allclose(frames[:, 0], np.sin(2 * np.pi * t), atol=1e-5)


def test_cli(tmp_path, go2_xml, monkeypatch, capsys):
    meta = reference_metadata()
    meta.save(tmp_path / "m.json")
    zero_policy_onnx(tmp_path / "p.onnx", meta.observation.size)
    monkeypatch.setenv("FAULTWALK_GO2_XML", str(go2_xml))
    cpu_sim.main(
        ["--policy", str(tmp_path / "p.onnx"), "--meta", str(tmp_path / "m.json"),
         "--vx", "0.5", "--seconds", "1"]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert '"fell": false' in out and '"steps": 50' in out
