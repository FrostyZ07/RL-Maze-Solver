"""Run a contract-conforming ONNX policy on the Unitree Go2 in plain CPU MuJoCo.

    python -m faultwalk.cpu_sim --policy policy.onnx --meta policy.json --vx 0.5 --seconds 10

Robot model: ``unitree_go2/go2.xml`` from MuJoCo Menagerie, downloaded once
from a pinned commit and cached (see :func:`fetch_menagerie_go2`). Only the
12 KB MJCF and its licence are fetched by default; the 28 MB of visual meshes
are optional and stripped from the model when absent. Physics is unaffected:
every Menagerie Go2 body has an explicit ``<inertial>``, the mesh geoms are
visual only (contype=conaffinity=0), and all collision geoms are primitives.

Why not ``robot_descriptions``: its ``go2_mj_description`` clones the entire
Menagerie repository (about 2 GB of models, plus GitPython and git) just to
read one 12 KB file.

Joint correspondence between the policy (training order, listed in the
metadata) and the MuJoCo model is resolved by joint *name*; actuators are found
through their transmission target joint, never by index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from faultwalk.contract import PolicyMetadata, check_onnx

MENAGERIE_COMMIT = "6da8b971844845de6ab27315aacc3a3fea4ddb8e"  # 2026-10-14, last unitree_go2 change
MENAGERIE_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
GO2_FILES = ("go2.xml", "LICENSE")
GO2_MESHES = tuple(
    f"assets/{n}.obj"
    for n in (
        "base_0 base_1 base_2 base_3 base_4 calf_0 calf_1 calf_mirror_0 calf_mirror_1 foot "
        "hip_0 hip_1 thigh_0 thigh_1 thigh_mirror_0 thigh_mirror_1"
    ).split()
)
BASE_BODY = "base"
IMU_SITE = "imu"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
FALL_TILT_DEG = 70.0  # mjlab velocity task `fell_over` termination (bad_orientation, 70 deg)
FALL_MIN_HEIGHT = 0.12  # base below this (m) also counts as fallen

Policy = Callable[[np.ndarray], np.ndarray]


class ModelUnavailable(OSError):
    """The Go2 MJCF is neither cached nor downloadable (for example, offline)."""


# --------------------------------------------------------------- model files
def cache_dir() -> Path:
    root = os.environ.get("FAULTWALK_CACHE") or Path.home() / ".cache" / "faultwalk"
    return Path(root) / "menagerie" / MENAGERIE_COMMIT / "unitree_go2"


def fetch_menagerie_go2(with_meshes: bool = False, timeout: float = 20.0) -> Path:
    """Return the path of a cached Menagerie ``go2.xml``, downloading it if needed.

    ``FAULTWALK_GO2_XML`` overrides everything with a local file. Raises
    :class:`ModelUnavailable` when a needed file cannot be downloaded.
    """
    override = os.environ.get("FAULTWALK_GO2_XML")
    if override:
        return Path(override)
    target = cache_dir()
    wanted = GO2_FILES + (GO2_MESHES if with_meshes else ())
    for rel in wanted:
        dst = target / rel
        if dst.exists():
            continue
        url = f"{MENAGERIE_RAW}/{MENAGERIE_COMMIT}/unitree_go2/{rel}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
                data = r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ModelUnavailable(f"cannot download {url}: {e}") from e
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as f:
            f.write(data)
        os.replace(f.name, dst)
    return target / "go2.xml"


# --------------------------------------------------------------------- model
@dataclass
class JointMap:
    """Policy-order indices into the MuJoCo model, resolved by name."""

    names: list[str]
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_id: np.ndarray


def build_model(
    meta: PolicyMetadata, xml_path: str | Path | None = None, training_dynamics: bool = True
) -> tuple[mujoco.MjModel, JointMap]:
    """Compile the Go2 on a flat floor and configure it as the policy expects.

    - timestep = ``control.physics_dt``, integrator implicitfast (mjlab default).
    - Each actuator becomes a MuJoCo position servo exactly like mjlab's
      ``BuiltinPositionActuator``: force = kp (ctrl - q) - kd qdot, ctrl
      unlimited, force clamped to ``pd.effort_limit``.
    - ``training_dynamics=True`` removes Menagerie's passive joint damping (2)
      and friction loss (0.2), which the training model does not have, and
      applies ``pd.armature``. Contact/solver settings (elliptic cone,
      impratio 100, foot friction 0.8) stay Menagerie's: that difference is the
      intended sim-to-sim gap.
    """
    xml_path = Path(xml_path) if xml_path else fetch_menagerie_go2()
    spec = mujoco.MjSpec.from_file(str(xml_path))
    mesh_dir = xml_path.parent / (spec.meshdir or "")
    if not all((mesh_dir / Path(m).name).exists() for m in GO2_MESHES):
        for g in list(spec.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(g)
        for m in list(spec.meshes):
            spec.delete(m)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    model = spec.compile()
    model.opt.timestep = meta.control.physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    jmap = map_joints(model, meta.joint_names)
    kp = np.asarray(meta.pd.stiffness)
    kd = np.asarray(meta.pd.damping)
    for i, a in enumerate(jmap.actuator_id):
        model.actuator_dyntype[a] = mujoco.mjtDyn.mjDYN_NONE
        model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[a, :] = 0
        model.actuator_biasprm[a, :] = 0
        model.actuator_gainprm[a, 0] = kp[i]
        model.actuator_biasprm[a, 1] = -kp[i]
        model.actuator_biasprm[a, 2] = -kd[i]
        model.actuator_gear[a, :] = 0
        model.actuator_gear[a, 0] = 1
        model.actuator_ctrllimited[a] = 0
        if meta.pd.effort_limit is not None:
            model.actuator_forcelimited[a] = 1
            model.actuator_forcerange[a] = (-meta.pd.effort_limit[i], meta.pd.effort_limit[i])
        else:
            model.actuator_forcelimited[a] = 0
    if training_dynamics:
        model.dof_damping[jmap.dof_adr] = 0.0
        model.dof_frictionloss[jmap.dof_adr] = 0.0
        if meta.pd.armature is not None:
            model.dof_armature[jmap.dof_adr] = np.asarray(meta.pd.armature)
    return model, jmap


def map_joints(model: mujoco.MjModel, names: list[str]) -> JointMap:
    """Resolve policy joint names to qpos/dof addresses and their actuator ids."""
    act_of_joint: dict[int, int] = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            j = int(model.actuator_trnid[a, 0])
            if j in act_of_joint:
                raise ValueError(f"joint {model.joint(j).name} has more than one actuator")
            act_of_joint[j] = a
    qadr, dadr, aid = [], [], []
    for nm in names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if j < 0:
            raise ValueError(f"joint {nm!r} not in model")
        if j not in act_of_joint:
            raise ValueError(f"joint {nm!r} has no actuator")
        qadr.append(model.jnt_qposadr[j])
        dadr.append(model.jnt_dofadr[j])
        aid.append(act_of_joint[j])
    return JointMap(list(names), np.array(qadr), np.array(dadr), np.array(aid))


# -------------------------------------------------------------------- policy
class OnnxPolicy:
    """onnxruntime wrapper; validates the model against the metadata first."""

    def __init__(self, path: str | Path, meta: PolicyMetadata):
        import onnxruntime as ort

        check_onnx(meta, path)
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = meta.onnx.input_name
        self.outputs = [meta.onnx.action_output]
        if meta.diagnosis is not None:
            self.outputs.append(meta.diagnosis.output_name)
        self.last_diagnosis: np.ndarray | None = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        res = self.session.run(self.outputs, {self.input: obs[None].astype(np.float32)})
        if len(res) > 1:
            self.last_diagnosis = res[1][0]
        return res[0][0]


# ---------------------------------------------------------------- simulation
@dataclass
class RolloutSummary:
    seconds: float
    steps: int
    fell: bool
    fall_time: float | None
    command: list[float]
    mean_lin_vel_body: list[float]  # (vx, vy), base frame
    mean_yaw_rate: float
    lin_vel_tracking_error: float  # mean |cmd_xy - v_xy| (m/s)
    yaw_rate_tracking_error: float  # mean |cmd_wz - wz| (rad/s)
    distance_xy: float
    mean_base_height: float


class Go2Sim:
    """Headless Go2 + policy loop that builds observations from the metadata."""

    def __init__(
        self,
        meta: PolicyMetadata,
        policy: Policy,
        xml_path: str | Path | None = None,
        training_dynamics: bool = True,
    ):
        meta.validate()
        self.meta = meta
        self.policy = policy
        self.model, self.jmap = build_model(meta, xml_path, training_dynamics)
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body(BASE_BODY).id
        self.imu_id = self.model.site(IMU_SITE).id
        self.default_q = np.asarray(meta.default_joint_pos, dtype=np.float64)
        self.action_scale = np.asarray(meta.action.scale, dtype=np.float64)
        self.command = np.zeros(3)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> np.ndarray:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.jmap.qpos_adr] = self.default_q
        d.qpos[2] = 0.5
        mujoco.mj_forward(m, d)
        lowest = min(
            float(d.geom(g).xpos[2] - m.geom(g).rbound[0])
            for g in FOOT_GEOMS
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0
        )
        d.qpos[2] = 0.5 - lowest + 0.002
        d.ctrl[self.jmap.actuator_id] = self.default_q
        mujoco.mj_forward(m, d)
        self.step_count = 0
        self.last_action = np.zeros(self.meta.action.size)
        self.history = {
            t.name: deque(maxlen=self.meta.observation.history_length)
            for t in self.meta.observation.terms
        }
        return self.observe(fill_history=True)

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        self.command = np.array([vx, vy, wz], dtype=np.float64)

    def base_rotation(self) -> np.ndarray:
        return self.data.xmat[self.base_id].reshape(3, 3)

    def imu_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """(angular, linear) velocity of the IMU site in its own frame (gyro, velocimeter)."""
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.imu_id, v, 1)
        return v[:3].copy(), v[3:].copy()

    def base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """(angular, linear) velocity of the base body frame, in the base frame."""
        v = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, v, 1
        )
        return v[:3].copy(), v[3:].copy()

    # -- observation ---------------------------------------------------------
    def term_value(self, name: str, params: dict) -> np.ndarray:
        d = self.data
        if name == "base_ang_vel":
            return self.imu_velocity()[0]
        if name == "base_lin_vel":
            return self.imu_velocity()[1]
        if name == "projected_gravity":
            return self.base_rotation().T @ np.array([0.0, 0.0, -1.0])
        if name == "command":
            return self.command.copy()
        if name == "phase":
            # unitree_rl_mjlab `phase`: t = episode steps * step_dt; zeroed when
            # ||command|| < 0.1 (standing).
            period = float(params["period"])
            thr = float(params.get("standing_threshold", 0.1))
            if np.linalg.norm(self.command) < thr:
                return np.zeros(2)
            ph = (self.step_count * self.meta.control.control_dt) % period / period
            return np.array([math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)])
        if name == "joint_pos":
            return d.qpos[self.jmap.qpos_adr] - self.default_q
        if name == "joint_vel":
            return d.qvel[self.jmap.dof_adr].copy()
        if name == "actions":
            return self.last_action.copy()
        raise KeyError(f"observation term {name!r} not implemented in cpu_sim")

    def observe(self, fill_history: bool = False) -> np.ndarray:
        parts = []
        for t in self.meta.observation.terms:
            x = self.term_value(t.name, t.params)
            if t.clip is not None:
                x = np.clip(x, t.clip[0], t.clip[1])
            x = x * np.asarray(t.scale_vector())
            buf = self.history[t.name]
            if fill_history:  # mjlab back-fills a fresh history with the first frame
                buf.clear()
                buf.extend([x] * buf.maxlen)
            else:
                buf.append(x)
            parts.extend(buf)  # oldest -> newest, term-major
        return np.concatenate(parts).astype(np.float32)

    # -- stepping ------------------------------------------------------------
    def step(self, obs: np.ndarray) -> np.ndarray:
        action = np.asarray(self.policy(obs), dtype=np.float64).reshape(-1)
        if action.shape != (self.meta.action.size,):
            raise ValueError(f"policy returned shape {action.shape}")
        if self.meta.action.clip is not None:
            action = np.clip(action, *self.meta.action.clip)
        target = self.default_q + self.action_scale * action
        self.data.ctrl[self.jmap.actuator_id] = target
        for _ in range(self.meta.control.decimation):
            mujoco.mj_step(self.model, self.data)
        self.last_action = action
        self.step_count += 1
        return self.observe()

    def fallen(self) -> bool:
        up_z = self.base_rotation()[2, 2]  # cos(tilt)
        return up_z < math.cos(math.radians(FALL_TILT_DEG)) or self.data.qpos[2] < FALL_MIN_HEIGHT

    def run(
        self, seconds: float, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0
    ) -> RolloutSummary:
        self.set_command(vx, vy, wz)
        obs = self.reset()
        n = max(1, round(seconds / self.meta.control.control_dt))
        start = self.data.qpos[:2].copy()
        v_sum = np.zeros(2)
        wz_sum = lin_err = yaw_err = h_sum = 0.0
        fall_time = None
        k = 0
        for k in range(1, n + 1):
            obs = self.step(obs)
            ang, lin = self.base_velocity()
            v_sum += lin[:2]
            wz_sum += ang[2]
            lin_err += float(np.linalg.norm(self.command[:2] - lin[:2]))
            yaw_err += abs(self.command[2] - ang[2])
            h_sum += self.data.qpos[2]
            if self.fallen():
                fall_time = k * self.meta.control.control_dt
                break
        return RolloutSummary(
            seconds=k * self.meta.control.control_dt,
            steps=k,
            fell=fall_time is not None,
            fall_time=fall_time,
            command=self.command.tolist(),
            mean_lin_vel_body=(v_sum / k).tolist(),
            mean_yaw_rate=float(wz_sum / k),
            lin_vel_tracking_error=float(lin_err / k),
            yaw_rate_tracking_error=float(yaw_err / k),
            distance_xy=float(np.linalg.norm(self.data.qpos[:2] - start)),
            mean_base_height=float(h_sum / k),
        )


def run_policy(
    policy_path: str | Path,
    meta_path: str | Path,
    seconds: float = 10.0,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    xml_path: str | Path | None = None,
    training_dynamics: bool = True,
) -> RolloutSummary:
    meta = PolicyMetadata.load(meta_path)
    sim = Go2Sim(meta, OnnxPolicy(policy_path, meta), xml_path, training_dynamics)
    return sim.run(seconds, vx, vy, wz)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True, help="ONNX policy file")
    p.add_argument("--meta", required=True, help="contract metadata JSON")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--model", default=None, help="local go2.xml (default: cached Menagerie)")
    p.add_argument(
        "--menagerie-dynamics",
        action="store_true",
        help="keep Menagerie's passive joint damping/friction instead of the training values",
    )
    a = p.parse_args(argv)
    s = run_policy(a.policy, a.meta, a.seconds, a.vx, a.vy, a.wz, a.model, not a.menagerie_dynamics)
    print(json.dumps(asdict(s), indent=2))


if __name__ == "__main__":
    main()
