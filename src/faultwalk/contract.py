"""Policy contract: the metadata every faultwalk policy ships with.

A policy is an ONNX file plus a JSON metadata file. The JSON says exactly how
to build the observation vector, how to turn the network output into joint
targets, which PD gains and control rate the policy was trained with, and which
outputs the ONNX graph exposes. Everything downstream (CPU MuJoCo evaluation,
browser demo, ROS 2 node) is written against this module, never against a
particular training run. See ``docs/policy_contract.md`` for the prose version.

Plain Python only. ``onnx`` / ``onnxruntime`` are imported lazily, and only by
:func:`check_onnx`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "0.1.0"
"""Semantic version of the contract format. Major bumps break readers."""

NUM_JOINTS = 12

GO2_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)  # fmt: skip
"""Go2 joint names. A policy may use any order of these; it must list it."""

DIAGNOSIS_CLASSES: tuple[str, ...] = ("healthy", "locked", "limp", "weak")
"""Provisional per-joint fault classes for the reserved ``diagnosis`` output."""

# Observation terms the CPU simulator knows how to build. Size is per frame;
# ``None`` means "number of joints".
KNOWN_TERMS: dict[str, int | None] = {
    "base_lin_vel": 3,  # IMU-site linear velocity, IMU frame (m/s)
    "base_ang_vel": 3,  # IMU-site angular velocity, IMU frame (rad/s)
    "projected_gravity": 3,  # unit gravity vector in base frame
    "command": 3,  # (vx, vy, wz) commanded, base frame
    # (sin, cos) of a gait clock; params: period (s); zero when |cmd| < standing_threshold
    "phase": 2,
    "joint_pos": None,  # q - default_q, policy joint order
    "joint_vel": None,  # qdot, policy joint order
    "actions": None,  # previous raw network output (before scale/offset)
}


class ContractError(ValueError):
    """Raised when metadata or an ONNX model does not satisfy the contract."""


@dataclass
class ObservationTerm:
    name: str
    size: int
    scale: float | list[float] = 1.0
    clip: list[float] | None = None  # [lo, hi], applied BEFORE scale (mjlab order)
    params: dict[str, Any] = field(default_factory=dict)

    def scale_vector(self) -> list[float]:
        if isinstance(self.scale, (int, float)):
            return [float(self.scale)] * self.size
        return [float(s) for s in self.scale]


@dataclass
class ObservationSpec:
    terms: list[ObservationTerm]
    history_length: int = 1
    # mjlab flattens each term's history (oldest -> newest) and then
    # concatenates terms, so the layout is term-major.
    history_layout: str = "term_major_oldest_first"

    @property
    def frame_size(self) -> int:
        return sum(t.size for t in self.terms)

    @property
    def size(self) -> int:
        return self.frame_size * self.history_length

    def slices(self) -> dict[str, slice]:
        """Slice of each term inside the full (history-flattened) vector."""
        out: dict[str, slice] = {}
        start = 0
        for t in self.terms:
            n = t.size * self.history_length
            out[t.name] = slice(start, start + n)
            start += n
        return out


@dataclass
class ActionSpec:
    size: int
    scale: list[float]
    # target = default_joint_pos + scale * action  (mjlab JointPositionAction
    # with use_default_offset=True). ``clip`` is applied to the raw action.
    clip: list[float] | None = None


@dataclass
class PDGains:
    stiffness: list[float]
    damping: list[float]
    effort_limit: list[float] | None = None  # |torque| bound per joint (N m)
    armature: list[float] | None = None  # reflected rotor inertia per joint


@dataclass
class ControlSpec:
    physics_dt: float
    decimation: int

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation


@dataclass
class DiagnosisSpec:
    """Reserved second output. Shape and meaning are provisional (contract 0.x)."""

    output_name: str = "diagnosis"
    shape: list[int] = field(default_factory=lambda: [NUM_JOINTS, len(DIAGNOSIS_CLASSES)])
    classes: list[str] = field(default_factory=lambda: list(DIAGNOSIS_CLASSES))
    kind: str = "logits"
    status: str = "reserved"


@dataclass
class OnnxIO:
    input_name: str = "obs"
    action_output: str = "actions"


@dataclass
class PolicyMetadata:
    joint_names: list[str]
    default_joint_pos: list[float]
    action: ActionSpec
    pd: PDGains
    control: ControlSpec
    observation: ObservationSpec
    onnx: OnnxIO = field(default_factory=OnnxIO)
    diagnosis: DiagnosisSpec | None = None
    robot: str = "unitree_go2"
    contract_version: str = CONTRACT_VERSION
    source: dict[str, Any] = field(default_factory=dict)  # free-form provenance

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["control"]["control_dt"] = self.control.control_dt
        d["observation"]["size"] = self.observation.size
        return d

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyMetadata:
        try:
            obs = d["observation"]
            ctrl = d["control"]
            meta = cls(
                joint_names=list(d["joint_names"]),
                default_joint_pos=[float(x) for x in d["default_joint_pos"]],
                action=ActionSpec(
                    size=int(d["action"]["size"]),
                    scale=_floats(d["action"]["scale"]),
                    clip=_opt_floats(d["action"].get("clip")),
                ),
                pd=PDGains(
                    stiffness=_floats(d["pd"]["stiffness"]),
                    damping=_floats(d["pd"]["damping"]),
                    effort_limit=_opt_floats(d["pd"].get("effort_limit")),
                    armature=_opt_floats(d["pd"].get("armature")),
                ),
                control=ControlSpec(
                    physics_dt=float(ctrl["physics_dt"]), decimation=int(ctrl["decimation"])
                ),
                observation=ObservationSpec(
                    terms=[
                        ObservationTerm(
                            name=t["name"],
                            size=int(t["size"]),
                            scale=t.get("scale", 1.0),
                            clip=_opt_floats(t.get("clip")),
                            params=dict(t.get("params", {})),
                        )
                        for t in obs["terms"]
                    ],
                    history_length=int(obs.get("history_length", 1)),
                    history_layout=obs.get("history_layout", "term_major_oldest_first"),
                ),
                onnx=OnnxIO(**d.get("onnx", {})),
                diagnosis=DiagnosisSpec(**d["diagnosis"]) if d.get("diagnosis") else None,
                robot=d.get("robot", "unitree_go2"),
                contract_version=str(d.get("contract_version", "")),
                source=dict(d.get("source", {})),
            )
        except (KeyError, TypeError) as e:
            raise ContractError(f"malformed metadata: {e!r}") from e
        meta.validate()
        # Derived fields, if present, must agree with the primary ones.
        if "control_dt" in ctrl and not math.isclose(
            float(ctrl["control_dt"]), meta.control.control_dt, rel_tol=1e-6
        ):
            raise ContractError("control.control_dt != physics_dt * decimation")
        if "size" in obs and int(obs["size"]) != meta.observation.size:
            raise ContractError(
                f"observation.size {obs['size']} != computed {meta.observation.size}"
            )
        return meta

    @classmethod
    def load(cls, path: str | Path) -> PolicyMetadata:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------ validate
    def validate(self, allow_unknown_terms: bool = False) -> None:
        err = _Errors()
        major = self.contract_version.split(".")[0]
        if major != CONTRACT_VERSION.split(".")[0]:
            err.add(
                f"contract_version {self.contract_version!r} incompatible with {CONTRACT_VERSION}"
            )

        n = len(self.joint_names)
        if n != NUM_JOINTS:
            err.add(f"expected {NUM_JOINTS} joints, got {n}")
        if len(set(self.joint_names)) != n:
            err.add("joint_names contains duplicates")
        if self.robot == "unitree_go2":
            unknown = set(self.joint_names) - set(GO2_JOINT_NAMES)
            if unknown:
                err.add(f"joint names not in the Go2 model: {sorted(unknown)}")

        for name, vec in (
            ("default_joint_pos", self.default_joint_pos),
            ("action.scale", self.action.scale),
            ("pd.stiffness", self.pd.stiffness),
            ("pd.damping", self.pd.damping),
            ("pd.effort_limit", self.pd.effort_limit),
            ("pd.armature", self.pd.armature),
        ):
            if vec is None:
                continue
            if len(vec) != n:
                err.add(f"{name} has length {len(vec)}, expected {n}")
            if not all(math.isfinite(v) for v in vec):
                err.add(f"{name} contains non-finite values")
        if self.action.size != n:
            err.add(f"action.size {self.action.size} != number of joints {n}")
        if any(v <= 0 for v in self.pd.stiffness):
            err.add("pd.stiffness must be > 0")
        if any(v < 0 for v in self.pd.damping):
            err.add("pd.damping must be >= 0")
        if self.pd.effort_limit is not None and any(v <= 0 for v in self.pd.effort_limit):
            err.add("pd.effort_limit must be > 0")
        if self.action.clip is not None and len(self.action.clip) != 2:
            err.add("action.clip must be [lo, hi]")

        if not (0 < self.control.physics_dt < 0.1):
            err.add(f"control.physics_dt {self.control.physics_dt} out of range")
        if self.control.decimation < 1:
            err.add("control.decimation must be >= 1")

        obs = self.observation
        if obs.history_length < 1:
            err.add("observation.history_length must be >= 1")
        if obs.history_layout != "term_major_oldest_first":
            err.add(f"unsupported history_layout {obs.history_layout!r}")
        if not obs.terms:
            err.add("observation.terms is empty")
        seen: set[str] = set()
        for t in obs.terms:
            if t.name in seen:
                err.add(f"duplicate observation term {t.name!r}")
            seen.add(t.name)
            if t.size < 1:
                err.add(f"term {t.name!r} has size {t.size}")
            expected = KNOWN_TERMS.get(t.name, -1)
            if expected == -1:
                if not allow_unknown_terms:
                    err.add(f"unknown observation term {t.name!r} (known: {sorted(KNOWN_TERMS)})")
            elif (expected or n) != t.size:
                err.add(f"term {t.name!r} has size {t.size}, expected {expected or n}")
            if not isinstance(t.scale, (int, float)) and len(t.scale) != t.size:
                err.add(f"term {t.name!r} scale length {len(t.scale)} != size {t.size}")
            if t.clip is not None and (len(t.clip) != 2 or t.clip[0] > t.clip[1]):
                err.add(f"term {t.name!r} clip must be [lo, hi]")
            if t.name == "phase" and float(t.params.get("period", 0)) <= 0:
                err.add("term 'phase' needs params.period > 0")

        if self.diagnosis is not None:
            dg = self.diagnosis
            if dg.output_name == self.onnx.action_output:
                err.add("diagnosis output name collides with the action output")
            if not dg.shape or any(int(s) < 1 for s in dg.shape):
                err.add(f"diagnosis.shape {dg.shape} invalid")
            if dg.shape and dg.shape[-1] != len(dg.classes):
                err.add("diagnosis.shape[-1] must equal len(diagnosis.classes)")
        err.raise_if_any()

    def joint_index(self, names: Sequence[str]) -> list[int]:
        """For each name in ``names``, its index in the policy joint order."""
        pos = {nm: i for i, nm in enumerate(self.joint_names)}
        missing = [nm for nm in names if nm not in pos]
        if missing:
            raise ContractError(f"joints not in policy order: {missing}")
        return [pos[nm] for nm in names]


# ---------------------------------------------------------------- onnx check
@dataclass
class OnnxSignature:
    inputs: dict[str, list[int | str | None]]
    outputs: dict[str, list[int | str | None]]


def onnx_signature(model_path: str | Path) -> OnnxSignature:
    """Read input/output names and shapes (``None``/str = dynamic) from an ONNX file."""
    try:
        import onnx

        m = onnx.load(str(model_path), load_external_data=False)

        def dims(vi) -> list[int | str | None]:
            out: list[int | str | None] = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    out.append(int(d.dim_value))
                else:
                    out.append(d.dim_param or None)
            return out

        inits = {i.name for i in m.graph.initializer}
        return OnnxSignature(
            inputs={i.name: dims(i) for i in m.graph.input if i.name not in inits},
            outputs={o.name: dims(o) for o in m.graph.output},
        )
    except ImportError:
        import onnxruntime as ort

        s = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return OnnxSignature(
            inputs={i.name: list(i.shape) for i in s.get_inputs()},
            outputs={o.name: list(o.shape) for o in s.get_outputs()},
        )


def check_onnx(meta: PolicyMetadata, model_path: str | Path) -> OnnxSignature:
    """Raise :class:`ContractError` unless the ONNX I/O matches the metadata.

    Rules: exactly one input, named ``meta.onnx.input_name``, shape
    ``[batch, observation.size]``; an output ``meta.onnx.action_output`` of shape
    ``[batch, action.size]``; a ``diagnosis`` output exists iff the metadata
    declares one, with shape ``[batch, *diagnosis.shape]``. ``batch`` may be 1
    or dynamic. No other outputs.
    """
    sig = onnx_signature(model_path)
    err = _Errors()
    if list(sig.inputs) != [meta.onnx.input_name]:
        err.add(f"inputs {list(sig.inputs)} != [{meta.onnx.input_name!r}]")
    else:
        _check_shape(err, "input", sig.inputs[meta.onnx.input_name], [meta.observation.size])

    if meta.onnx.action_output not in sig.outputs:
        err.add(f"missing action output {meta.onnx.action_output!r}; have {list(sig.outputs)}")
    else:
        _check_shape(err, "action output", sig.outputs[meta.onnx.action_output], [meta.action.size])

    expected = {meta.onnx.action_output}
    if meta.diagnosis is not None:
        name = meta.diagnosis.output_name
        expected.add(name)
        if name not in sig.outputs:
            err.add(f"metadata declares diagnosis output {name!r} but the model lacks it")
        else:
            _check_shape(err, "diagnosis output", sig.outputs[name], list(meta.diagnosis.shape))
    extra = set(sig.outputs) - expected
    if extra:
        err.add(f"unexpected outputs {sorted(extra)} (a diagnosis output must be declared)")
    err.raise_if_any()
    return sig


# ------------------------------------------------------------------ helpers
def _check_shape(err: _Errors, what: str, shape: list, tail: list[int]) -> None:
    if len(shape) != 1 + len(tail):
        err.add(f"{what} rank {len(shape)} != {1 + len(tail)} (shape {shape})")
        return
    batch = shape[0]
    if isinstance(batch, int) and batch != 1:
        err.add(f"{what} batch dim {batch} must be 1 or dynamic")
    for got, want in zip(shape[1:], tail, strict=True):
        if got != want:
            err.add(f"{what} shape {shape} != [batch, {', '.join(map(str, tail))}]")
            return


def _floats(v: Any) -> list[float]:
    return [float(x) for x in v]


def _opt_floats(v: Any) -> list[float] | None:
    return None if v is None else _floats(v)


class _Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, msg: str) -> None:
        self.items.append(msg)

    def raise_if_any(self) -> None:
        if self.items:
            raise ContractError("; ".join(self.items))
