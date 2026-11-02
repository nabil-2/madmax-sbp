"""Calibrated step-to-position conversion for each spider sub-thread.

Ported from spider-bead-pull-ai. Endpoint coordinates are in metres.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

Xyz = tuple[float, float, float]
CALIBRATION_SCHEMA_VERSION = 2
DEFAULT_CALIBRATION_PATH = Path("config/bead_pull_calibration.json")
IMPLAUSIBLE_SUB_THREAD_LENGTH_M = 5.0


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_xyz(value: Any, which: str) -> Xyz:
    """Parse a required (x, y, z) triple; raise if it is missing or malformed.

    The 3-D endpoints are no longer optional: the step<->metre conversion of a
    sub-thread is derived from the distance between them, so both are mandatory.
    """
    if value is None:
        raise ValueError(f"{which}_xyz_m is required (used to derive the step<->metre scale)")
    xyz = tuple(float(c) for c in value)
    if len(xyz) != 3:
        raise ValueError(f"expected an (x, y, z) triple for {which}_xyz_m, got {value!r}")
    return xyz


@dataclass(frozen=True)
class SubThreadCalibration:
    "One pass of the thread through the booster."

    index: int
    start_steps: int
    end_steps: int
    start_xyz_m: Xyz
    end_xyz_m: Xyz
    name: str | None = None
    margin_start_m: float = 0.0
    margin_end_m: float = 0.0

    @property
    def direction(self) -> int:
        """+1 if moving into the booster increases the step count, else -1."""
        return 1 if self.end_steps >= self.start_steps else -1

    @property
    def length_steps(self) -> int:
        return abs(self.end_steps - self.start_steps)

    @property
    def length_m(self) -> float:
        """Physical length of the pass: the Euclidean distance between its two 3-D
        endpoints, in metres (the pass is modelled as the straight, taut segment
        between them)."""
        return math.dist(self.start_xyz_m, self.end_xyz_m)

    @property
    def steps_per_meter(self) -> float:
        """Step<->metre conversion for *this* sub-thread, derived from its own
        anchors: the motor step span divided by the physical length between the two
        3-D endpoints (``steps = metres * steps_per_meter``)."""
        return self.length_steps / self.length_m

    @property
    def label(self) -> str:
        return self.name if self.name else f"sub_thread_{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "margin_start_m": self.margin_start_m,
            "margin_end_m": self.margin_end_m,
            "start_xyz_m": list(self.start_xyz_m),
            "end_xyz_m": list(self.end_xyz_m),
            "start_steps": self.start_steps,
            "end_steps": self.end_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubThreadCalibration":
        return cls(
            index=int(data["index"]),
            start_steps=int(data["start_steps"]),
            end_steps=int(data["end_steps"]),
            start_xyz_m=_as_xyz(data.get("start_xyz_m"), "start"),
            end_xyz_m=_as_xyz(data.get("end_xyz_m"), "end"),
            name=data.get("name"),
            margin_start_m=float(data.get("margin_start_m", 0.0)),
            margin_end_m=float(data.get("margin_end_m", 0.0)),
        )


@dataclass
class Calibration:
    "The measured sub-thread anchors."

    sub_threads: list[SubThreadCalibration]
    created_utc: str = field(default_factory=_utcnow)
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    @property
    def n_sub_threads(self) -> int:
        return len(self.sub_threads)

    def indices(self) -> list[int]:
        return [s.index for s in self.sub_threads]

    def get(self, index: int) -> SubThreadCalibration:
        for sub in self.sub_threads:
            if sub.index == index:
                return sub
        raise KeyError(f"No sub-thread with index {index}; have {self.indices()}")

    def length_m(self, index: int) -> float:
        return self.get(index).length_m

    def steps_for_position(self, index: int, x_m: float) -> int:
        """Absolute motor step count for bead position ``x_m`` (metres from zero)."""
        sub = self.get(index)
        return sub.start_steps + sub.direction * int(round(x_m * sub.steps_per_meter))

    def position_for_steps(self, index: int, steps: int) -> float:
        """Bead position (metres from zero) for an absolute step count."""
        sub = self.get(index)
        return (steps - sub.start_steps) * sub.direction / sub.steps_per_meter

    def booster_xyz(self, index: int, position_m: float) -> "Xyz":
        "3-D booster-coordinate position (metres) of a bead at ``position_m`` along a sub-thread."
        sub = self.get(index)
        length_m = sub.length_m
        frac = 0.0 if length_m == 0 else position_m / length_m
        s, e = sub.start_xyz_m, sub.end_xyz_m
        return (
            s[0] + frac * (e[0] - s[0]),
            s[1] + frac * (e[1] - s[1]),
            s[2] + frac * (e[2] - s[2]),
        )

    def scan_points_m(
        self,
        index: int,
        step_size_m: float,
        length_m: float | None = None,
        include_endpoint: bool = True,
    ) -> np.ndarray:
        "Bead positions (metres from zero) to measure within a sub-thread: ``margin_start_m``, +step, +2*step, ... up to ``length - margin_end_m`` (calibrated length unless overridden). The non-measurement margins are skipped at both ends."
        if step_size_m <= 0:
            raise ValueError("step_size_m must be positive")
        sub = self.get(index)
        span = self.length_m(index) if length_m is None else float(length_m)
        if span < 0:
            raise ValueError("length_m must be non-negative")
        lo = sub.margin_start_m
        hi = span - sub.margin_end_m
        if hi < lo:
            raise ValueError(
                f"Sub-thread {sub.label}: margins "
                f"({sub.margin_start_m} + {sub.margin_end_m} m) leave no room in the "
                f"{span:.4f} m scan."
            )
        eps = step_size_m * 1e-9
        # round away arange's float noise (sub-nm precision is irrelevant here)
        points = [float(x) for x in np.round(np.arange(lo, hi + eps, step_size_m), 9)]
        on_endpoint = bool(points) and abs(points[-1] - hi) <= eps
        if include_endpoint:
            if not on_endpoint:
                points.append(round(hi, 9))
        elif on_endpoint and len(points) > 1:
            points.pop()
        return np.asarray(points, dtype=float)

    def validate(self) -> None:
        seen: set[int] = set()
        for sub in self.sub_threads:
            if sub.index in seen:
                raise ValueError(f"Duplicate sub-thread index {sub.index}")
            seen.add(sub.index)
            if sub.length_steps == 0:
                raise ValueError(
                    f"Sub-thread {sub.label} has start_steps == end_steps "
                    "(zero step span); re-calibrate its end point."
                )
            if sub.margin_start_m < 0 or sub.margin_end_m < 0:
                raise ValueError(f"Sub-thread {sub.label} has a negative margin.")
            for xyz, which in ((sub.start_xyz_m, "start"), (sub.end_xyz_m, "end")):
                if len(xyz) != 3:
                    raise ValueError(
                        f"Sub-thread {sub.label}: {which}_xyz_m must have 3 components."
                    )
            if sub.length_m == 0:
                raise ValueError(
                    f"Sub-thread {sub.label} has coincident start_xyz_m and end_xyz_m "
                    "(zero physical length); the step<->metre scale is undefined."
                )
            if sub.length_m > IMPLAUSIBLE_SUB_THREAD_LENGTH_M:
                # Almost always a unit slip: coordinates entered in centimetres or
                # millimetres into fields that are metres.  It cannot be an error --
                # the code has no way to know the booster's size -- but it silently
                # rescales steps_per_meter by 100x or 1000x, so say so loudly.
                print(
                    f"WARNING: sub-thread {sub.label} spans {sub.length_m:.3f} m "
                    f"between its 3-D endpoints, which is longer than any plausible "
                    f"booster pass ({IMPLAUSIBLE_SUB_THREAD_LENGTH_M} m). Check that "
                    "start_xyz_m / end_xyz_m are in METRES -- they set this "
                    f"sub-thread's scale ({sub.steps_per_meter:.6g} steps/m)."
                )

    def to_dict(self) -> dict[str, Any]:
        # Calibration file = measured data only (step anchors) plus provenance.
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "sub_threads": [s.to_dict() for s in self.sub_threads],
        }

    def save(self, path: str | Path = DEFAULT_CALIBRATION_PATH) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_calibration_file(cls, path: str | Path) -> "Calibration":
        """Build a calibration from the sub-thread anchors in ``path`` (index,
        name, start/end steps and start/end 3-D positions).  The step<->metre
        conversion is derived per sub-thread from those anchors, so nothing else is
        needed."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        subs = sorted(
            (SubThreadCalibration.from_dict(entry) for entry in data["sub_threads"]),
            key=lambda s: s.index,
        )
        cal = cls(
            sub_threads=subs,
            created_utc=data.get("created_utc", _utcnow()),
            schema_version=int(data.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
        )
        cal.validate()
        return cal

    def summary(self) -> str:
        lines = [
            f"Calibration ({self.n_sub_threads} sub-threads; "
            "steps_per_meter derived per sub-thread)",
            f"{'idx':>3}  {'name':<12} {'start':>9} {'end':>9} "
            f"{'dir':>3} {'len_m':>8} {'steps/m':>10} {'mrg_s':>7} {'mrg_e':>7}",
        ]
        for sub in sorted(self.sub_threads, key=lambda s: s.index):
            lines.append(
                f"{sub.index:>3}  {sub.label:<12} {sub.start_steps:>9} "
                f"{sub.end_steps:>9} {sub.direction:>3} {sub.length_m:>8.4f} "
                f"{sub.steps_per_meter:>10.6g} "
                f"{sub.margin_start_m:>7.4f} {sub.margin_end_m:>7.4f}"
            )
        return "\n".join(lines)


def capture_endpoint(motor, captured, index, which):
    """Save the current logical step count in captured[index]['start/end_steps']."""
    if which not in ("start", "end"):
        raise ValueError("which must be 'start' or 'end'.")
    position = motor.get_position()
    captured.setdefault(index, {})[f"{which}_steps"] = position
    print(f"Sub-thread {index}: {which} = {position} steps")
    return position


def make_calibration(sub_threads, captured, coordinate_unit_m=1.0):
    """Combine entered XYZ endpoints and captured step counts; store SI units."""
    if not math.isfinite(coordinate_unit_m) or coordinate_unit_m <= 0:
        raise ValueError("coordinate_unit_m must be finite and positive.")
    result = []
    for entry in sub_threads:
        index = entry["index"]
        if index not in captured or not {"start_steps", "end_steps"} <= captured[index].keys():
            raise ValueError(f"Sub-thread {index} needs start and end step captures.")
        result.append(
            SubThreadCalibration(
                index=index,
                name=entry.get("name"),
                start_steps=captured[index]["start_steps"],
                end_steps=captured[index]["end_steps"],
                start_xyz_m=tuple(float(x) * coordinate_unit_m for x in entry["start_xyz"]),
                end_xyz_m=tuple(float(x) * coordinate_unit_m for x in entry["end_xyz"]),
                margin_start_m=entry.get("margin_start_m", 0.0),
                margin_end_m=entry.get("margin_end_m", 0.0),
            )
        )
    calibration = Calibration(sub_threads=result)
    calibration.validate()
    return calibration
