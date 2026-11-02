"""Plan and acquire calibrated SBP scans, with VNA traces and position logs.

Run from the repository root: python -m src.acquisition --help
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import argparse
import csv
import dataclasses
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Sequence
import numpy as np
from .calibration import Calibration, SubThreadCalibration, Xyz, _utcnow
from .motor import EmergencyStop, SimulatedMotor, Stepper, unwind_direction_sign

DEFAULT_CONFIG = "config/measurement_config.json"


@dataclass
class StepperConfig:
    port: str | None = None  # serial port, e.g. "/dev/ttyUSB0" or "COM5"
    baud: int = 9600
    motor: int = 1
    configure: bool = True  # push profile/voltage on connect
    microsteps: int = 8
    acceleration: int = 1000
    speed: int = 500
    driving_voltage: float = 6.8
    holding_voltage: float = 2.0
    # physical rotation that unwinds the thread and advances the bead:
    # "clockwise" or "counterclockwise" (aliases "cw"/"ccw"; +1/-1 also accepted).
    unwind_direction: str = "counterclockwise"
    # Persisted motor position + home file; null => the default shared absolute
    # path (~/.madmax_bead_pull/motor_state.json), so every program agrees.
    state_file: str | None = None
    # Refuse a single move larger than this many steps (a runaway circuit-breaker,
    # since this rig has no limit switch); null disables the check.
    max_step_magnitude: int | None = 500_000
    # Soft end-stops: the absolute step range the bead may occupy.  Any move that
    # would leave it is refused before the motor is commanded.  null on either
    # side => unbounded there.  Establish them with the calibration notebook.
    min_position_steps: int | None = None
    max_position_steps: int | None = None

    def __post_init__(self) -> None:
        # fail fast on a bad value rather than at motor-open time
        unwind_direction_sign(self.unwind_direction)
        if (
            self.min_position_steps is not None
            and self.max_position_steps is not None
            and self.min_position_steps > self.max_position_steps
        ):
            raise ValueError(
                f"stepper.min_position_steps ({self.min_position_steps}) must not "
                f"exceed stepper.max_position_steps ({self.max_position_steps})."
            )


@dataclass
class VnaConfig:
    ip_address: str = "169.254.127.120"
    channel: int = 1
    port_a: int = 2  # first index of S<port_a><port_b>
    port_b: int = 2  # second index of S<port_a><port_b>
    start_freq_Hz: float = 18e9
    stop_freq_Hz: float = 24e9
    points: int = 1001
    averages: int = 1

    def __post_init__(self) -> None:
        # Catch these at config-load time: every one of them would otherwise
        # surface only once the instrument is open and the bead already moving.
        if self.points < 2:
            raise ValueError(f"vna.points must be at least 2, got {self.points}.")
        if not self.start_freq_Hz < self.stop_freq_Hz:
            raise ValueError(
                f"vna.start_freq_Hz ({self.start_freq_Hz}) must be below "
                f"vna.stop_freq_Hz ({self.stop_freq_Hz})."
            )
        if self.averages < 1:
            raise ValueError(f"vna.averages must be at least 1, got {self.averages}.")


@dataclass
class ScanConfig:
    step_size_m: float = 0.01
    settle_s: float = 0.5
    sub_thread_indices: list[int] | None = None  # None => all calibrated sub-threads
    length_m: float | None = None  # None => calibrated length per sub-thread
    include_endpoint: bool = True
    home_first: bool = True

    def __post_init__(self) -> None:
        if self.step_size_m <= 0:
            raise ValueError(f"scan.step_size_m must be positive, got {self.step_size_m}.")
        if self.settle_s < 0:
            raise ValueError(f"scan.settle_s must not be negative, got {self.settle_s}.")
        if self.length_m is not None and self.length_m <= 0:
            raise ValueError(f"scan.length_m must be positive when set, got {self.length_m}.")


@dataclass
class MeasurementConfig:
    calibration_file: str = "config/bead_pull_calibration.json"
    # disk spacings (mm) of the setup under test and a free-text note; recorded
    # verbatim in the run metadata so each run is self-describing.
    disk_spacings_mm: list[float] = field(default_factory=list)
    comment: str = ""
    output_root: str = "data/bead_pull_runs"
    output_dir: str | None = None  # None => timestamped under output_root
    scan: ScanConfig = field(default_factory=ScanConfig)
    stepper: StepperConfig = field(default_factory=StepperConfig)
    vna: VnaConfig = field(default_factory=VnaConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "MeasurementConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeasurementConfig":
        output = data.get("output") or {}
        return cls(
            calibration_file=data.get("calibration_file", cls.calibration_file),
            disk_spacings_mm=list(data.get("disk_spacings_mm") or []),
            comment=data.get("comment", ""),
            output_root=output.get("root", cls.output_root),
            output_dir=output.get("directory"),
            scan=_build(ScanConfig, data.get("scan")),
            stepper=_build(StepperConfig, data.get("stepper")),
            vna=_build(VnaConfig, data.get("vna")),
        )

    def resolve_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return Path(self.output_root) / f"run_{stamp}"


def _build(dc_cls, data: dict[str, Any] | None):
    """Build a dataclass from a config sub-dict, ignoring (and warning about)
    unknown keys so a typo cannot silently change a setting elsewhere."""
    data = data or {}
    known = {f.name for f in dataclasses.fields(dc_cls)}
    unknown = set(data) - known
    if unknown:
        print(f"WARNING: ignoring unknown {dc_cls.__name__} keys: {sorted(unknown)}")
    return dc_cls(**{k: v for k, v in data.items() if k in known})


_LOG_COLUMNS = [
    "sub_thread_index",
    "sub_thread_name",
    "point_index",
    "n_points",
    "position_m",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "timestamp_utc",
    "measurement_json",
]


@dataclass
class ScanPoint:
    "One bead position the controller stops at and measures."

    sub_thread_index: int
    sub_thread_name: str | None
    point_index: int
    n_points: int
    position_m: float
    position_xyz_m: "Xyz"
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_thread_index": self.sub_thread_index,
            "sub_thread_name": self.sub_thread_name,
            "point_index": self.point_index,
            "n_points": self.n_points,
            "position_m": self.position_m,
            "position_xyz_m": list(self.position_xyz_m),
            "timestamp_utc": self.timestamp_utc,
        }


def scan_targets(
    calibration: Calibration,
    step_size_m: float,
    sub_thread_indices: Sequence[int] | None = None,
    lengths_m: dict[int, float] | None = None,
    include_endpoint: bool = True,
) -> Iterator[tuple[SubThreadCalibration, int, int, float, int]]:
    """Yield ``(sub, point_index, n_points, x_m, target_steps)`` for every planned
    bead position, without moving anything."""
    indices = (
        list(sub_thread_indices)
        if sub_thread_indices is not None
        else sorted(calibration.indices())
    )
    lengths_m = lengths_m or {}
    for index in indices:
        sub = calibration.get(index)
        points = calibration.scan_points_m(
            index, step_size_m, lengths_m.get(index), include_endpoint
        )
        for i, x_m in enumerate(points):
            yield sub, i, len(points), float(x_m), calibration.steps_for_position(index, float(x_m))


class BeadPullController:
    """Drives the bead to calibrated positions on a motor."""

    def __init__(
        self,
        motor,
        calibration: Calibration,
        settle_s: float = 0.0,
        position_tolerance_steps: int | None = 5,
        logger=print,
    ) -> None:
        calibration.validate()
        self.motor = motor
        self.calibration = calibration
        self.settle_s = settle_s
        self.position_tolerance_steps = position_tolerance_steps
        self.log = logger

    def home(self) -> None:
        self.log("Homing to the saved home position ...")
        self.motor.home()
        self.log(f"Homed; position = {self.motor.get_position()} steps")

    def move_to(self, sub_thread_index: int, x_m: float) -> int:
        """Move the bead to ``x_m`` metres from a sub-thread's zero; return the
        actual motor position afterwards."""
        target = self.calibration.steps_for_position(sub_thread_index, x_m)
        self.motor.move_by(target - self.motor.get_position())
        if self.settle_s:
            time.sleep(self.settle_s)
        actual = self.motor.get_position()
        tol = self.position_tolerance_steps
        if tol is not None and abs(actual - target) > tol:
            self.log(
                f"WARNING: sub-thread {sub_thread_index} x={x_m:.4f} m: "
                f"target {target} steps but motor at {actual} (off by {actual - target})"
            )
        return actual

    def iter_points(
        self,
        step_size_m: float,
        sub_thread_indices: Sequence[int] | None = None,
        lengths_m: dict[int, float] | None = None,
        include_endpoint: bool = True,
    ) -> Iterator[ScanPoint]:
        """Walk every requested sub-thread point by point, yielding a
        :class:`ScanPoint` once the bead is parked at each position."""
        for sub, i, n, x_m, _target_steps in scan_targets(
            self.calibration, step_size_m, sub_thread_indices, lengths_m, include_endpoint
        ):
            if i == 0:
                self.log(f"Sub-thread {sub.index} ({sub.label}): {n} points")
            actual_steps = self.move_to(sub.index, x_m)
            # record the true position actually reached (steps -> metres), and the
            # same point mapped into the 3-D booster coordinate system
            position_m = self.calibration.position_for_steps(sub.index, actual_steps)
            position_xyz_m = self.calibration.booster_xyz(sub.index, position_m)
            yield ScanPoint(sub.index, sub.name, i, n, position_m, position_xyz_m, _utcnow())


def run_scan(
    controller: BeadPullController,
    measurement,
    step_size_m: float,
    output_dir: str | Path,
    sub_thread_indices: Sequence[int] | None = None,
    lengths_m: dict[int, float] | None = None,
    include_endpoint: bool = True,
    home_first: bool = True,
    metadata: dict[str, Any] | None = None,
    progress: bool = True,
    overwrite: bool = False,
) -> Path:
    "Walk every requested sub-thread in ``step_size_m`` increments, calling ``measurement.measure(point)`` at each parked position. Writes ``scan_log.csv``, ``scan_log.jsonl`` and ``scan_manifest.json`` into ``output_dir`` and returns it."
    output_dir = Path(output_dir)
    csv_path = output_dir / "scan_log.csv"
    jsonl_path = output_dir / "scan_log.jsonl"
    if not overwrite:
        existing = [p.name for p in (csv_path, jsonl_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"{output_dir} already contains a scan ({', '.join(existing)}). "
                "Point output.directory somewhere else, or pass overwrite=True "
                "(--overwrite) to replace it."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = sum(
        1
        for _ in scan_targets(
            controller.calibration, step_size_m, sub_thread_indices, lengths_m, include_endpoint
        )
    )
    manifest = {
        "started_utc": _utcnow(),
        "step_size_m": step_size_m,
        "include_endpoint": include_endpoint,
        "home_first": home_first,
        "n_points_planned": n_total,
        "sub_thread_indices": (
            list(sub_thread_indices)
            if sub_thread_indices is not None
            else sorted(controller.calibration.indices())
        ),
        "lengths_m_override": lengths_m or {},
        "calibration": {
            **controller.calibration.to_dict(),
            # step<->metre scale is derived per sub-thread (step span / 3-D length),
            # recorded here so the run is a self-contained record.
            "derived": [
                {
                    "index": sub.index,
                    "length_m": sub.length_m,
                    "steps_per_meter": sub.steps_per_meter,
                }
                for sub in controller.calibration.sub_threads
            ],
        },
        "metadata": metadata or {},
    }
    manifest_path = output_dir / "scan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    points = controller.iter_points(step_size_m, sub_thread_indices, lengths_m, include_endpoint)
    if progress:
        try:
            from tqdm import tqdm

            points = tqdm(points, total=n_total, desc="bead-pull scan", unit="pt")
        except ImportError:
            pass

    setup = getattr(measurement, "setup", None)
    teardown = getattr(measurement, "teardown", None)

    n_measured = 0
    status = "failed"
    try:
        # setup() is inside the try so that a failure part-way through it -- a VNA
        # that connects and then rejects the sweep, say -- still reaches
        # teardown() and releases the instrument.
        if callable(setup):
            setup()
        if home_first:
            controller.home()
        with (
            csv_path.open("w", newline="", encoding="utf-8") as csv_fh,
            jsonl_path.open("w", encoding="utf-8") as jsonl_fh,
        ):
            writer = csv.DictWriter(csv_fh, fieldnames=_LOG_COLUMNS)
            writer.writeheader()
            for point in points:
                result = measurement.measure(point)
                record = point.to_dict()
                # jsonl keeps the xyz triple as a list next to the full result
                jsonl_fh.write(json.dumps({**record, "measurement": result}) + "\n")
                jsonl_fh.flush()
                # csv flattens the xyz triple into three scalar columns
                x, y, z = record.pop("position_xyz_m") or (None, None, None)
                writer.writerow(
                    {
                        **record,
                        "position_x_m": x,
                        "position_y_m": y,
                        "position_z_m": z,
                        "measurement_json": json.dumps(result),
                    }
                )
                csv_fh.flush()
                n_measured += 1
        status = "completed"
    except (KeyboardInterrupt, EmergencyStop):
        status = "aborted"
        raise
    finally:
        if callable(teardown):
            teardown()
        # Always close the manifest out, so an interrupted run is still a
        # self-describing record of how far it got rather than one that merely
        # looks unfinished.
        manifest["finished_utc"] = _utcnow()
        manifest["status"] = status
        manifest["n_points_measured"] = n_measured
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


class VnaMeasurement:
    "Rohde & Schwarz VNA measurement of S<port_a><port_b> at each bead position."

    # Wait this long before re-reading a trace the instrument refused, so a
    # persistent fault is not hammered ten times in a few milliseconds.
    RETRY_DELAY_S = 0.5

    def __init__(
        self,
        output_dir: str | Path,
        ip_address: str = VnaConfig.ip_address,
        channel: int = VnaConfig.channel,
        port_a: int = VnaConfig.port_a,
        port_b: int = VnaConfig.port_b,
        start_freq: float = VnaConfig.start_freq_Hz,
        stop_freq: float = VnaConfig.stop_freq_Hz,
        points: int = VnaConfig.points,
        averages: int = VnaConfig.averages,
        testmode: bool = False,
        overwrite: bool = False,
        trace_filename: str = "s11_traces.h5",
        log_filename: str = "vna_log.txt",
        max_retries: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.ip_address = ip_address
        self.channel = int(channel)
        self.port_a = int(port_a)
        self.port_b = int(port_b)
        self.points = int(points)
        self.averages = int(averages)
        self.testmode = testmode
        self.trace_filename = trace_filename
        self.log_filename = log_filename
        self.max_retries = max_retries
        self.freqs = np.linspace(start_freq, stop_freq, self.points)
        self._vna = None
        self._h5 = None
        self._dset = None
        self._n = 0

    @property
    def s_param(self) -> str:
        return f"S{self.port_a}{self.port_b}"

    # -- lifecycle ---------------------------------------------------------
    def _sync_frequency_axis(self, channel) -> None:
        "Read the sweep back from the instrument and adopt what it really set."
        try:
            actual_points = int(channel.points)
            actual_start = float(channel.start_frequency_Hz)
            actual_stop = float(channel.stop_frequency_Hz)
        except Exception as exc:
            print(
                f"WARNING: could not read the sweep back from the VNA ({exc}); "
                "trusting the configured frequency axis."
            )
            return
        if (
            actual_points == self.points
            and math.isclose(actual_start, self.freqs[0], rel_tol=1e-9)
            and math.isclose(actual_stop, self.freqs[-1], rel_tol=1e-9)
        ):
            return
        print(
            "WARNING: the VNA set a different sweep from the one requested "
            f"({self.freqs[0]/1e9:.4f}-{self.freqs[-1]/1e9:.4f} GHz / {self.points} "
            f"points requested, {actual_start/1e9:.4f}-{actual_stop/1e9:.4f} GHz / "
            f"{actual_points} points reported). Recording the instrument's axis."
        )
        self.points = actual_points
        self.freqs = np.linspace(actual_start, actual_stop, actual_points)

    def setup(self) -> None:
        import h5py

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.trace_filename
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"{path} already exists. Point output.directory somewhere else, or "
                "pass --overwrite to replace the previous traces."
            )

        if self.testmode:
            print("VNA testmode: storing zero traces (no instrument connected).")
        else:
            from rohdeschwarz.instruments.vna import Vna

            self._vna = Vna()
            try:
                self._vna.open_tcp(self.ip_address)
                self._vna.open_log(str(self.output_dir / self.log_filename))
                self._vna.settings.output_power_on = True
                channel = self._vna.channel(self.channel)
                channel.start_frequency_Hz = self.freqs[0]
                channel.stop_frequency_Hz = self.freqs[-1]
                channel.points = len(self.freqs)
                channel.averages = self.averages
                channel.sweep_count = self.averages
                self._sync_frequency_axis(channel)
                print("Connected to VNA:", self._vna.query("*IDN?").strip())
            except BaseException:
                # Do not leave a half-configured instrument with its output power
                # on because setup failed after the connection came up.
                self.teardown()
                raise

        try:
            self._h5 = h5py.File(path, "w" if self.overwrite else "w-")
            self._h5.create_dataset("frequencies_Hz", data=self.freqs)
            self._dset = self._h5.create_dataset(
                "s_parameter",
                shape=(0, self.points),
                maxshape=(None, self.points),
                dtype=np.complex128,
                chunks=(1, self.points),
            )
        except BaseException:
            self.teardown()
            raise
        self._dset.attrs["s_param"] = self.s_param
        self._dset.attrs["port_a"] = self.port_a
        self._dset.attrs["port_b"] = self.port_b
        self._dset.attrs["channel"] = self.channel
        self._dset.attrs["averages"] = self.averages
        self._dset.attrs["testmode"] = self.testmode
        self._h5.flush()
        print(
            f"Writing complex {self.s_param} traces to {path} "
            f"({self.points} freq points, "
            f"{self.freqs[0]/1e9:.3f}-{self.freqs[-1]/1e9:.3f} GHz)."
        )

    def teardown(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception as exc:
                print(f"WARNING: could not close the trace file cleanly: {exc}")
            self._h5 = None
            self._dset = None
        if self._vna is not None:
            try:
                self._vna.settings.output_power_on = False
            except Exception:
                pass
            close = getattr(self._vna, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._vna = None

    # -- acquisition -------------------------------------------------------
    def _read_trace(self, attempt: int = 0):
        """Read the complex S<port_a><port_b> trace from the VNA (with retries)."""
        if self.testmode:
            return np.zeros(self.points, dtype=np.complex128)
        channel = self._vna.channel(self.channel)
        try:
            if channel.averages != self.averages or channel.sweep_count != self.averages:
                channel.averages = self.averages
                channel.sweep_count = self.averages
            # measure([p, q]) returns the S-matrix for those ports, shape (P, P, F),
            # with result[i, j] = S_{ports[i], ports[j]}. Pick out S_{port_a,port_b}.
            ports = [self.port_a] if self.port_a == self.port_b else [self.port_a, self.port_b]
            result = np.asarray(channel.measure(ports)).reshape(len(ports), len(ports), -1)
            ia = ports.index(self.port_a)
            ib = ports.index(self.port_b)
            return result[ia, ib, :].flatten()
        except KeyboardInterrupt:
            raise
        except Exception:
            is_error = getattr(self._vna, "is_error", None)
            if callable(is_error) and is_error():
                raise RuntimeError("VNA reported an error during measurement")
            print(
                f"VNA measurement glitch (attempt {attempt + 1}/" f"{self.max_retries + 1}):",
                getattr(self._vna, "errors", "?"),
            )
            if attempt < self.max_retries:
                clear = getattr(self._vna, "clear_status", None)
                if callable(clear):
                    clear()
                time.sleep(self.RETRY_DELAY_S)  # give the instrument time to recover
                return self._read_trace(attempt + 1)
            raise RuntimeError(f"VNA measurement failed after {self.max_retries + 1} attempts")

    def measure(self, point: ScanPoint) -> dict[str, Any]:
        trace = self._read_trace()
        if trace.shape[0] != self.points:
            raise ValueError(f"VNA returned {trace.shape[0]} points, expected {self.points}")
        row = self._n
        self._dset.resize(row + 1, axis=0)
        self._dset[row, :] = trace
        self._h5.flush()
        self._n += 1

        magnitude = np.abs(trace)
        peak = int(np.argmax(magnitude))
        return {
            "trace_index": row,
            "trace_file": self.trace_filename,
            "s_param": self.s_param,
            "n_freq": int(self.points),
            "abs_mean": float(magnitude.mean()),
            "abs_max": float(magnitude.max()),
            "peak_freq_Hz": float(self.freqs[peak]),
        }


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to the measurement config JSON")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned positions and exit (no motion, no measurement)",
    )
    p.add_argument("--simulate", action="store_true", help="use a simulated motor (no serial port)")
    p.add_argument(
        "--vna-testmode",
        action="store_true",
        help="store zero VNA traces (no instrument; pairs with --simulate for a full no-hardware run)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing scan in the output directory (refused by default)",
    )
    return p


def build_motor(cfg: MeasurementConfig, simulate: bool):
    st = cfg.stepper
    if simulate:
        # the simulated motor honours the same soft end-stops, so a dry run
        # exercises the limit checks exactly as a real one does
        return SimulatedMotor(
            verbose=True,
            min_position_steps=st.min_position_steps,
            max_position_steps=st.max_position_steps,
        )
    if not st.port:
        raise SystemExit(
            "ERROR: stepper.port is not set in the config (or use --simulate / --dry-run)."
        )
    return Stepper.open(
        st.port,
        baud=st.baud,
        motor=st.motor,
        configure=st.configure,
        microsteps=st.microsteps,
        acceleration=st.acceleration,
        speed=st.speed,
        driving_voltage=st.driving_voltage,
        holding_voltage=st.holding_voltage,
        unwind_direction=st.unwind_direction,
        state_path=st.state_file,
        max_step_magnitude=st.max_step_magnitude,
        min_position_steps=st.min_position_steps,
        max_position_steps=st.max_position_steps,
    )


def build_measurement(
    cfg: MeasurementConfig, output_dir: Path, testmode: bool, overwrite: bool = False
) -> VnaMeasurement:
    v = cfg.vna
    return VnaMeasurement(
        output_dir=output_dir,
        ip_address=v.ip_address,
        channel=v.channel,
        port_a=v.port_a,
        port_b=v.port_b,
        start_freq=v.start_freq_Hz,
        stop_freq=v.stop_freq_Hz,
        points=v.points,
        averages=v.averages,
        testmode=testmode,
        overwrite=overwrite,
    )


def load_calibration(cfg: MeasurementConfig) -> Calibration:
    """Load the sub-thread anchors (calibration file).  The step<->metre
    conversion is derived per sub-thread from those anchors, so nothing else is
    needed."""
    return Calibration.from_calibration_file(cfg.calibration_file)


def plan_scan(config):
    """Return planned positions/step targets without connecting to hardware."""
    calibration = load_calibration(config)
    scan = config.scan
    lengths = (
        None if scan.length_m is None else {index: scan.length_m for index in calibration.indices()}
    )
    return list(
        scan_targets(
            calibration, scan.step_size_m, scan.sub_thread_indices, lengths, scan.include_endpoint
        )
    )


def acquire(
    config, *, simulate=False, vna_testmode=False, overwrite=False, progress=True, config_path=None
):
    """Run one scan and return its output directory. Motor release is guaranteed.

    simulate selects the in-memory motor; vna_testmode selects zero VNA traces.
    Set both True for a complete acquisition run without instruments.
    """
    calibration = load_calibration(config)
    scan = config.scan
    lengths = (
        None if scan.length_m is None else {index: scan.length_m for index in calibration.indices()}
    )
    output = config.resolve_output_dir()
    measurement = build_measurement(config, output, vna_testmode, overwrite)
    motor = build_motor(config, simulate)
    try:
        controller = BeadPullController(motor, calibration, settle_s=scan.settle_s)
        return run_scan(
            controller,
            measurement,
            scan.step_size_m,
            output,
            sub_thread_indices=scan.sub_thread_indices,
            lengths_m=lengths,
            include_endpoint=scan.include_endpoint,
            home_first=scan.home_first,
            overwrite=overwrite,
            progress=progress,
            metadata={
                "config_file": str(config_path) if config_path is not None else None,
                "config": dataclasses.asdict(config),
                "disk_spacings_mm": config.disk_spacings_mm,
                "comment": config.comment,
                "simulate": simulate,
                "vna_testmode": vna_testmode,
            },
        )
    finally:
        motor.shutdown()


def summarize_run(directory):
    """Read a run's status and trace dimensions without loading all traces."""
    import h5py

    directory = Path(directory)
    manifest = json.loads((directory / "scan_manifest.json").read_text())
    with h5py.File(directory / "s11_traces.h5", "r") as handle:
        trace = handle["s_parameter"]
        frequency = handle["frequencies_Hz"][:]
        return dict(
            status=manifest["status"],
            planned=manifest["n_points_planned"],
            measured=manifest["n_points_measured"],
            trace_shape=trace.shape,
            s_parameter=trace.attrs["s_param"],
            frequency_range_Hz=(float(frequency[0]), float(frequency[-1])),
        )


def create_demo_config(directory):
    """Prepare an example calibration/config for a short instrument-free run.

    Existing config/calibration files are retained. Each scan gets its own
    timestamped folder; no previous scan is overwritten.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    examples = Path(__file__).resolve().parents[1] / "config"
    calibration_path = directory / "bead_pull_calibration.json"
    if not calibration_path.exists():
        calibration_path.write_text(
            (examples / "bead_pull_calibration.example.json").read_text(), encoding="utf-8"
        )
    config_path = directory / "measurement_config.json"
    if not config_path.exists():
        values = json.loads((examples / "measurement_config.example.json").read_text())
        values["calibration_file"] = str(calibration_path)
        values["output"]["root"] = str(directory / "runs")
        values["scan"].update(step_size_m=0.05, settle_s=0.0)
        values["vna"]["points"] = 11
        values["comment"] = "Example calibration and simulated acquisition; no measured data."
        config_path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return config_path


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    config = MeasurementConfig.from_file(args.config)
    if args.dry_run:
        targets = plan_scan(config)
        print(f"Planned scan: {len(targets)} points")
        print("thread  point  position [m]  target [steps]")
        for sub, index, _, position, steps in targets:
            print(f"{sub.index:6d} {index:6d} {position:13.6f} {steps:15d}")
    else:
        output = acquire(
            config,
            simulate=args.simulate,
            vna_testmode=args.vna_testmode,
            overwrite=args.overwrite,
            config_path=args.config,
        )
        print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
