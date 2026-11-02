"""L6470 serial motor, saved position/home, limits and simulated motor.

Ported from spider-bead-pull-ai. Positive logical steps follow thread unwinding.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence
from .calibration import _utcnow

MOTOR_STATE_ENV_VAR = "MADMAX_BEAD_PULL_STATE"
DEFAULT_MOTOR_STATE_PATH = Path.home() / ".madmax_bead_pull" / "motor_state.json"
DEFAULT_MEASUREMENT_CONFIG_PATH = Path("config/measurement_config.json")
_CLOCKWISE_ALIASES = {"clockwise", "cw"}
_COUNTERCLOCKWISE_ALIASES = {"counterclockwise", "anticlockwise", "anti-clockwise", "ccw"}


class EmergencyStop(RuntimeError):
    "Raised inside a move when an emergency stop halts the motor mid-motion."


class PositionLimitError(RuntimeError):
    "Raised when a move would take the bead outside the allowed step range."


def _last_int(reply: Sequence[str]) -> int | None:
    """Last integer token in a controller reply, or ``None`` if there is none."""
    last: int | None = None
    for line in reply:
        for token in str(line).replace(",", " ").split():
            try:
                last = int(token)
            except ValueError:
                continue
    return last


def _parse_int(reply: Sequence[str]) -> int:
    """Pull the last integer token out of a controller reply (e.g. ``getpos``)."""
    last = _last_int(reply)
    if last is None:
        raise ValueError(f"Could not parse an integer position from reply: {reply!r}")
    return last


def _strip_echo(lines: Sequence[str], command: str) -> list[str]:
    "Drop the controller's echo of ``command`` from the front of a reply."
    sent = command.strip().lower()
    out = list(lines)
    while out and out[0].strip().lower() == sent:
        out.pop(0)
    return out


def _opt_int(value: Any) -> int | None:
    """Coerce a value to ``int``, returning ``None`` for missing/unparsable input."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def motor_state_path(path: str | Path | None = None) -> Path:
    "Resolve the motor-state file location."
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(MOTOR_STATE_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_MOTOR_STATE_PATH


@dataclass
class MotorState:
    "The persisted motor position and home position (logical frame, steps)."

    path: Path
    position_steps: int | None = None
    home_steps: int | None = None
    updated_utc: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MotorState":
        """Load the state file, or return an empty state if it is absent/corrupt."""
        resolved = motor_state_path(path)
        state = cls(path=resolved)
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return state  # missing / unreadable / malformed -> establish it later
        if isinstance(data, dict):
            state.position_steps = _opt_int(data.get("position_steps"))
            state.home_steps = _opt_int(data.get("home_steps"))
            updated = data.get("updated_utc")
            state.updated_utc = str(updated) if updated is not None else None
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_steps": self.position_steps,
            "home_steps": self.home_steps,
            "updated_utc": self.updated_utc,
        }

    def _save(self) -> None:
        self.updated_utc = _utcnow()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on the same filesystem

    def set_position(self, steps: int) -> int:
        """Persist ``steps`` as the current motor position (no write if unchanged)."""
        steps = int(steps)
        if steps != self.position_steps:
            self.position_steps = steps
            self._save()
        return steps

    def set_home(self, steps: int) -> int:
        """Persist ``steps`` as the home position (no write if unchanged)."""
        steps = int(steps)
        if steps != self.home_steps:
            self.home_steps = steps
            self._save()
        return steps


def set_saved_position(steps: int, path: str | Path | None = None) -> int:
    "Write ``steps`` into the shared motor-state file with no controller attached."
    return MotorState.load(path).set_position(int(steps))


def unwind_direction_sign(unwind_direction: str | int) -> int:
    "Map an ``unwind_direction`` setting to the raw-rotation sign (+1/-1) that unwinds the thread."
    if isinstance(unwind_direction, (int, float)) and not isinstance(unwind_direction, bool):
        sign = int(unwind_direction)
        if sign in (1, -1):
            return sign
        raise ValueError(f"unwind_direction sign must be +1 or -1, got {unwind_direction!r}")
    key = str(unwind_direction).strip().lower()
    if key in _CLOCKWISE_ALIASES:
        return 1
    if key in _COUNTERCLOCKWISE_ALIASES:
        return -1
    raise ValueError(
        f"unwind_direction must be 'clockwise' or 'counterclockwise' "
        f"(or +1/-1), got {unwind_direction!r}"
    )


class PositionLimitMixin:
    "Soft end-stops: an allowed ``[min, max]`` step range the bead may occupy."

    min_position_steps: int | None = None
    max_position_steps: int | None = None
    _limits_bypassed: bool = False

    def _init_limits(self, min_position_steps: int | None, max_position_steps: int | None) -> None:
        """Set the initial range (called from ``__init__``; validates ordering)."""
        self.min_position_steps, self.max_position_steps = _validate_limits(
            min_position_steps, max_position_steps
        )
        self._limits_bypassed = False

    # -- read ---------------------------------------------------------------
    def get_limits(self) -> tuple[int | None, int | None]:
        """The allowed ``(min, max)`` step range; ``None`` means unbounded."""
        return self.min_position_steps, self.max_position_steps

    @property
    def limits_bypassed(self) -> bool:
        """Whether the range check is currently suspended."""
        return self._limits_bypassed

    def limits_description(self) -> str:
        """One-line human-readable summary of the range and bypass state."""
        lo, hi = self.get_limits()
        if lo is None and hi is None:
            span = "no position limits set (unbounded)"
        else:
            span = (
                f"allowed range: [{'-inf' if lo is None else lo}, "
                f"{'+inf' if hi is None else hi}] steps"
            )
        return span + (" -- BYPASSED" if self._limits_bypassed else "")

    # -- write --------------------------------------------------------------
    def set_limits(
        self, min_position_steps: int | None, max_position_steps: int | None
    ) -> tuple[int | None, int | None]:
        "Set the allowed step range (in memory). Returns the new ``(min, max)``."
        self.min_position_steps, self.max_position_steps = _validate_limits(
            min_position_steps, max_position_steps
        )
        try:
            current = self.get_position()
        except Exception:  # pragma: no cover - position may be unreadable
            current = None
        if current is not None and self._limit_excess(current) > 0:
            print(
                f"WARNING: the current position ({current} steps) lies outside the "
                f"new range [{self.min_position_steps}, {self.max_position_steps}]. "
                "Only moves back toward the range will be allowed."
            )
        print(f"Position limits set -- {self.limits_description()}")
        return self.get_limits()

    def save_limits(self, config_path: str | Path = DEFAULT_MEASUREMENT_CONFIG_PATH) -> Path:
        """Write the current range into the measurement config so it outlives the
        session.  Thin wrapper around :func:`save_position_limits`."""
        return save_position_limits(self.min_position_steps, self.max_position_steps, config_path)

    def set_min_position(self, steps: int | None = None) -> int | None:
        """Set the lower bound; default: the current position.  Returns it."""
        value = self.get_position() if steps is None else int(steps)
        self.set_limits(value, self.max_position_steps)
        return self.min_position_steps

    def set_max_position(self, steps: int | None = None) -> int | None:
        """Set the upper bound; default: the current position.  Returns it."""
        value = self.get_position() if steps is None else int(steps)
        self.set_limits(self.min_position_steps, value)
        return self.max_position_steps

    # -- bypass (calibration mode) -----------------------------------------
    def enable_limit_bypass(self) -> None:
        "Suspend the range check so the bead can be driven past the limits."
        self._limits_bypassed = True
        print(
            "CALIBRATION MODE: position limits BYPASSED -- moves outside "
            f"[{self.min_position_steps}, {self.max_position_steps}] steps are now "
            "allowed. Jog carefully and watch the bead; call "
            "disable_limit_bypass() as soon as you are done."
        )

    def disable_limit_bypass(self) -> None:
        """Re-arm the range check (the normal state)."""
        self._limits_bypassed = False
        print(f"Position limits re-armed -- {self.limits_description()}")
        try:
            current = self.get_position()
        except Exception:  # pragma: no cover - position may be unreadable
            return
        if self._limit_excess(current) > 0:
            print(
                f"WARNING: the bead is at {current} steps, outside the allowed "
                "range. Only moves back toward the range will be allowed; set new "
                "limits with set_limits() if this position is meant to be valid."
            )

    @contextmanager
    def bypass_limits(self):
        """Context manager form of the bypass -- always re-arms on exit.

        ``with motor.bypass_limits(): motor.move_by(-5000)``
        """
        previous = self._limits_bypassed
        self.enable_limit_bypass()
        try:
            yield self
        finally:
            if previous:
                self._limits_bypassed = True
            else:
                self.disable_limit_bypass()

    # -- the check itself ---------------------------------------------------
    def _limit_excess(self, position: int) -> int:
        """How far ``position`` falls outside the allowed range (0 if inside)."""
        lo, hi = self.min_position_steps, self.max_position_steps
        if lo is not None and position < lo:
            return lo - position
        if hi is not None and position > hi:
            return position - hi
        return 0

    def _check_limits(self, current: int, target: int) -> None:
        """Raise :class:`PositionLimitError` unless the move is allowed."""
        if self._limits_bypassed:
            return
        if self.min_position_steps is None and self.max_position_steps is None:
            return
        excess = self._limit_excess(target)
        if excess == 0 or excess < self._limit_excess(current):
            return  # inside the range, or moving back toward it
        lo = "-inf" if self.min_position_steps is None else self.min_position_steps
        hi = "+inf" if self.max_position_steps is None else self.max_position_steps
        raise PositionLimitError(
            f"Refusing to move from {current} to {target} steps: that is {excess} "
            f"steps outside the allowed range [{lo}, {hi}]. Adjust the range with "
            "set_limits(), or -- if you are deliberately exploring beyond it to "
            "calibrate the limits -- use 'with motor.bypass_limits():' / "
            "enable_limit_bypass()."
        )


def _validate_limits(
    min_position_steps: Any, max_position_steps: Any
) -> tuple[int | None, int | None]:
    """Coerce a ``(min, max)`` pair to ints (or ``None``) and check the ordering."""
    lo = None if min_position_steps is None else int(min_position_steps)
    hi = None if max_position_steps is None else int(max_position_steps)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"min_position_steps ({lo}) must not exceed max_position_steps ({hi}).")
    return lo, hi


def save_position_limits(
    min_position_steps: int | None,
    max_position_steps: int | None,
    config_path: str | Path = DEFAULT_MEASUREMENT_CONFIG_PATH,
) -> Path:
    "Write the allowed step range into the measurement config's ``stepper`` block."
    lo, hi = _validate_limits(min_position_steps, max_position_steps)
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    stepper = data.setdefault("stepper", {})
    if not isinstance(stepper, dict):
        raise ValueError(f"{path}: the 'stepper' entry is not a JSON object.")
    stepper["min_position_steps"] = lo
    stepper["max_position_steps"] = hi
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem
    print(f"Saved position limits [{lo}, {hi}] steps to {path}")
    return path


class SimulatedMotor(PositionLimitMixin):
    "In-memory motor for dry runs and tests; tracks an integer step counter."

    def __init__(
        self,
        start_position: int = 0,
        home_position: int | None = 0,
        verbose: bool = False,
        min_position_steps: int | None = None,
        max_position_steps: int | None = None,
    ) -> None:
        self._position = int(start_position)
        self._home = int(home_position) if home_position is not None else None
        self.verbose = verbose
        self._init_limits(min_position_steps, max_position_steps)

    def home(self) -> None:
        if self._home is None:
            raise RuntimeError("No home position set; call set_home() first.")
        self.move_to_position(self._home)
        if self.verbose:
            print(f"[SimulatedMotor] home -> {self._position}")

    def get_position(self) -> int:
        return self._position

    def move_by(self, delta_steps: int) -> None:
        delta = int(delta_steps)
        self._check_limits(self._position, self._position + delta)
        self._position += delta
        if self.verbose:
            print(f"[SimulatedMotor] move_by {delta:+d} -> {self._position}")

    def move_to_position(self, steps: int) -> None:
        """Move to the absolute step count ``steps`` (relative move, as on hardware)."""
        self.move_by(int(steps) - self._position)

    def set_home(self, position: int | None = None) -> int:
        """Record a home position (default: the current position)."""
        self._home = int(position) if position is not None else self._position
        if self.verbose:
            print(f"[SimulatedMotor] set_home -> {self._home}")
        return self._home

    @property
    def home_position(self) -> int | None:
        return self._home

    def set_position(self, steps: int) -> int:
        """Declare the current position to be ``steps`` (no motion), as on real hardware."""
        self._position = int(steps)
        if self.verbose:
            print(f"[SimulatedMotor] set_position -> {self._position}")
        return self._position

    def emergency_stop(self) -> None:
        """Simulated emergency stop: no motion to halt, just report it.

        Present so notebooks/scripts can wire a STOP button to the same call in
        SIMULATE mode as on real hardware.
        """
        if self.verbose:
            print("[SimulatedMotor] EMERGENCY STOP (no-op; motor is simulated)")

    def shutdown(self) -> None:
        pass


class Stepper(PositionLimitMixin):
    "Stepper motor on the L6470 ASCII controller over serial."

    # Refuse any single move larger than this many steps unless overridden; a
    # cheap circuit-breaker against a runaway from a bad target (the physical
    # limit switch that would have stopped one is not present on this rig).
    DEFAULT_MAX_STEP_MAGNITUDE = 500_000

    # Controller power-on default (steps/s), per the instruction set; the real
    # value is recorded by setprofile().  Used only to bound how long a move may
    # take before the wait loop gives up -- see :meth:`_move_timeout_s`.
    DEFAULT_SPEED_STEPS_PER_S = 500
    # A move is abandoned after this multiple of its expected duration, plus a
    # fixed allowance for acceleration and command latency.
    WAIT_TIMEOUT_FACTOR = 4.0
    WAIT_TIMEOUT_FLOOR_S = 30.0

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        timeout: float = 1.0,
        motor: int = 1,
        reset_delay: float = 2.0,
        unwind_direction: str | int = "counterclockwise",
        state_path: str | Path | None = None,
        max_step_magnitude: int | None = DEFAULT_MAX_STEP_MAGNITUDE,
        min_position_steps: int | None = None,
        max_position_steps: int | None = None,
    ) -> None:
        import serial  # lazy: only needed for real hardware

        # +1/-1 mapping a logical "advance the bead / unwind" move onto the raw
        # rotation sense; the high-level interface (move_by/get_position) works in
        # the logical frame so the calibration step anchors are unaffected.
        # Everything that can be validated without the port is validated *first*,
        # so a bad setting raises before the serial device is claimed rather than
        # leaving it open and locked against the next attempt.
        self.direction_sign = unwind_direction_sign(unwind_direction)
        self.max_step_magnitude = (
            int(max_step_magnitude) if max_step_magnitude is not None else None
        )
        # Soft end-stops: the allowed absolute step range (see PositionLimitMixin).
        self._init_limits(min_position_steps, max_position_steps)
        # Persisted position + home, shared with every other program on this
        # computer.  Loaded now; the logical frame is anchored to it in
        # restore_position() once the port is up and any reset has happened.
        self.state = MotorState.load(state_path)

        self.ser = serial.Serial(port, baud, timeout=timeout)
        try:
            self.motor = motor
            self.speed_steps_per_s = self.DEFAULT_SPEED_STEPS_PER_S
            # Serialise access to the port so an emergency stop issued from another
            # thread (e.g. a notebook STOP button) never interleaves its bytes with
            # a command in flight.  Re-entrant so nested command helpers are fine.
            self._io_lock = threading.RLock()
            # Held for the whole of a move, so two threads (a background jog and a
            # foreground one, say) cannot have two rotations in flight at once.
            self._move_lock = threading.Lock()
            # Set by emergency_stop(); a move in progress checks it and aborts.
            self._estop = threading.Event()
            # Latched by emergency_stop(): a hard stop loses steps, so the absolute
            # frame is no longer trustworthy and further moves are refused until
            # set_position() re-establishes it.
            self._position_untrusted = False
            # Logical frame: position = direction_sign * raw_register + _offset_steps.
            # The offset lives here rather than in the controller because the step
            # register cannot be written to an arbitrary value (``setpos`` drives the
            # motor), and it is deliberately *not* persisted: it is only meaningful
            # together with one particular register baseline, and that baseline is
            # re-established on every connect by restore_position().
            self._offset_steps = 0
            time.sleep(reset_delay)  # USB-serial bridges often reset on open
            self.ser.reset_input_buffer()
            # Anchor the logical frame straight away, so a directly constructed
            # Stepper reports positions in the same frame as one built by open()
            # (which re-anchors after its reset zeroes the register).
            self.restore_position()
        except BaseException:
            self.ser.close()
            raise

    @classmethod
    def open(
        cls,
        port: str,
        baud: int = 115200,
        motor: int = 1,
        timeout: float = 1.0,
        microsteps: int = 8,
        acceleration: int = 1000,
        speed: int = 500,
        driving_voltage: float = 6.8,
        holding_voltage: float = 2.0,
        configure: bool = True,
        unwind_direction: str | int = "counterclockwise",
        state_path: str | Path | None = None,
        max_step_magnitude: int | None = DEFAULT_MAX_STEP_MAGNITUDE,
        min_position_steps: int | None = None,
        max_position_steps: int | None = None,
    ) -> "Stepper":
        dev = cls(
            port,
            baud=baud,
            timeout=timeout,
            motor=motor,
            unwind_direction=unwind_direction,
            state_path=state_path,
            max_step_magnitude=max_step_magnitude,
            min_position_steps=min_position_steps,
            max_position_steps=max_position_steps,
        )
        if configure:
            dev.reset()
            dev.setprofile(microsteps, acceleration, speed)
            dev.setvoltage(driving_voltage, holding_voltage)
        # After any reset (which zeroes the register), bind the logical frame back
        # to the last saved position.  Moves nothing.
        dev.restore_position()
        return dev

    # -- raw I/O / command set --------------------------------------------
    def cmd(self, cmd: str, expect_reply: bool = False) -> list[str]:
        "Send one ASCII command and return the controller's reply lines."
        with self._io_lock:
            self.ser.write((cmd + "\n").encode("ascii"))
            self.ser.flush()
            if not expect_reply:
                time.sleep(0.05)  # let any echo arrive, then drain
            lines: list[str] = []
            words = cmd.split()
            scalar_query = expect_reply and bool(words) and words[0].lower() in {"getpos", "busy"}
            original_timeout = self.ser.timeout
            deadline = time.monotonic() + (original_timeout or 1.0)
            pending = b""
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # An echo or a partial line must not start a fresh timeout.
                    self.ser.timeout = 0 if original_timeout == 0 else remaining
                    # readline() can restart its timeout for each byte; a slow
                    # trickle would then exceed the command's overall deadline.
                    raw = self.ser.read(1)
                    if not raw:
                        break
                    pending += raw
                    while b"\n" in pending:
                        raw_line, pending = pending.split(b"\n", 1)
                        line = raw_line.decode("ascii", errors="replace").strip()
                        if not line or not _strip_echo([line], cmd):
                            continue
                        lines.append(line)
                        if scalar_query:
                            try:
                                int(line)
                            except ValueError:
                                pass  # Unusual/multiline formats use the legacy drain.
                            else:
                                return lines
                # Never accept truncated digits as a scalar position/status.
                # Preserve raw-command replies that do not end in a newline.
                if pending and not scalar_query:
                    lines.extend(
                        _strip_echo([pending.decode("ascii", errors="replace").strip()], cmd)
                    )
                return lines
            finally:
                self.ser.timeout = original_timeout

    def reset(self):
        return self.cmd("reset")

    def gotoswitch(self):
        # WARNING: rotates until a limit switch triggers.  This rig has no switch,
        # so this never stops on its own -- do NOT use it for homing (that was the
        # original runaway bug).  home() uses a saved position instead.
        return self.cmd(f"gotoswitch {self.motor}")

    def wait(self):
        return self.cmd(f"wait {self.motor}")

    def busy(self):
        return self.cmd(f"busy {self.motor}", expect_reply=True)

    def rotate(self, steps):
        return self.cmd(f"rotate {self.motor} {int(steps)}")

    def getpos(self):
        return self.cmd(f"getpos {self.motor}", expect_reply=True)

    def setpos(self, step):
        # WARNING: despite the name this *drives to* an absolute position (see the
        # instruction set) -- it is a move, not a way to teach the controller where
        # it is.  Nothing in this class uses it; to re-establish the absolute frame
        # without moving the motor, use set_position().
        return self.cmd(f"setpos {self.motor} {int(step)}")

    def clearpos(self):
        return self.cmd(f"clearpos {self.motor}")

    def hiz(self):
        return self.cmd(f"hiz {self.motor}")

    def softstop(self):
        return self.cmd(f"softstop {self.motor}")

    def hardstop(self):
        return self.cmd(f"hardstop {self.motor}")

    def setprofile(self, microsteps=8, acceleration=1000, speed=500):
        # remember the speed: it is what makes an expected move duration -- and so
        # a usable wait timeout -- computable (see _move_timeout_s).
        self.speed_steps_per_s = max(int(speed), 1)
        return self.cmd(
            f"setprofile {self.motor} {int(microsteps)} {int(acceleration)} {int(speed)}"
        )

    def setvoltage(self, driving=6.8, holding=2.0):
        return self.cmd(f"setvoltage {self.motor} {driving} {holding}")

    # -- high-level interface ---------------------------------------------
    def restore_position(self) -> None:
        "Anchor the logical frame to the persisted position. Moves nothing."
        raw = self._read_raw_position()
        saved = self.state.position_steps
        if saved is None:
            self._offset_steps = 0
            if raw is not None:
                self._persist_logical(raw)
            return
        if raw is None:
            # Controller silent: assume the register holds the zero it resets to,
            # so the saved position still reads back unchanged.  A stale register
            # would only matter if the motor also answered later, in which case
            # get_position() reports it and the operator can re-reference.
            raw = 0
            print(
                "WARNING: controller returned no position on connect; assuming its "
                f"step register is 0 while anchoring to the saved position ({saved} "
                "steps). Check the serial connection and motor power."
            )
        self._offset_steps = int(saved) - self.direction_sign * raw

    def _read_raw_position(self) -> int | None:
        """Controller step register (raw frame), or ``None`` if it gave no reply.

        Reading the position must never crash a session: after power is cut or a
        cable is pulled, ``getpos`` returns an empty reply (the original crash).
        """
        try:
            return _parse_int(self.getpos())
        except ValueError:
            return None

    def _persist_logical(self, raw: int) -> int:
        """Convert a raw register reading to the logical frame and persist it."""
        return self.state.set_position(self.direction_sign * raw + self._offset_steps)

    def get_position(self) -> int:
        "Current motor position in the logical frame."
        raw = self._read_raw_position()
        if raw is not None:
            return self._persist_logical(raw)
        tracked = self.state.position_steps
        if tracked is None:
            raise RuntimeError(
                "Cannot determine the motor position: the controller returned no "
                "reply and no position has ever been saved. Check the serial "
                "connection and motor power, then set the position with "
                "set_position()."
            )
        print(
            "WARNING: controller returned no position; using the last saved value "
            f"({tracked} steps). Check the serial connection and motor power."
        )
        return tracked

    def _guard_move(self, delta: int) -> None:
        "Vet a move before a single byte reaches the controller."
        if self._position_untrusted:
            raise EmergencyStop(
                "Refusing to move: the motor was hard-stopped by an emergency stop, "
                "so the L6470 step counter has lost steps and the absolute position "
                "is no longer trustworthy. Bring the bead to a known reference and "
                "call set_position(<steps>) to re-establish the frame; that clears "
                "this latch."
            )
        cap = self.max_step_magnitude
        if cap is not None and abs(delta) > cap:
            raise ValueError(
                f"Refusing to move {delta:+d} steps: exceeds the safety limit of "
                f"{cap} steps (max_step_magnitude). If this move really is "
                "intended, raise max_step_magnitude when opening the motor."
            )
        if self._limits_bypassed:
            return
        if self.min_position_steps is None and self.max_position_steps is None:
            return
        current = self.get_position()
        self._check_limits(current, current + delta)

    def _is_busy(self, default_when_unknown: bool = False) -> bool:
        "Whether the controller reports the motor still rotating (``busy`` query)."
        try:
            reply = self.busy()
        except Exception:  # pragma: no cover - hardware quirk
            return default_when_unknown
        flag = _last_int(reply)
        if flag is not None:
            return flag != 0
        text = " ".join(reply).lower()
        if any(w in text for w in ("busy", "moving", "run")):
            return True
        if any(w in text for w in ("idle", "stop", "ready", "done")):
            return False
        return default_when_unknown

    def _move_timeout_s(self, delta_steps: int) -> float:
        "How long a ``delta_steps`` move may take before it is abandoned."
        speed = max(self.speed_steps_per_s, 1)
        return self.WAIT_TIMEOUT_FLOOR_S + self.WAIT_TIMEOUT_FACTOR * abs(delta_steps) / speed

    def _wait_until_idle(self, timeout_s: float | None = None) -> None:
        "Block until the motor has stopped, *interruptibly* and with a deadline."
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if self._estop.is_set():
                return
            self.wait()
            if self._estop.is_set():
                return
            # Silence or a generic acknowledgement does not confirm the motor
            # has stopped. Keep waiting, bounded by the move deadline below.
            if not self._is_busy(default_when_unknown=True):
                return
            if deadline is not None and time.monotonic() > deadline:
                self.emergency_stop()
                raise EmergencyStop(
                    f"Motor stop could not be confirmed within {timeout_s:.0f} s "
                    "of a move that should have finished long before; it has been "
                    "hard-stopped. Check the motor, its supply and the serial "
                    "link, then re-establish the position with set_position()."
                )
            time.sleep(0.05)

    def emergency_stop(self) -> None:
        "Halt the motor immediately and switch off its coils."
        self._estop.set()
        # Latch the position as untrusted: a hard stop loses steps, so every
        # later calibrated move is refused until set_position() re-anchors the
        # frame.  Setting it here (rather than only documenting it) is what stops
        # a scan resuming onto a step frame that no longer means anything.
        self._position_untrusted = True
        got = self._io_lock.acquire(timeout=0.5)
        try:
            self.ser.write(f"hardstop {self.motor}\n".encode("ascii"))
            self.ser.write(f"hiz {self.motor}\n".encode("ascii"))
            self.ser.flush()
        except Exception as exc:  # pragma: no cover - hardware quirk
            print(f"WARNING: emergency_stop could not write to the controller: {exc}")
        finally:
            if got:
                self._io_lock.release()
        print(
            "EMERGENCY STOP: hardstop + hiz sent -- motor halted and coils off. "
            "The absolute position is now unreliable (a hard stop loses steps); "
            "re-establish it with set_position() before any further move."
        )

    def move_by(self, delta_steps: int) -> None:
        delta = int(delta_steps)
        if delta == 0:
            return
        # One move at a time: a background jog and a foreground one would
        # otherwise both be waiting on -- and commanding -- the same motor, and
        # the position bookkeeping of whichever finished second would be wrong.
        if not self._move_lock.acquire(blocking=False):
            raise RuntimeError(
                "A move is already in progress on this motor (started from another "
                "thread, e.g. a background jog). Wait for it to finish, or stop it "
                "with emergency_stop(), before commanding another."
            )
        try:
            self._guard_move(delta)
            # Arm a fresh emergency-stop latch for this move.
            self._estop.clear()
            try:
                # translate a logical (unwind-positive) step delta into the raw
                # rotation sense the configured unwind direction demands.  The
                # rotate command is inside the guard too: an interrupt landing
                # between sending it and starting the wait would otherwise leave
                # the motor turning with nothing left to stop it.
                self.rotate(self.direction_sign * delta)
                self._wait_until_idle(self._move_timeout_s(delta))
            except KeyboardInterrupt:
                # Interrupt button pressed mid-move: stop the motor, then propagate.
                self.emergency_stop()
                raise EmergencyStop(
                    "Move aborted by keyboard interrupt; motor hard-stopped."
                ) from None
            if self._estop.is_set():
                # A STOP button (another thread) fired during the move.
                raise EmergencyStop("Move aborted by emergency stop; motor hard-stopped.")
            # Keep the persisted position current: prefer the controller's own
            # count, but if it is silent advance the saved position by the
            # commanded delta so we never lose track of where the bead is.
            raw = self._read_raw_position()
            if raw is not None:
                self._persist_logical(raw)
                return
            base = self.state.position_steps
            if base is None:
                print(
                    "WARNING: controller returned no position after the move and "
                    "none has ever been saved; the absolute position is unknown. "
                    "Bring the bead to a known reference and call set_position()."
                )
                return
            # Only the saved position is advanced; ``_offset_steps`` is left alone
            # deliberately, because it stays correct either way -- if the register
            # did count the move it now reads back base + delta through the same
            # offset, and if the motor never turned it still reads back base.
            self.state.set_position(base + delta)
            print(
                "WARNING: controller returned no position after the move; advanced "
                f"the saved position by {delta:+d} steps instead."
            )
        finally:
            self._move_lock.release()

    def move_to_position(self, steps: int) -> None:
        "Move the bead to the absolute step count ``steps`` (logical frame)."
        self.move_by(int(steps) - self.get_position())

    def home(self) -> None:
        "Drive the bead to the saved home position."
        target = self.state.home_steps
        if target is None:
            raise RuntimeError(
                "No home position has been set. Set one with set_home() (e.g. in "
                "the calibration notebook) before calling home()."
            )
        self.move_to_position(int(target))

    def set_home(self, position: int | None = None) -> int:
        """Record a home position (default: the current position) and persist it.

        Saved to the shared state file so every program on this computer homes to
        the same place.  Returns the home step count.
        """
        pos = int(position) if position is not None else self.get_position()
        return self.state.set_home(pos)

    @property
    def home_position(self) -> int | None:
        """The saved home position (logical steps), or ``None`` if unset."""
        return self.state.home_steps

    def set_position(self, steps: int) -> int:
        "Declare the bead's current position to be ``steps``. Moves nothing."
        steps = int(steps)
        raw = self._read_raw_position()
        if raw is None:
            raw = 0
            print(
                "WARNING: controller returned no position; assuming its step "
                f"register is 0 while declaring the position to be {steps} steps. "
                "Check the serial connection and motor power."
            )
        self._offset_steps = steps - self.direction_sign * raw
        was_untrusted = self._position_untrusted
        self._position_untrusted = False
        self._estop.clear()
        if was_untrusted:
            print(f"Position re-established at {steps} steps; moves are allowed again.")
        return self.state.set_position(steps)

    def shutdown(self) -> None:
        try:
            self.hiz()
        except Exception as exc:  # controller already gone: still close the port
            print(f"WARNING: could not switch off the coils on shutdown: {exc}")
        finally:
            if self.ser.is_open:
                self.ser.close()


def jog_background(motor, steps):
    """Jog in a background thread so the notebook STOP button stays responsive.

    Keep the returned thread and join it before starting another jog.
    """

    def move():
        try:
            motor.move_by(int(steps))
            print(f"Position: {motor.get_position()} steps")
        except Exception as exc:
            print(f"Jog halted: {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=move, daemon=True)
    thread.start()
    return thread


def show_emergency_stop(motor):
    """Display a notebook button that calls the motor's immediate hard stop."""
    import ipywidgets as widgets
    from IPython.display import display

    button = widgets.Button(
        description="EMERGENCY STOP",
        button_style="danger",
        layout=widgets.Layout(width="240px", height="55px"),
    )
    button.on_click(lambda _: motor.emergency_stop())
    display(button)
    return button


def list_serial_ports():
    """Return available serial device names and descriptions; opens no device."""
    from serial.tools import list_ports

    return [(port.device, port.description) for port in list_ports.comports()]
