"""Fit transverse-integrated bead fields to a longitudinal multilayer model.

Input ET has axes (z, frequency), in V*m at the stated input-power normalization.
All longitudinal coordinates are positive distances OUT from the mirror, in m.
The default fits one real amplitude per frequency, keeping the measured gaps
and bead z positions fixed. Gaps, bead z positions, or both can be adjusted,
always shared across frequencies. A configurable penalty favors the measured
geometry. Use --geometry all to run all four combinations.

The observable is |ET|, with the finite-sphere Mie correction of Egge et al.,
JCAP 04 (2024) 005, eq. (5.6): https://arxiv.org/abs/2311.13359 . The normalized
Mie factors follow chapter 14 and the Julia listing in Jacob Egge's thesis:
https://ediss.sub.uni-hamburg.de/handle/ediss/11205 . Six multipoles are used.
The model supplies longitudinal phase; measured absolute phase and line signs
are not fitted. The fit assumes a common transverse mode through the stack.

The coherent integral includes every layer, the conducting mirror half-space,
and the adiabatically terminated exterior, as in the existing Julia workflow.
It does not integrate an antenna standing wave over the 3.249 m antenna path.
No Monte Carlo or uncertainty estimation is performed.

Usage: python -m src.fit_1D data/outputs/RUN_NAME/integrated_fields.npz
Default outputs stay under data/outputs/; -o overrides the path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import spherical_jn, spherical_yn

if __package__:
    from .analysis_band import require_analysis_band
    from .data_paths import fit_output_path
else:
    from analysis_band import require_analysis_band
    from data_paths import fit_output_path


C_LIGHT = 299792458.0
EPSILON_0 = 8.85418782e-12


@dataclass(frozen=True)
class ModelConfig:
    """Surface-to-surface gaps ordered from mirror to antenna; SI units.

    Disk permittivity/loss, mirror conductivity and disk radius are inherited
    notebook assumptions, not newly measured values. Bead parameters and the
    normalization must match the upstream ET conversion.
    """

    gap_widths_m: tuple[float, ...] = (7.803e-3, 8.853e-3, 8.855e-3)
    disk_thickness_m: float = 1e-3
    disk_eps_r: float = 9.3
    disk_loss_tangent: float = 1e-5
    mirror_conductivity_S_m: float = 5e7
    bead_diameter_m: float = 2.93e-3
    bead_eps_r: float = 9.23
    disk_radius_m: float = 0.15
    input_power_W: float = 1.0

    def __post_init__(self):
        gaps = np.asarray(self.gap_widths_m, dtype=float)
        if gaps.ndim != 1 or not np.all(np.isfinite(gaps)) or np.any(gaps <= 0):
            raise ValueError("gap_widths_m must be a vector of positive finite gaps.")
        object.__setattr__(self, "gap_widths_m", tuple(gaps))
        for name in ("disk_thickness_m", "disk_eps_r", "bead_diameter_m",
                     "bead_eps_r", "disk_radius_m", "input_power_W"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.bead_eps_r <= 1:
            raise ValueError("The dielectric bead model requires bead_eps_r > 1.")
        if not np.isfinite(self.disk_loss_tangent) or self.disk_loss_tangent < 0:
            raise ValueError("disk_loss_tangent must be finite and nonnegative.")
        if np.isnan(self.mirror_conductivity_S_m) or self.mirror_conductivity_S_m <= 0:
            raise ValueError("mirror_conductivity_S_m must be positive (inf gives a perfect mirror).")


def _axes(frequencies_Hz, z_from_mirror_m):
    frequency = np.atleast_1d(np.asarray(frequencies_Hz, dtype=float))
    z = np.atleast_1d(np.asarray(z_from_mirror_m, dtype=float))
    if (frequency.ndim != 1 or frequency.size == 0
            or not np.all(np.isfinite(frequency)) or np.any(frequency <= 0)):
        raise ValueError("frequencies_Hz must be a nonempty positive finite vector.")
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("z_from_mirror_m must be a nonempty finite vector, with z >= 0.")
    return frequency, z


def bead_factors(frequencies_Hz, diameter_m=2.93e-3, eps_r=9.23, order=6):
    """Return normalized complex (delta_e, delta_c); both tend to 1 for kr -> 0."""
    frequency, _ = _axes(frequencies_Hz, [0])
    if not np.isfinite(diameter_m) or diameter_m <= 0 or not np.isfinite(eps_r) or eps_r <= 1:
        raise ValueError("The bead diameter must be positive and its permittivity > 1.")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("The multipole order must be a positive integer.")
    x = np.pi * frequency * diameter_m / C_LIGHT
    m = np.sqrt(eps_r)

    def j(n, value):
        return np.cos(value) / value if n == -1 else spherical_jn(n, value)

    def h(n, value):
        return j(n, value) + 1j * spherical_yn(n, value)

    delta_e = np.zeros(frequency.size, dtype=complex)
    delta_c = np.zeros_like(delta_e)
    # Includes the static-polarizability normalization from the thesis listing.
    prefactor = (m*m + 2) / (2*x*(m*m - 1))
    for n in range(1, order + 1):
        kn = (2*n + 1) * prefactor * (j(n-1, x)*j(n, m*x) - m*j(n, x)*j(n-1, m*x))
        ln = prefactor * (
            (n+1)*(j(n-2, x)*j(n-1, m*x) - m*j(n-1, x)*j(n-2, m*x))
            + n*(j(n, x)*j(n+1, m*x) - m*j(n+1, x)*j(n, m*x)))
        dj = -j(n-1, x) + j(n+1, x)
        dj_inside = -j(n-1, m*x) + j(n+1, m*x)
        dh = h(n-1, x) - h(n+1, x)
        numerator = dj*h(n, x) + j(n, x)*dh
        cn = numerator / (m*dj_inside*h(n, x) + j(n, m*x)*dh)
        dn = m*x*numerator / (
            m*x*dj_inside*h(n, x)
            + j(n, m*x)*(m*m*x*dh + (m*m-1)*h(n, x)))
        delta_e += (-1)**n * (kn*cn - ln*dn)
        delta_c += kn*cn + ln*dn
    return delta_e, delta_c


@dataclass
class ModelFields:
    field: np.ndarray
    bead_magnitude: np.ndarray
    integral: np.ndarray


def evaluate_model(frequencies_Hz, z_from_mirror_m, config=ModelConfig()) -> ModelFields:
    """Unit-amplitude model; field axes (z, frequency), integral axes (frequency,).

    At the mirror the incident wave is 1 and the reflected wave is rho. Waves
    propagate as R*exp(-i*k*n*z) + L*exp(+i*k*n*z), with passive n having Im(n)>0.
    Bead magnitudes inside disks are NaN: the vacuum-sphere correction does not
    apply there. The unperturbed field remains available throughout the stack.
    """
    frequency, z = _axes(frequencies_Hz, z_from_mirror_m)
    k = 2*np.pi*frequency / C_LIGHT
    if np.isinf(config.mirror_conductivity_S_m):
        rho = -np.ones(frequency.size, dtype=complex)
        integral = np.zeros(frequency.size, dtype=complex)
    else:
        nm = np.sqrt(1 + 1j*config.mirror_conductivity_S_m / (2*np.pi*frequency*EPSILON_0))
        rho = (1 - nm) / (1 + nm)
        integral = 1j*(1 + rho) / (k*nm)
    right, left = np.ones(frequency.size, dtype=complex), rho.copy()
    delta_e, delta_c = bead_factors(frequency, config.bead_diameter_m, config.bead_eps_r)
    nd = np.sqrt(config.disk_eps_r * (1 + 1j*config.disk_loss_tangent))
    widths = np.empty(2*len(config.gap_widths_m))
    widths[::2], widths[1::2] = config.gap_widths_m, config.disk_thickness_m
    indices = np.ones(widths.size + 1, dtype=complex)
    indices[1::2] = nd
    field = np.empty((z.size, frequency.size), dtype=complex)
    bead_magnitude = np.full(field.shape, np.nan)
    edge = 0.0
    for layer, n in enumerate(indices):
        width = widths[layer] if layer < widths.size else np.inf
        mask = (z >= edge) & (z < edge + width)
        phase = np.exp(-1j*(z[mask] - edge)[:, None]*k*n)
        forward, backward = right*phase, left/phase
        field[mask] = forward + backward
        if layer % 2 == 0:
            convolved_squared = delta_e*(forward**2 + backward**2) + 2*delta_c*forward*backward
            bead_magnitude[mask] = np.sqrt(np.abs(convolved_squared))
        if layer == widths.size:
            # Abel limit of the exterior integral with exp(-eta*z), eta -> 0+.
            integral += (right - left) / (1j*k*n)
            break
        p = np.exp(-1j*k*n*width)
        integral += (right*(1 - p) + left*(1/p - 1)) / (1j*k*n)
        right, left = right*p, left/p  # reciprocal, not conjugate, for lossy n
        next_n = indices[layer + 1]
        right, left = (((next_n+n)*right + (next_n-n)*left) / (2*next_n),
                       ((next_n-n)*right + (next_n+n)*left) / (2*next_n))
        edge += width
    return ModelFields(field, bead_magnitude, integral)


def _check_bead_positions(z, config):
    radius = config.bead_diameter_m / 2
    if np.any(z < radius):
        raise ValueError("A measurement bead overlaps the mirror surface.")
    disk_start = 0.0
    for gap in config.gap_widths_m:
        disk_start += gap
        disk_end = disk_start + config.disk_thickness_m
        if np.any((z > disk_start - radius) & (z < disk_end + radius)):
            raise ValueError("A measurement bead overlaps a disk; check z positions and geometry.")
        disk_start = disk_end


@dataclass
class FitResult:
    frequencies_Hz: np.ndarray
    z_from_mirror_m: np.ndarray
    z_nominal_m: np.ndarray
    gap_widths_nominal_m: np.ndarray
    ET_measured: np.ndarray
    amplitude: np.ndarray
    ET_fitted_magnitude: np.ndarray
    ET_unperturbed: np.ndarray
    integral_ET_dz: np.ndarray
    beta2: np.ndarray
    residual_rms: np.ndarray
    config: ModelConfig
    fit_method: str
    geometry_penalty: float
    gap_bound_m: float
    position_bound_m: float

    def evaluate(self, z_from_mirror_m):
        """Reconstructed complex ET(z,f), in the model's phase convention."""
        return evaluate_model(self.frequencies_Hz, z_from_mirror_m, self.config).field * self.amplitude

    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        if path.suffix != ".npz":
            path = Path(f"{path}.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {field.name: getattr(self, field.name) for field in fields(self) if field.name != "config"}
        np.savez_compressed(path, **arrays, **asdict(self.config))
        return path


def fit_1d(frequencies_Hz, z_from_mirror_m, ET, *, config=ModelConfig(),
           fit_gaps=False, fit_positions=False, gap_bound_m=0.5e-3,
           position_bound_m=0.5e-3, geometry_penalty=1e-3) -> FitResult:
    """Fit |ET|, profiling out one nonnegative amplitude per frequency.

    ET must be reconstructed from raw data restricted to 20–22 GHz. A saved
    full-sweep reconstruction must be regenerated before fitting.

    Adjustable geometry is shared across the spectrum: one value per gap and
    one z coordinate per measurement plane. Each may move by at most its bound
    from the measured value. The objective is

        mean_f(sum_z(residual**2) / sum_z(abs(ET)**2))
        + geometry_penalty * sum((geometry_shift / geometry_bound)**2).

    Zero-signal bins do not affect geometry. The penalty is a preference for
    the measured geometry, not an experimental uncertainty or confidence level.
    Set it to zero for unregularized fits. These are local bounded optimizations;
    geometry parameters can be correlated even when the fit converges.
    """
    frequency, z = _axes(frequencies_Hz, z_from_mirror_m)
    require_analysis_band(frequency, "1D reconstructed field input")
    z_nominal = z.copy()
    gaps_nominal = np.asarray(config.gap_widths_m)
    measured = np.asarray(ET, dtype=complex)
    if measured.shape != (z.size, frequency.size) or not np.all(np.isfinite(measured)):
        raise ValueError("ET must be finite with shape (z, frequency).")
    _check_bead_positions(z, config)
    magnitude = np.abs(measured)
    if not np.isfinite(geometry_penalty) or geometry_penalty < 0:
        raise ValueError("geometry_penalty must be finite and nonnegative.")

    def project(current_config, current_z):
        model = evaluate_model(frequency, current_z, current_config)
        denominator = np.sum(model.bead_magnitude**2, axis=0)
        if not np.all(np.isfinite(denominator)) or np.any(denominator <= np.finfo(float).tiny):
            raise ValueError("The model has no usable bead response at the measurement positions.")
        amplitude = np.sum(model.bead_magnitude*magnitude, axis=0) / denominator
        return model, amplitude

    if fit_gaps or fit_positions:
        if fit_gaps and (not np.isfinite(gap_bound_m) or gap_bound_m <= 0 or not config.gap_widths_m):
            raise ValueError("Fitting gaps requires disks and a positive finite gap_bound_m.")
        if fit_positions and (not np.isfinite(position_bound_m) or position_bound_m <= 0):
            raise ValueError("Fitting z positions requires a positive finite position_bound_m.")
        signal = np.linalg.norm(magnitude, axis=0)
        active = signal > np.finfo(float).tiny
        n_gap = len(gaps_nominal) if fit_gaps else 0
        n_position = z.size if fit_positions else 0
        if (z.size - 1)*np.count_nonzero(active) <= n_gap + n_position:
            raise ValueError("Too few independent field magnitudes to fit the adjustable geometry.")
        # Keep every measured sphere in the same vacuum region over ALL allowed
        # gap/position combinations. Smaller bounds can be used near surfaces.
        gap_motion = gap_bound_m if fit_gaps else 0.0
        z_motion = position_bound_m if fit_positions else 0.0
        if np.any(gaps_nominal - gap_motion <= 0):
            raise ValueError("gap_bound_m permits nonpositive gaps.")
        edge = 0.0
        radius = config.bead_diameter_m / 2
        if np.any(z - z_motion < radius):
            raise ValueError("Position bounds allow a bead to overlap the mirror.")
        for index, gap in enumerate(gaps_nominal):
            edge += gap
            motion = (index + 1)*gap_motion + z_motion
            if np.any((z > edge - motion - radius)
                      & (z < edge + config.disk_thickness_m + motion + radius)):
                raise ValueError("Geometry bounds allow a disk to overlap a bead; reduce the bounds.")
            edge += config.disk_thickness_m

        initial = np.r_[gaps_nominal if fit_gaps else [], z if fit_positions else []]*1e3
        bounds = np.r_[np.full(n_gap, gap_bound_m), np.full(n_position, position_bound_m)]*1e3

        def unpack(parameters_mm):
            current = replace(config, gap_widths_m=tuple(parameters_mm[:n_gap]*1e-3)) if fit_gaps else config
            current_z = parameters_mm[n_gap:]*1e-3 if fit_positions else z_nominal
            return current, current_z

        def residual(parameters_mm):
            current, current_z = unpack(parameters_mm)
            model, amplitude = project(current, current_z)
            data_residual = ((model.bead_magnitude[:, active]*amplitude[active] - magnitude[:, active])
                             / (signal[active]*np.sqrt(np.count_nonzero(active)))).ravel()
            preference = np.sqrt(geometry_penalty)*(parameters_mm - initial)/bounds
            return np.r_[data_residual, preference]

        fitted = least_squares(residual, initial, bounds=(initial-bounds, initial+bounds),
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
        if not fitted.success:
            raise RuntimeError(f"Geometry fit failed: {fitted.message}")
        config, z = unpack(fitted.x)
        _check_bead_positions(z, config)

    model, amplitude = project(config, z)
    prediction = model.bead_magnitude*amplitude
    integral = model.integral*amplitude
    area = np.pi*config.disk_radius_m**2
    beta2 = EPSILON_0*(2*np.pi*frequency)**2*np.abs(integral)**2 / (8*C_LIGHT*area*config.input_power_W)
    method = {(False, False): "fixed", (True, False): "gaps",
              (False, True): "positions", (True, True): "both"}[bool(fit_gaps), bool(fit_positions)]
    return FitResult(frequency, z, z_nominal, gaps_nominal, measured, amplitude, prediction, model.field*amplitude,
                     integral, beta2, np.sqrt(np.mean((prediction-magnitude)**2, axis=0)),
                     config, method, geometry_penalty, gap_bound_m, position_bound_m)


def fit_all_geometries(frequencies_Hz, z_from_mirror_m, ET, *, config=ModelConfig(),
                       gap_bound_m=0.5e-3, position_bound_m=0.5e-3, geometry_penalty=1e-3):
    """Return independent fits for fixed, gaps, positions, and both geometries."""
    return {
        mode: fit_1d(frequencies_Hz, z_from_mirror_m, ET, config=config,
                     fit_gaps=fit_gaps, fit_positions=fit_positions, gap_bound_m=gap_bound_m,
                     position_bound_m=position_bound_m, geometry_penalty=geometry_penalty)
        for mode, fit_gaps, fit_positions in (
            ("fixed", False, False), ("gaps", True, False),
            ("positions", False, True), ("both", True, True))
    }


def fit_file(input_path: str | Path, output_path: str | Path | None = None, *,
             config: ModelConfig | None = None, fit_gaps=False, fit_positions=False,
             gap_bound_m=0.5e-3, position_bound_m=0.5e-3, geometry_penalty=1e-3) -> Path:
    """Read any NPZ with ET, frequencies_Hz, z_from_mirror_m; honor saved physics.

    Saved ModelConfig keys override defaults. An explicit config overrides all
    saved parameters. No pickle loading or legacy filename/layout is required.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else fit_output_path(input_path, "1D")
    if output_path.suffix != ".npz":
        output_path = Path(f"{output_path}.npz")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Choose an output path different from the input data.")
    with np.load(input_path, allow_pickle=False) as data:
        if config is None:
            saved = {}
            for field in fields(ModelConfig):
                if field.name in data:
                    value = data[field.name]
                    saved[field.name] = tuple(value) if field.name == "gap_widths_m" else value.item()
            config = ModelConfig(**saved)
        result = fit_1d(data["frequencies_Hz"], data["z_from_mirror_m"], data["ET"],
                        config=config, fit_gaps=fit_gaps, fit_positions=fit_positions,
                        gap_bound_m=gap_bound_m, position_bound_m=position_bound_m,
                        geometry_penalty=geometry_penalty)
    return result.save(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="NPZ containing ET, frequencies_Hz, z_from_mirror_m")
    parser.add_argument("-o", "--output", type=Path,
                        help="output NPZ (default: INPUT_STEM_fit_1D.npz under data/outputs/)")
    parser.add_argument("--geometry", choices=("fixed", "gaps", "positions", "both", "all"), default="fixed",
                        help="which geometry to adjust (default: fixed); all writes four suffixed NPZ files")
    parser.add_argument("--gap-bound-mm", type=float, default=0.5, help="maximum shift per gap (default: 0.5 mm)")
    parser.add_argument("--position-bound-mm", type=float, default=0.5, help="maximum shift per bead z (default: 0.5 mm)")
    parser.add_argument("--geometry-penalty", type=float, default=1e-3,
                        help="preference for measured geometry; 0 disables it (default: 0.001)")
    args = parser.parse_args(argv)
    modes = ("fixed", "gaps", "positions", "both") if args.geometry == "all" else (args.geometry,)
    output = args.output or fit_output_path(args.input, "1D")
    if output.suffix != ".npz":
        output = Path(f"{output}.npz")
    for mode in modes:
        target = output.with_name(output.stem + f"_{mode}.npz") if args.geometry == "all" else output
        path = fit_file(args.input, target, fit_gaps=mode in ("gaps", "both"),
                        fit_positions=mode in ("positions", "both"), gap_bound_m=args.gap_bound_mm*1e-3,
                        position_bound_m=args.position_bound_mm*1e-3, geometry_penalty=args.geometry_penalty)
        print(f"Longitudinal fit ({mode}) saved to: {path}")


if __name__ == "__main__":
    main()
