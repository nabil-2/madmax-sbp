"""Use the existing 5_ notebook/MADBead analysis after SBP transverse integration.

The Julia backend supplies outlier rejection, Monte Carlo amplitude fits,
longitudinal integration and boost statistics. Only the shared geometry driver
is an SBP extension. This module handles numeric files and notebook plotting;
it contains no replacement least-squares or field-to-boost calculation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
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
    mirror_z_m: float = 3.249

    def __post_init__(self):
        gaps = np.asarray(self.gap_widths_m, dtype=float)
        if gaps.ndim != 1 or not np.all(np.isfinite(gaps)) or np.any(gaps <= 0):
            raise ValueError("gap_widths_m must be a vector of positive finite gaps.")
        object.__setattr__(self, "gap_widths_m", tuple(gaps))
        for name in (
            "disk_thickness_m",
            "disk_eps_r",
            "bead_diameter_m",
            "bead_eps_r",
            "disk_radius_m",
            "input_power_W",
            "mirror_z_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.bead_eps_r <= 1:
            raise ValueError("The dielectric bead model requires bead_eps_r > 1.")
        if not np.isfinite(self.disk_loss_tangent) or self.disk_loss_tangent < 0:
            raise ValueError("disk_loss_tangent must be finite and nonnegative.")
        if np.isnan(self.mirror_conductivity_S_m) or self.mirror_conductivity_S_m <= 0:
            raise ValueError(
                "mirror_conductivity_S_m must be positive (inf gives a perfect mirror)."
            )


def _axes(frequencies_Hz, z_from_mirror_m):
    frequency = np.atleast_1d(np.asarray(frequencies_Hz, dtype=float))
    z = np.atleast_1d(np.asarray(z_from_mirror_m, dtype=float))
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or not np.all(np.isfinite(frequency))
        or np.any(frequency <= 0)
    ):
        raise ValueError("frequencies_Hz must be a nonempty positive finite vector.")
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError(
            "z_from_mirror_m must be a nonempty finite vector, with z >= 0."
        )
    return frequency, z


def bead_factors(frequencies_Hz, diameter_m=2.93e-3, eps_r=9.23, order=3):
    """Return normalized complex (delta_e, delta_c); both tend to 1 for kr -> 0."""
    frequency, _ = _axes(frequencies_Hz, [0])
    if (
        not np.isfinite(diameter_m)
        or diameter_m <= 0
        or not np.isfinite(eps_r)
        or eps_r <= 1
    ):
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
    prefactor = (m * m + 2) / (2 * x * (m * m - 1))
    for n in range(1, order + 1):
        kn = (
            (2 * n + 1)
            * prefactor
            * (j(n - 1, x) * j(n, m * x) - m * j(n, x) * j(n - 1, m * x))
        )
        ln = prefactor * (
            (n + 1)
            * (j(n - 2, x) * j(n - 1, m * x) - m * j(n - 1, x) * j(n - 2, m * x))
            + n * (j(n, x) * j(n + 1, m * x) - m * j(n + 1, x) * j(n, m * x))
        )
        dj = -j(n - 1, x) + j(n + 1, x)
        dj_inside = -j(n - 1, m * x) + j(n + 1, m * x)
        dh = h(n - 1, x) - h(n + 1, x)
        numerator = dj * h(n, x) + j(n, x) * dh
        cn = numerator / (m * dj_inside * h(n, x) + j(n, m * x) * dh)
        dn = (
            m
            * x
            * numerator
            / (
                m * x * dj_inside * h(n, x)
                + j(n, m * x) * (m * m * x * dh + (m * m - 1) * h(n, x))
            )
        )
        delta_e += (-1) ** n * (kn * cn - ln * dn)
        delta_c += kn * cn + ln * dn
    return delta_e, delta_c


@dataclass
class ModelFields:
    field: np.ndarray
    bead_magnitude: np.ndarray
    integral: np.ndarray


def model_parameters(config, amplitude=1.0):
    """Translate SBP geometry to the existing MADBead parameter names."""
    parameters = {
        "E_0": amplitude,
        "z_m": config.mirror_z_m,
        "σ_m": config.mirror_conductivity_S_m,
        "n_disk": len(config.gap_widths_m),
        "r_b": config.bead_diameter_m / 2,
        "ϵ_b": config.bead_eps_r,
    }
    for i, gap in enumerate(config.gap_widths_m, 1):
        parameters.update(
            {
                f"d_v_{i}": gap,
                f"d_d_{i}": config.disk_thickness_m,
                f"ϵ_d_{i}": config.disk_eps_r,
                f"tanD_d_{i}": config.disk_loss_tangent,
            }
        )
    return parameters


def evaluate_model(frequencies_Hz, z_from_mirror_m, config=ModelConfig()):
    """Plot the original MADBead mean-parameter model; no fitting or boost calculation."""
    if __package__:
        from .full_bead_model import evaluate_full_model
    else:
        from full_bead_model import evaluate_full_model
    frequency, z = _axes(frequencies_Hz, z_from_mirror_m)
    results = [evaluate_full_model(f, z, model_parameters(config)) for f in frequency]
    return ModelFields(
        np.column_stack([r.field[:, 0] for r in results]),
        np.column_stack([r.bead_magnitude[:, 0] for r in results]),
        np.array([1j * r.integral[0] for r in results]),
    )


def evaluate_saved_model(saved, frequencies_Hz, z_from_mirror_m):
    """Evaluate the saved MC mean parameters, including their field amplitude."""
    if __package__:
        from .full_bead_model import evaluate_full_model
    else:
        from full_bead_model import evaluate_full_model
    frequency, z = _axes(frequencies_Hz, z_from_mirror_m)
    indices = np.array([np.argmin(abs(saved["frequencies_Hz"] - f)) for f in frequency])
    if not np.allclose(
        np.asarray(saved["frequencies_Hz"])[indices], frequency, rtol=0, atol=1
    ):
        raise ValueError(
            "Requested frequency was not retained by the 1D outlier filter."
        )
    names = [key for key in saved if key.startswith("parameter__")]
    if names:
        results = [
            evaluate_full_model(
                f,
                z,
                {
                    key.removeprefix("parameter__")
                    .replace("eps", "ϵ")
                    .replace("sigma", "σ"): float(saved[key][i])
                    for key in names
                },
            )
            for f, i in zip(frequency, indices)
        ]
        return ModelFields(
            np.column_stack([r.field[:, 0] for r in results]),
            np.column_stack([r.bead_magnitude[:, 0] for r in results]),
            np.array([1j * r.integral[0] for r in results]),
        )
    # Read-only support for earlier saved figures; new fits always store MC parameters.
    values = {
        item.name: (
            tuple(saved[item.name])
            if item.name == "gap_widths_m"
            else np.asarray(saved[item.name]).item()
        )
        for item in fields(ModelConfig)
        if item.name in saved
    }
    result = evaluate_model(frequency, z, ModelConfig(**values))
    amplitude = np.asarray(saved["amplitude"])[indices]
    return ModelFields(
        result.field * amplitude,
        result.bead_magnitude * abs(amplitude),
        result.integral * amplitude,
    )


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
    beta2_error: np.ndarray
    residual_rms: np.ndarray
    config: ModelConfig
    fit_method: str
    geometry_penalty: float
    gap_bound_m: float
    position_bound_m: float
    extras: dict = field(default_factory=dict)

    def arrays(self):
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"config", "extras"}
        }
        result.update(asdict(self.config))
        result.update(self.extras)
        return result

    def evaluate(self, z_from_mirror_m):
        return evaluate_saved_model(
            self.arrays(), self.frequencies_Hz, z_from_mirror_m
        ).field

    def save(self, output_path):
        output_path = Path(output_path)
        if output_path.suffix != ".npz":
            output_path = Path(str(output_path) + ".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.arrays())
        return output_path


def julia_executable():
    requested = os.environ.get("MADMAX_SBP_JULIA")
    executable = requested or shutil.which("julia")
    if not executable and Path("/software/julia/default/bin/julia").is_file():
        executable = "/software/julia/default/bin/julia"
    if not executable:
        raise FileNotFoundError(
            "Install Julia or set MADMAX_SBP_JULIA; then instantiate julia/analysis_1d."
        )
    return executable


def fit_1d(
    frequencies_Hz,
    z_from_mirror_m,
    ET,
    *,
    config=ModelConfig(),
    fit_gaps=False,
    fit_positions=False,
    gap_bound_m=0.5e-3,
    position_bound_m=0.5e-3,
    geometry_penalty=1e-3,
    n_mc=100,
    relative_error=0.05,
    outlier_window=51,
    outlier_threshold=12.0,
    seed=20261025,
):
    """Use the existing ET-to-boost pipeline with the SBP geometry amendments.

    Inputs must already have been reconstructed using only raw 20–22 GHz data.
    Frequency rejection and 100-draw MC defaults come from the 5_ notebook.
    Shared geometry is fitted first; the existing MC routine then fits E_0.
    """
    frequency, z = _axes(frequencies_Hz, z_from_mirror_m)
    require_analysis_band(frequency, "1D reconstructed field input")
    measured = np.asarray(ET, dtype=complex)
    if measured.shape != (len(z), len(frequency)):
        raise ValueError("Expected ET with axes (thread,frequency).")
    if fit_gaps and (not np.isfinite(gap_bound_m) or gap_bound_m <= 0):
        raise ValueError("gap_bound_m must be positive.")
    if fit_positions and (not np.isfinite(position_bound_m) or position_bound_m <= 0):
        raise ValueError("position_bound_m must be positive.")
    if not np.isfinite(geometry_penalty) or geometry_penalty < 0:
        raise ValueError("geometry_penalty must be nonnegative.")
    payload = {
        **asdict(config),
        "frequencies_Hz": frequency,
        "z_from_mirror_m": z,
        "ET": measured,
        "fit_gaps": int(fit_gaps),
        "fit_positions": int(fit_positions),
        "gap_bound_m": gap_bound_m,
        "position_bound_m": position_bound_m,
        "geometry_penalty": geometry_penalty,
        "n_mc": int(n_mc),
        "relative_error": relative_error,
        "outlier_window": int(outlier_window),
        "outlier_threshold": outlier_threshold,
        "seed": int(seed),
    }
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.setdefault("JULIA_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="sbp-1d-") as directory:
        source = Path(directory) / "input.npz"
        output = Path(directory) / "result.npz"
        np.savez(source, **payload)
        subprocess.run(
            [
                julia_executable(),
                "--startup-file=no",
                "--project=" + str(root / "julia/analysis_1d"),
                str(root / "src/run_1d.jl"),
                str(source),
                str(output),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        with np.load(output, allow_pickle=False) as archive:
            result = {key: np.array(archive[key]) for key in archive.files}
    fitted_config = replace(config, gap_widths_m=tuple(result.pop("gap_widths_m")))
    names = {item.name for item in fields(FitResult)} - {
        "config",
        "fit_method",
        "geometry_penalty",
        "gap_bound_m",
        "position_bound_m",
        "extras",
    }
    values = {name: result.pop(name) for name in names}
    result.update(
        analysis_backend="existing_5_notebook_MADBead",
        seed=int(seed),
        integral_statistic="mean_abs_MC",
        outlier_window=int(outlier_window),
        outlier_threshold=float(outlier_threshold),
    )
    mode = {
        (False, False): "fixed",
        (True, False): "gaps",
        (False, True): "positions",
        (True, True): "both",
    }[bool(fit_gaps), bool(fit_positions)]
    return FitResult(
        **values,
        config=fitted_config,
        fit_method=mode,
        geometry_penalty=geometry_penalty,
        gap_bound_m=gap_bound_m,
        position_bound_m=position_bound_m,
        extras=result,
    )


def fit_all_geometries(
    frequencies_Hz, z_from_mirror_m, ET, *, config=ModelConfig(), **options
):
    return {
        mode: fit_1d(
            frequencies_Hz,
            z_from_mirror_m,
            ET,
            config=config,
            fit_gaps=adjust_gaps,
            fit_positions=adjust_positions,
            **options,
        )
        for mode, adjust_gaps, adjust_positions in [
            ("fixed", False, False),
            ("gaps", True, False),
            ("positions", False, True),
            ("both", True, True),
        ]
    }


def fit_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    config: ModelConfig | None = None,
    fit_gaps=False,
    fit_positions=False,
    gap_bound_m=0.5e-3,
    position_bound_m=0.5e-3,
    geometry_penalty=1e-3,
    **options,
) -> Path:
    """Read any NPZ with ET, frequencies_Hz, z_from_mirror_m; honor saved physics.

    Saved ModelConfig keys override defaults. An explicit config overrides all
    saved parameters. No pickle loading or legacy filename/layout is required.
    """
    input_path = Path(input_path)
    output_path = (
        Path(output_path)
        if output_path is not None
        else fit_output_path(input_path, "1D")
    )
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
                    saved[field.name] = (
                        tuple(value) if field.name == "gap_widths_m" else value.item()
                    )
            config = ModelConfig(**saved)
        result = fit_1d(
            data["frequencies_Hz"],
            data["z_from_mirror_m"],
            data["ET"],
            config=config,
            fit_gaps=fit_gaps,
            fit_positions=fit_positions,
            gap_bound_m=gap_bound_m,
            position_bound_m=position_bound_m,
            geometry_penalty=geometry_penalty,
            **options,
        )
    return result.save(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", type=Path, help="NPZ containing ET, frequencies_Hz, z_from_mirror_m"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output NPZ (default: INPUT_STEM_fit_1D.npz under data/outputs/)",
    )
    parser.add_argument(
        "--geometry",
        choices=("fixed", "gaps", "positions", "both", "all"),
        default="fixed",
        help="which geometry to adjust (default: fixed); all writes four suffixed NPZ files",
    )
    parser.add_argument(
        "--gap-bound-mm",
        type=float,
        default=0.5,
        help="maximum shift per gap (default: 0.5 mm)",
    )
    parser.add_argument(
        "--position-bound-mm",
        type=float,
        default=0.5,
        help="maximum shift per bead z (default: 0.5 mm)",
    )
    parser.add_argument(
        "--geometry-penalty",
        type=float,
        default=1e-3,
        help="preference for measured geometry; 0 disables it (default: 0.001)",
    )
    parser.add_argument("--n-mc", type=int, default=100)
    parser.add_argument("--relative-error", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args(argv)
    modes = (
        ("fixed", "gaps", "positions", "both")
        if args.geometry == "all"
        else (args.geometry,)
    )
    output = args.output or fit_output_path(args.input, "1D")
    if output.suffix != ".npz":
        output = Path(f"{output}.npz")
    for mode in modes:
        target = (
            output.with_name(output.stem + f"_{mode}.npz")
            if args.geometry == "all"
            else output
        )
        path = fit_file(
            args.input,
            target,
            fit_gaps=mode in ("gaps", "both"),
            fit_positions=mode in ("positions", "both"),
            gap_bound_m=args.gap_bound_mm * 1e-3,
            position_bound_m=args.position_bound_mm * 1e-3,
            geometry_penalty=args.geometry_penalty,
            n_mc=args.n_mc,
            relative_error=args.relative_error,
            seed=args.seed,
        )
        print(f"Longitudinal fit ({mode}) saved to: {path}")


if __name__ == "__main__":
    main()
