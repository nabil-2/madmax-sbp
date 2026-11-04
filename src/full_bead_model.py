"""Evaluate MADBead fields from a supplied parameter dictionary for plotting.

This evaluator reads no measurement or reference files. Both spider and
full-reference plots supply their own independently obtained parameters.

The input parameters are central values from ``p_final_ν`` (or one saved Monte
Carlo draw), not a new fit. ``E_0`` scales a transverse field integral, so field
outputs have units V m and the longitudinal integral has units V m².

This preserves MADBead/field_1D_fit.jl's conjugated backward phase inside lossy
dielectrics for reproducibility. The 1D fitting backend now uses that original
Julia implementation directly. This plotting helper reports the longitudinal
integral multiplied by -i; its magnitude is unchanged. A model at mean parameters must not
replace the archive's Monte Carlo-derived boost or uncertainty.
"""

from collections.abc import Mapping

import numpy as np

from .fit_1D import C_LIGHT, EPSILON_0, ModelFields, bead_factors


def _number(parameters, key, *, positive=False, nonnegative=False):
    value = float(parameters[key])
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"Invalid saved full-bead parameter {key!r}: {value}.")
    return value


def _geometry(parameters):
    number = _number(parameters, "n_disk", nonnegative=True)
    count = int(number)
    if count != number:
        raise ValueError("n_disk must be a nonnegative integer.")
    gaps = np.array([_number(parameters, f"d_v_{i}", positive=True) for i in range(1, count+1)])
    thickness = np.array([_number(parameters, f"d_d_{i}", positive=True) for i in range(1, count+1)])
    return gaps, thickness


def disk_edges(parameters: Mapping[str, float]) -> np.ndarray:
    """Return disk [near face, far face] coordinates outward from the mirror."""
    gaps, thickness = _geometry(parameters)
    far_faces = np.cumsum(gaps+thickness)
    return np.column_stack((far_faces-thickness, far_faces))


def evaluate_full_model(frequency_Hz: float, z_from_mirror_m,
                        parameters: Mapping[str, float]) -> ModelFields:
    """Evaluate one saved frequency, preserving each disk's individual properties.

    ``field`` and ``bead_magnitude`` have shape (z, 1), and ``integral`` has
    shape (1,). All outputs include the saved ``E_0`` amplitude. Bead magnitude
    is NaN inside dielectric disks; display code should also exclude bead
    centres closer than one radius to an interface.

    Coordinates are u = z_m - z_absolute. The supplied coordinates already
    use u, so the fitted absolute mirror position is not subtracted again.
    """
    frequency = float(frequency_Hz)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency_Hz must be a positive finite scalar.")
    z = np.atleast_1d(np.asarray(z_from_mirror_m, dtype=float))
    if z.ndim != 1 or not z.size or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("z_from_mirror_m must be a nonempty finite vector with z >= 0.")
    gaps, thickness = _geometry(parameters)
    count = len(gaps)
    permittivity = np.array([_number(parameters, f"ϵ_d_{i}", positive=True) for i in range(1, count+1)])
    loss = np.array([_number(parameters, f"tanD_d_{i}", nonnegative=True) for i in range(1, count+1)])
    amplitude = _number(parameters, "E_0")
    conductivity = float(parameters["σ_m"])
    if np.isnan(conductivity) or conductivity <= 0:
        raise ValueError("σ_m must be positive; infinity gives a perfect mirror.")
    bead_radius = _number(parameters, "r_b", positive=True)
    bead_epsilon = _number(parameters, "ϵ_b", positive=True)
    k = 2*np.pi*frequency/C_LIGHT
    if np.isinf(conductivity):
        rho = -1.0+0j
        mirror_term = 0j
    else:
        mirror_index = np.sqrt(1+1j*conductivity/(2*np.pi*frequency*EPSILON_0))
        rho = (1-mirror_index)/(1+mirror_index)
        mirror_term = -(1+rho)*amplitude/mirror_index
    delta_e, delta_c = bead_factors([frequency], 2*bead_radius, bead_epsilon)
    delta_e, delta_c = delta_e[0], delta_c[0]
    widths = np.empty(2*count)
    widths[::2], widths[1::2] = gaps, thickness
    indices = np.ones(2*count+1, dtype=complex)
    indices[1::2] = np.sqrt(permittivity*(1+1j*loss))
    field = np.empty(z.shape, dtype=complex)
    bead_magnitude = np.full(z.shape, np.nan)
    inward, outward = complex(amplitude), rho*amplitude
    legacy_integral_numerator = mirror_term
    edge = 0.0
    for layer, index in enumerate(indices):
        width = widths[layer] if layer < len(widths) else np.inf
        selected = (z >= edge) & (z < edge+width)
        phase = np.exp(-1j*k*index*(z[selected]-edge))
        # Preserve the legacy evaluator exactly, including its lossy-disk
        # conjugation. Interface propagation below uses reciprocal phases.
        forward = inward*phase
        backward = outward*phase.conj()
        field[selected] = forward+backward
        if layer % 2 == 0:
            response = delta_e*(forward**2+backward**2)+2*delta_c*inward*outward
            bead_magnitude[selected] = np.sqrt(abs(response))
        if layer == len(widths):
            # Splitting the exterior vacuum at absolute z=0 in MADBead is
            # algebraically equivalent to this single adiabatic tail.
            legacy_integral_numerator += (inward-outward)/index
            break
        p = np.exp(-1j*k*index*width)
        legacy_integral_numerator += (inward*(1-p)+outward*(p.conjugate()-1))/index
        inward, outward = inward*p, outward/p
        next_index = indices[layer+1]
        inward, outward = (
            ((next_index+index)*inward+(next_index-index)*outward)/(2*next_index),
            ((next_index-index)*inward+(next_index+index)*outward)/(2*next_index),
        )
        edge += width
    integral = -1j*legacy_integral_numerator/k
    return ModelFields(field[:, None], bead_magnitude[:, None], np.array([integral]))
