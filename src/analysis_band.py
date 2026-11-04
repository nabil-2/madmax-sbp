"""The measurement analysis uses only the inclusive 20–22 GHz band."""

from pathlib import Path

import h5py
import numpy as np


def frequency_mask(frequencies_Hz):
    """Select the analysis band before any measurement processing."""
    frequency = np.asarray(frequencies_Hz, dtype=float)
    if frequency.ndim != 1 or not np.all(np.isfinite(frequency)):
        raise ValueError("Expected a finite one-dimensional frequency axis.")
    keep = (frequency >= 20e9) & (frequency <= 22e9)
    if not np.any(keep):
        raise ValueError("No measurement frequencies lie within 20–22 GHz.")
    return keep


def require_analysis_band(frequencies_Hz, description):
    """Reject products whose preprocessing may have used other frequencies."""
    if not np.all(frequency_mask(frequencies_Hz)):
        raise ValueError(
            f"{description} contains frequencies outside 20–22 GHz. "
            "Regenerate the analysis from raw traces restricted to 20–22 GHz; "
            "cropping an existing result does not remove its preprocessing dependencies."
        )


def load_raw_traces(run_dir):
    """Read only in-band HDF5 trace columns; leave the raw file unchanged."""
    with h5py.File(Path(run_dir) / "s11_traces.h5", "r") as handle:
        frequency = np.asarray(handle["frequencies_Hz"], dtype=float)
        traces = handle["s_parameter"]
        if traces.ndim != 2 or traces.shape[1] != frequency.size:
            raise ValueError("The trace array must have shape (measurement point, frequency).")
        keep = frequency_mask(frequency)
        return frequency[keep], np.asarray(traces[:, np.flatnonzero(keep)], dtype=complex)
