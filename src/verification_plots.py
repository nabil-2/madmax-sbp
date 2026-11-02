"""Plots for SBP verification; figures are returned for notebook display/save."""
import matplotlib.pyplot as plt
import numpy as np
from .verification_core import beam_model
from .sbp_verification import match_full_to_reduced


def plot_fields(run, *, z_index=0, frequency_index=0):
    """Full-grid field magnitude, fitted field and their residual, at one plane."""
    data = run['data']
    original_z = data['z_ixs_used'][z_index]
    full = data['E_full'][:, :, original_z, frequency_index]
    fit = beam_model(data['x_plt'], data['y_plt'], *run['params'][z_index, frequency_index])
    maximum = max(np.max(abs(full)), np.max(abs(fit)))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained')
    x, y = data['x_pos'] * 1e3, data['y_pos'] * 1e3
    for ax, field, title in zip(axes[:2], (full, fit), ('Full bead pull', 'Limited-line fit')):
        image = ax.pcolormesh(x, y, abs(field).T, shading='auto', vmin=0, vmax=maximum)
        ax.set_title(title)
    fig.colorbar(image, ax=list(axes[:2]), label=r'$|E_{\mathrm{conv}}|$ [V/m]')
    axes[0].plot(data['x_reduced'] * 1e3, data['y_reduced'] * 1e3, '.', color='white', ms=2)
    residual = abs(full - fit) / max(np.linalg.norm(full) / np.sqrt(full.size), 1e-30)
    image = axes[2].pcolormesh(x, y, residual.T * 100, shading='auto', cmap='magma')
    axes[2].set_title('Complex residual\n/ full-grid RMS')
    fig.colorbar(image, ax=axes[2], label='%')
    for ax in axes:
        ax.set(xlabel='x [mm]', ylabel='y [mm]', aspect='equal')
    f = data['frequencies'][frequency_index] * 1e-9
    z = data['z_pos'][original_z] * 1e3
    fig.suptitle(f'{f:.3f} GHz · acquisition z = {z:.1f} mm')
    return fig


def plot_integrals(result, *, frequency_index=0):
    """Compare complex transverse integrals along the acquired z coordinate."""
    full = result['ETs_full'][:, frequency_index]
    reduced = result['ETs_reduced'][:, frequency_index]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained')
    for ax, transform, label in zip(axes, (np.abs, np.real, np.imag),
                                    (r'$|E_T|$', r'$\mathrm{Re}(E_T)$', r'$\mathrm{Im}(E_T)$')):
        ax.plot(result['z_full'] * 1e3, transform(full), 'o', label='Full bead pull')
        ax.plot(result['z_reduced'] * 1e3, transform(reduced), 'x', ms=7, label='Limited lines')
        ax.set(xlabel='Acquisition z [mm]', ylabel=label + ' [V m]')
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(f"{result['frequencies'][frequency_index] * 1e-9:.3f} GHz")
    return fig


def plot_integral_error(result):
    """Relative complex ET error over matched planes, separately per frequency."""
    full, _ = match_full_to_reduced(result['ETs_full'], result['z_full'], result['z_reduced'])
    denominator = np.linalg.norm(full, axis=0)
    error = np.full(denominator.shape, np.nan)
    np.divide(np.linalg.norm(result['ETs_reduced'] - full, axis=0), denominator,
              out=error, where=denominator > 0)
    fig, ax = plt.subplots(figsize=(7, 3), layout='constrained')
    ax.plot(result['frequencies'] * 1e-9, error * 100, '.-')
    ax.set(xlabel='Frequency [GHz]', ylabel='Complex ET error [%]')
    ax.grid(alpha=.2)
    return fig

