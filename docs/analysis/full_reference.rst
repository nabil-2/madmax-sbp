Full bead-pull comparison
=========================

Code: :codefile:`src/full_bead_comparison.py`, :codefile:`src/full_bead_model.py`.
The 1D/3D notebooks display this comparison alongside their own results.

1. Read saved results, without rerunning the reference fit
----------------------------------------------------------

``result_bf_21_0.jld2`` contains a 20.5–21.5 GHz sweep with 1,001 bins,
28 longitudinal transverse-integral samples, fitted parameters and 100 saved
Monte Carlo fit draws per frequency. The reader extracts numeric values and
saved uncertainties, then caches them by the source-file hash. It does not
analyze a raw full-grid reflection scan here.

2. Match frequencies and coordinates
------------------------------------

Only acquired frequencies matching within 1 Hz are compared; no frequency
interpolation is used. The displayed spider grid has 201 common bins. Each
setup uses its own mirror origin:

.. math::

   z_{\mathrm{full,out}}=z_{m,\mathrm{fit}}(\nu)-z_{\mathrm{saved}}.

This aligns mirror coordinates, not geometries or sampled z positions.

3. Compare fields, boost and residuals
--------------------------------------

The full-reference boost and its uncertainty are read **as saved**. Reference
field curves are evaluated at the saved mean parameters, including each disk's
own thickness, permittivity and loss. They are not Monte Carlo mean fields.
Reference curves retain the archive's own model-evaluation convention.

Absolute panels retain each result's normalization. Shape panels divide each
boost curve by its own maximum, and each field-frequency column by its own
maximum where indicated. Magnitude residuals use

.. math::

   r(\nu)=\frac{\|m(z,\nu)-|E_T(z,\nu)|\|_2}{\||E_T(z,\nu)|\|_2}

on each measurement's own sampled z. Spider 3D field-integral points are model
integrals over valid horizontal midpoint sections, not new measurements along z.

The normalization audit finds a factor approximately **20.16129** between the
saved reference boost and the spider formula with the archive's saved Monte
Carlo overlap convention. Its physical power/coupling metadata are incomplete;
no inferred correction is applied. Consequently, absolute agreement of the
two boost curves is not an independently calibrated validation.
