1D analysis
===========

**Raw line → radial field → transverse integral → longitudinal fit → boost.**

Notebook: :notebook:`sbp_1d_analysis.ipynb`.
Field reconstruction: :codefile:`src/sbp_analysis.py`.
Longitudinal fit: :codefile:`src/fit_1D.py`.

1. Read traces and subtract the reference
-----------------------------------------

Bind scan-log positions to HDF5 traces by ``trace_index``; average repeated
physical positions before interpolation. For each calibrated thread of length L,
fit a complex baseline to its first/last 25 mm:

.. math::

   (b_0,b_1)=\arg\min_{b_0,b_1}\sum_{s\in\mathrm{edges}}
     |\Gamma(s,\nu)-b_0-b_1s/L|^2,\qquad
   \Delta\Gamma(s,\nu)=\Gamma(s,\nu)-b_0-b_1s/L.

The code sets :math:`\Delta\Gamma` to zero in those reference regions. This radial pipeline does
not use verification's time gate or the 3D drift-template correction.

2. Find the centre and fold the measured diameter
-------------------------------------------------

Use linear interpolation, zero outside the measured interval, and one shared
centre per thread across the frequencies supplied to reconstruction:

.. math::

   s_c=\arg\max_c
   \frac{\int\sum_{\nu}\operatorname{Re}[\Delta\Gamma(s,\nu)^*\Delta\Gamma(2c-s,\nu)]\,ds}
        {\int\sum_{\nu}|\Delta\Gamma(s,\nu)|^2\,ds}.

The implementation searches 129 candidate centres and refines the best bracket.
Keep :math:`R_{\mathrm{int}}=\min(s_c-25\,\mathrm{mm},
L-25\,\mathrm{mm}-s_c)`, then fold:

.. math::

   \Delta\Gamma_r(r,\nu)=\tfrac12[\Delta\Gamma(s_c-r,\nu)+\Delta\Gamma(s_c+r,\nu)],
   \qquad 0\leq r\leq R_{\mathrm{int}}.

The radial grid includes measured distances from the centre, zero and the final
radius. The last folded value is forced to zero where one half enters the
off-beam reference. This is a radial continuation of measured samples, not a
transverse Gaussian or Bessel fit.

3. Convert to field and integrate
---------------------------------

Unwrap :math:`\theta=\arg \Delta\Gamma_r` in radius, then align the centre's branch across
frequency. Apply the conversion factor :math:`C_b(\nu)` supplied by the
:doc:`bead-response interface <bead_response>`:

.. math::

   E_{\mathrm{conv}}(r,\nu)=C_b(\nu)
   \sqrt{|\Delta\Gamma_r|}\,e^{i\theta/2},\qquad
   E_T(\nu)=2\pi\int_0^{R_{\mathrm{int}}}rE_{\mathrm{conv}}(r,\nu)\,dr.

The integral is trapezoidal on the radial grid. Its phase branch is retained
for diagnostics; the longitudinal fit uses :math:`y_{j\nu}=|E_T(z_j,\nu)|`.
The reconstruction file is ``integrated_fields.npz``.

4. Booster-model interface
--------------------------

The 1D booster model is treated as a **black box**. The fit supplies its trial
geometry and the measurement frequencies; the model supplies the responses
needed for comparison and boost calculation.

.. list-table:: Inputs and outputs
   :header-rows: 1
   :widths: 30 70

   * - Quantity
     - Meaning
   * - Inputs
     - Frequency :math:`\nu`, clear disk gaps, thickness/radius, disk material/loss,
       mirror conductivity, bead parameters and thread z coordinates
   * - Field shape
     - Unit-amplitude complex longitudinal shape :math:`e(z,\nu)`
   * - Measurement response
     - Finite-bead magnitude :math:`m_{j\nu}` at each measurement plane
   * - Overlap
     - Longitudinal overlap :math:`I(\nu)` used to calculate boost, including
       the model's mirror and exterior contributions

The finite-bead response is the separate :doc:`bead_response` black box.
The whole bead must lie in vacuum. A fitted amplitude scales the returned
shape and overlap; it is the measurement that sets their normalization.
Internal propagation, interface relations and overlap evaluation are left to
:codefile:`src/fit_1D.py`.

5. Fit amplitudes and geometry
------------------------------

For each trial geometry g, eliminate one nonnegative amplitude per frequency:

.. math::

   a_{\nu}=\frac{\sum_j m_{j\nu}(g)y_{j\nu}}{\sum_jm_{j\nu}(g)^2},\qquad
   \mathcal L(g)=\frac1{N_{\nu}}\sum_{\nu}
      \frac{\sum_j[a_{\nu}m_{j\nu}(g)-y_{j\nu}]^2}{\sum_jy_{j\nu}^2}
      +\lambda\sum_k\left(\frac{g_k-g_{k0}}{b_k}\right)^2.

Zero-signal bins do not constrain geometry. Bounded least squares adjusts the
selected geometry variables; default bounds :math:`b_k=0.5` mm and
:math:`\lambda=0.001` express a preference for nominal positions, not a measured
uncertainty. Bounds must keep every bead in its original vacuum region.

.. list-table:: Four independent fits
   :header-rows: 1

   * - Notebook value
     - Free geometry
   * - ``fixed``
     - None
   * - ``gaps``
     - Three clear disk gaps
   * - ``positions``
     - Four thread z coordinates
   * - ``both``
     - Gaps and thread z

All geometry parameters are shared across the fitted band. This 1D model has no
disk or thread tilt parameters; those belong to :doc:`three_dimensional`.

6. Calculate boost from the returned overlap
--------------------------------------------

Use the returned overlap :math:`I(\nu)` and the fitted amplitude :math:`a_{\nu}`:

.. math::

   \boxed{\beta^2(\nu)=\frac{\epsilon_0\omega^2}
       {8c_0\,\pi R^2P_{\mathrm{in}}}\,|a_{\nu} I(\nu)|^2}.

The plotted relative residual is
:math:`\|a_{\nu}m_{\nu}-y_{\nu}\|_2/\|y_{\nu}\|_2` across threads. Saved fit NPZ files contain
amplitudes, fitted geometry, model fields, overlap, boost and residuals.
This fitter does not calculate statistical uncertainty; geometry-bound lines
and differences between geometry cases are not confidence bands.
