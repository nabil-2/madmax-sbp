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
     |S(s,f)-b_0-b_1s/L|^2,\qquad
   D(s,f)=S(s,f)-b_0-b_1s/L.

The code sets D to zero in those reference regions. This radial pipeline does
not use verification's time gate or the 3D drift-template correction.

2. Find the centre and fold the measured diameter
-------------------------------------------------

Use linear interpolation, zero outside the measured interval, and one shared
centre per thread across the frequencies supplied to reconstruction:

.. math::

   s_c=\arg\max_c
   \frac{\int\sum_f\operatorname{Re}[D(s,f)^*D(2c-s,f)]\,ds}
        {\int\sum_f|D(s,f)|^2\,ds}.

The implementation searches 129 candidate centres and refines the best bracket.
Keep :math:`R_{\mathrm{int}}=\min(s_c-25\,\mathrm{mm},
L-25\,\mathrm{mm}-s_c)`, then fold:

.. math::

   D_r(r,f)=\tfrac12[D(s_c-r,f)+D(s_c+r,f)],
   \qquad 0\leq r\leq R_{\mathrm{int}}.

The radial grid includes measured distances from the centre, zero and the final
radius. The last folded value is forced to zero where one half enters the
off-beam reference. This is a radial continuation of measured samples, not a
transverse Gaussian or Bessel fit.

3. Convert to field and integrate
---------------------------------

Unwrap :math:`\theta=\arg D_r` in radius, then align the centre's branch across
frequency. Apply the conversion factor :math:`C_b(f)` supplied by the
:doc:`bead-response interface <bead_response>`:

.. math::

   E_{\mathrm{conv}}(r,f)=C_b(f)
   \sqrt{|D_r|}\,e^{i\theta/2},\qquad
   E_T(f)=2\pi\int_0^{R_{\mathrm{int}}}rE_{\mathrm{conv}}(r,f)\,dr.

The integral is trapezoidal on the radial grid. Its phase branch is retained
for diagnostics; the longitudinal fit uses :math:`y_{jf}=|E_T(z_j,f)|`.
The reconstruction file is ``integrated_fields.npz``.

4. Evaluate the longitudinal model and bead response
----------------------------------------------------

Within a layer with index n and local coordinate :math:`\zeta`, write the
unit-amplitude longitudinal shape as

.. math::

   e(\zeta)=A e^{-ikn\zeta}+B e^{+ikn\zeta},\qquad k=\omega/c_0.

Start at the mirror with :math:`A=1`, :math:`B=\rho`, where

.. math::

   n_m=\sqrt{1+i\sigma_m/(\omega\epsilon_0)},\qquad
   \rho=(1-n_m)/(1+n_m).

The default :math:`\sigma_m=5\times10^7\,\mathrm{S/m}` is a model assumption;
infinite conductivity gives :math:`\rho=-1`. In a disk,
:math:`n_d=\sqrt{\epsilon_d(1+i\tan\delta)}`. At a layer exit, propagate by
:math:`p=e^{-iknd}` and enforce the interface relations:

.. math::

   A'=Ap,\quad B'=B/p,\qquad
   \begin{pmatrix}A_{\mathrm{next}}\\B_{\mathrm{next}}\end{pmatrix}
   =\frac1{2n_{\mathrm{next}}}
   \begin{pmatrix}n_{\mathrm{next}}+n&n_{\mathrm{next}}-n\\
                  n_{\mathrm{next}}-n&n_{\mathrm{next}}+n\end{pmatrix}
   \begin{pmatrix}A'\\B'\end{pmatrix}.

The measured magnitude is compared to
:math:`m_{jf}=\sqrt{|Q[e_+,e_-]|}` at each bead position, using the two-direction
:doc:`finite-bead operator <bead_response>`. The whole sphere must lie in vacuum.
The unit shape e becomes a transverse integral after multiplication by the fitted
amplitude; it is not a separately measured local electric field.

5. Fit amplitudes and geometry
------------------------------

For each trial geometry g, eliminate one nonnegative amplitude per frequency:

.. math::

   a_f=\frac{\sum_j m_{jf}(g)y_{jf}}{\sum_jm_{jf}(g)^2},\qquad
   \mathcal L(g)=\frac1{N_f}\sum_f
      \frac{\sum_j[a_fm_{jf}(g)-y_{jf}]^2}{\sum_jy_{jf}^2}
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

6. Integrate longitudinally and calculate boost
-----------------------------------------------

The model accumulates the analytic integral of each layer:

.. math::

   I_{\mathrm{layer}}=\frac{A(1-p)+B(p^{-1}-1)}{ikn}.

Include the conducting-mirror term
:math:`I_m=i(1+\rho)/(kn_m)` and the outgoing exterior Abel limit
:math:`I_{\mathrm{ext}}=(A-B)/(ikn)` at the final boundary.
With :math:`I=I_m+\sum I_{\mathrm{layer}}+I_{\mathrm{ext}}`:

.. math::

   \boxed{\beta^2(f)=\frac{\epsilon_0\omega^2}
       {8c_0\,\pi R^2P_{\mathrm{in}}}\,|a_f I(f)|^2}.

The plotted relative residual is
:math:`\|a_fm_f-y_f\|_2/\|y_f\|_2` across threads. Saved fit NPZ files contain
amplitudes, fitted geometry, model fields, overlap, boost and residuals.
This fitter does not calculate statistical uncertainty; geometry-bound lines
and differences between geometry cases are not confidence bands.
