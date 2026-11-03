3D analysis
===========

**Corrected complex chords + independent beam measurement → geometry fit →
receive-coupled boost → conditional uncertainty.**

Notebook: :notebook:`sbp_3d_analysis.ipynb`.
Preparation: :codefile:`src/analysis_3d_inputs.py`.
Fit: :codefile:`src/analysis_3d.py`.

1. Prepare the measured complex S22
-----------------------------------

Keep only 20–22 GHz and align the scan log by ``trace_index``. Construct a
complex drift template T from the last five minus first five T0 traces.
For every point i on thread j, let :math:`d_i(f)=S_i(f)-\bar S_{j,\mathrm{first5}}(f)`.
Fit a **real** template coefficient only in :math:`\mathcal C=[21.4,22]` GHz:

.. math::

   a_i=\frac{\sum_{f\in\mathcal C}w_{if}\operatorname{Re}[d_i(f)T(f)^*]}
                 {\sum_{f\in\mathcal C}w_{if}|T(f)|^2},\qquad
   y_i(f)=S_i(f)-a_iT(f).

Start with unit weights and perform six robust updates:

.. math::

   e_{if}=d_i-a_iT,\quad
   s_i=\max(\operatorname{median}_{f\in\mathcal C}|e_{if}|,10^{-5}),\quad
   w_{if}=\min\!\left(1,\frac{1.5s_i}{\max(|e_{if}|,10^{-16})}\right).

This is an empirical additive correction, not an independently calibrated
instrument model. Trace IDs 73, 74, 192, 193, 194 and 195 are excluded from the
profile fits and the 3D likelihood. The remaining 324 points include **T0–T3**.

2. Select the geometry-fit frequencies
--------------------------------------

Preliminary fits to each corrected chord use dimensionless
:math:`u=s/(0.1\,\mathrm m)` and

.. math::

   y(u)=b_0+b_1u+A\exp\{[-2/w^2+i\eta](u-u_c)^2\}.

For each trial :math:`(u_c,w,\eta)`, solve :math:`(b_0,b_1,A)` by complex least
squares, then optimize the real/imaginary residuals. Bounds are
:math:`u_c\in[-1.1,1.1]`, :math:`w\in[0.2,2.2]`, :math:`\eta\in[-8,8]`.
Continuation across frequency has a default-seed restart when needed.
Accept a profile if optimization succeeds, no parameter is within
:math:`10^{-3}` of its bounds, and :math:`\mathrm{RMS}(y-\hat y)/|A|<0.20`.

The shared mask requires acceptance on **T1–T3**, giving 33 frequencies in the
saved data. T0 is still included in the subsequent full-tilt likelihood.
No fitted Gaussian is rotated or supplied as a field to that likelihood.

The preparation file also stores a circular-Gaussian diagnostic integral:

.. math::

   E_{T,\mathrm{diag}}=\sqrt{A/\kappa_b}\;2\pi e^{-q s_c^2}
       \int_0^R r e^{-qr^2}I_0(2qrs_c)\,dr,+   \qquad q=\frac{w^{-2}-i\eta/2}{(0.1\,\mathrm m)^2},\quad
   s_c=0.1u_c\,\mathrm m.

Here :math:`\kappa_b=i\omega\epsilon_0\alpha_0/4` for 1 W and
:math:`I_0` is the modified Bessel function. The integral uses 64-point
Gauss–Legendre quadrature. This diagnostic does **not** enter the 3D objective.

3. Supply the independent incident beam
---------------------------------------

Interpolate the six measured S31 Gaussian magnitude fits in frequency. At each f:

.. math::

   g(\mathbf r)=\sum_{\ell=1}^{6}A_\ell
       e^{-|\mathbf r-\mathbf r_\ell|^2/w_\ell^2},\qquad
   N^2=\sum_{\ell,m}A_\ell A_m
      \frac{\pi w_\ell^2w_m^2}{w_\ell^2+w_m^2}
      e^{-|\mathbf r_\ell-\mathbf r_m|^2/(w_\ell^2+w_m^2)}.

Then :math:`E_{\mathrm{in}}=\sqrt{2Z_0P_{\mathrm{in}}}\,g/N` has 1 W over the
whole infinite transverse plane. It is not renormalized inside the disk aperture.
The supplied magnitude data have no measured phase; the baseline assumes flat
phase. Scan z = 50–100 mm is not registered to the spider mirror. Its profile
mixture is placed at the model's outer reference face; propagation from a known
physical measurement plane is not inferred. Beam parameters stay fixed during
this geometry fit. The :doc:`model_interface` describes the solver inputs and outputs.

4. Parameterize disks and actual bead paths
-------------------------------------------

For disk d and thread j:

.. math::

   z_{d,\mathrm{near}}(x,y)=z_{d0}+\Delta z_d+s_{dx}x+s_{dy}y,\qquad
   z_{d,\mathrm{far}}=z_{d,\mathrm{near}}+1\,\mathrm{mm},

.. math::

   v_i=\frac{s_i-s_{\min}}{s_{\max}-s_{\min}},\qquad
   z_{ji}=(1-v_i)(z_{j0}+\Delta z_{jA})+
                    v_i(z_{j0}+\Delta z_{jB}).

Recorded x/y stay fixed. A/B are the retained scan endpoints, not mechanical
supports; paths are straight, without sag. There are **17** shared parameters:
three disk shifts, six disk slopes and eight endpoint shifts.

Central gap changes are bounded by ±2 mm, slope components by ±5 mrad and
endpoint shifts by ±2 mm. Continuous-path constraints require at least one
bead radius of normal clearance from neighboring face planes and the mirror.
Face-plane checks cover the chord clipped to radius :math:`R+r_b`; they are
conservative near the finite disk rim. Additional constraints keep propagation
distances inside the bank/exterior evaluator's domain. There is no extra
nominal-geometry penalty in this 3D objective.

5. Predict the bead response at every measured point
----------------------------------------------------

The solver returns incoming/outgoing directional fields. Convert their
phasor convention before applying the shared :doc:`bead_response`:

.. math::

   e_-=(E_{\mathrm{inward}})^*,\quad
   e_+=(E_{\mathrm{outward}})^*,\qquad
   Q=\delta_e(e_+^2+e_-^2)+2\delta_c e_+e_-,\qquad
   t_{jif}=\overline{\kappa_b(f)Q_{jif}}.

The final conjugation is part of the stored-S22 convention and is retained from
the original analysis. The likelihood uses this complex quadratic template;
it never fits square-root phases or an independently adjusted boost curve.

6. Profile the backgrounds and common complex scale
---------------------------------------------------

For each thread, let :math:`X_j=[1,u]` and
:math:`P_j=I-X_jX_j^+`, where + denotes the pseudoinverse. At every frequency:

.. math::

   c_f=\frac{\sum_j(P_jt_{jf})^\dagger(P_jy_{jf})}
                  {\sum_j\|P_jt_{jf}\|^2},\qquad
   b_{jf}=X_j^+(y_{jf}-c_ft_{jf}),\qquad
   \hat y_{jf}=c_ft_{jf}+X_jb_{jf}.

One c is shared by all four threads. Geometry minimizes

.. math::

   r_f^2=\frac{\sum_j\|c_fP_jt_{jf}-P_jy_{jf}\|^2}
                    {\sum_j\|P_jy_{jf}\|^2},\qquad
   \mathcal L=\frac1{|\mathcal F_{\mathrm{fit}}|}
                     \sum_{f\in\mathcal F_{\mathrm{fit}}}r_f^2.

SLSQP enforces the geometry constraints. Frequencies have equal normalized
weight; samples within a frequency have equal absolute weight, so stronger
threads dominate. Individual-thread residuals use the same c. The notebook's
optional fit is one explicit start; the saved result was selected from multiple
starts. A small pooled residual does not establish a unique physical geometry.

7. Calculate boost and diagnostic fields
----------------------------------------

The :doc:`model_interface` computes the receive-coupled geometry response
:math:`\beta^2_{\mathrm{recv},g}` for the fixed 1-W beam. Since the reflection
template is quadratic in field, its fitted amplitude factor is
:math:`\sqrt{|c_f|}` and its power factor is :math:`|c_f|`:

.. math::

   \boxed{\beta^2(f)=|c_f|\,\beta^2_{\mathrm{recv},g}(f)},\qquad
   |E|_{\mathrm{plot}}=\sqrt{|c_f|}\,|e_++e_-|.

Maps are horizontal sections at thread midpoint z, not sloped sampling paths.
Blank pixels follow aperture/clearance masks; exterior T0 extends beyond the disk.
Line-response plots instead show the finite-bead proxy
:math:`\sqrt{|c_fQ|}`. A 401-bin bank is needed to recompute the full 20–22 GHz
spectrum; the local fitting bank contains only the 33 selected bins.

8. Form conditional uncertainty bands
-------------------------------------

Code: :codefile:`src/analysis_3d_exports.py`.
Hold geometry, beam, materials and calibration fixed. Demean the complex
residuals :math:`e_j=y_j-\hat y_j` per thread/frequency and resample circular
blocks of eight adjacent chord samples. The same sampled indices are used
across frequencies. Each draw updates only the common scale:

.. math::

   c_f^{(b)}=c_f+
      \frac{\sum_j(P_jt_{jf})^\dagger e_{jf}^{(b)}}
           {\sum_j\|P_jt_{jf}\|^2},\qquad
   \beta^{2(b)}_f=|c_f^{(b)}|\,\beta^2_{\mathrm{recv},g,f}.

Default: 1,000 draws, seed 20261025. Stored pointwise quantiles are
2.5%, 16%, 50%, 84% and 97.5%; plots shade 16–84%. These are conditional
residual-resampling bands, not total or simultaneous confidence intervals.
They exclude geometry, beam/phase/registration, materials, absolute calibration
and model discrepancy. Several saved fitted coordinates reach their bounds.
