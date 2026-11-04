3D analysis
===========

**Measured complex chords + independent beam measurement → geometry fit →
receive-coupled boost → conditional uncertainty.**

Notebook: :notebook:`sbp_3d_analysis.ipynb`.
Preparation: :codefile:`src/analysis_3d_inputs.py`.
Fit: :codefile:`src/analysis_3d.py`.

Fitting model: `BoostFractor.jl <https://github.com/nabil-2/BoostFractor.jl>`_,
used through this repository's scalar-field adapters for tilted disks and
thread paths. The booster model is treated as a black box.

1. Prepare the measured reflection
----------------------------------

Read only raw :math:`\Gamma` columns within the inclusive 20–22 GHz band before
profile fitting or frequency selection. The same raw-data loader is used by
the 1D pipeline. Align the scan log by ``trace_index`` and retain the complex
reflection values unchanged:

.. math::

   \Gamma_i(\nu)=\Gamma_i^{\mathrm{measured}}(\nu),\qquad
   20\leq\nu/\mathrm{GHz}\leq22.

No endpoint-derived template is subtracted and no separate drift coefficient
is estimated. Complex affine backgrounds are fitted together with the model
response in Step 6. Trace IDs 73, 74, 192, 193, 194 and 195 remain excluded from
the profile fits and the 3D likelihood. The remaining 324 points include **T0–T3**.
The prepared archive stores ``observed_S22`` and
``reflection_preprocessing="none"``; old template-corrected input archives are
rejected and must be regenerated.

The current results use a fresh fit of all 17 geometry parameters to measured
S22 at the 35 quality-selected frequencies. The fitted geometry is then used
for all 401 frequencies in 20–22 GHz, including new common scales, affine
backgrounds, conditional bands and field maps. The bounds and incident-beam
assumptions remain part of this conditional result.

2. Select the geometry-fit frequencies
--------------------------------------

Preliminary fits to each measured chord use dimensionless
:math:`u=s/(0.1\,\mathrm m)` and

.. math::

   \Gamma(u,\nu)=b_0+b_1u+A\exp\{[-2/w^2+i\eta](u-u_c)^2\}.

For each trial :math:`(u_c,w,\eta)`, solve :math:`(b_0,b_1,A)` by complex least
squares, then optimize the real/imaginary residuals. Bounds are
:math:`u_c\in[-1.1,1.1]`, :math:`w\in[0.2,2.2]`, :math:`\eta\in[-8,8]`.
Continuation across frequency has a default-seed restart when needed.
Accept a profile if optimization succeeds, no parameter is within
:math:`10^{-3}` of its bounds, and :math:`\mathrm{RMS}(\Gamma-\widehat\Gamma)/|A|<0.20`.

The shared mask requires acceptance on **T1–T3**, giving 35 frequencies in the
current uncorrected measurement. T0 is still included in the full-tilt likelihood.
No fitted Gaussian is rotated or supplied as a field to that likelihood.

The preparation file also stores a circular-Gaussian diagnostic integral:

.. math::

   E_{T,\mathrm{diag}}=\sqrt{A/\kappa_b}\;2\pi e^{-q s_c^2}
       \int_0^R r e^{-qr^2}I_0(2qrs_c)\,dr,
   \qquad q=\frac{w^{-2}-i\eta/2}{(0.1\,\mathrm m)^2},\quad
   s_c=0.1u_c\,\mathrm m.

Here :math:`\kappa_b(\nu)` is supplied by the black-box bead response
with the 1-W convention, and :math:`I_0` is the modified
Bessel function. The integral uses 64-point
Gauss–Legendre quadrature. This diagnostic does **not** enter the 3D objective.

For the bead response, see J. Egge et al., *Experimental determination of
axion signal power of dish antennas and dielectric haloscopes using the
reciprocity approach*, `JCAP 04 (2024) 005
<https://doi.org/10.1088/1475-7516/2024/04/005>`_, Section 3 and Appendix B.

3. Supply the independent incident beam
---------------------------------------

Discard transmission-beam samples outside 20–22 GHz before interpolation.
Interpolate the six independent transmission-beam (:math:`S_{31}`) Gaussian
magnitude fits in frequency. At each :math:`\nu`:

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
this geometry fit.

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

The bead-response formula is Eq. (5.6) of
`Egge et al., JCAP 04 (2024) 005 <https://arxiv.org/pdf/2311.13359#page=12>`_.
In the paper's notation, the full equation is

.. math::

   \frac{4P_{\mathrm{in}}\Delta\Gamma}{i\omega\alpha_0}
      =\delta_e^+\left(E_+^2e^{-2ikz}+E_-^2e^{+2ikz}\right)
       +2\delta_c^+E_+E_-.

The exponential factors describe propagation to the bead position. With local
wave amplitudes :math:`e_+=E_+e^{-ikz}` and :math:`e_-=E_-e^{+ikz}`, and
writing :math:`\delta_e,\delta_c` for :math:`\delta_e^+,\delta_c^+`, the
right-hand side becomes

.. math::

   Q=\delta_e(e_+^2+e_-^2)+2\delta_c e_+e_-.

The 3D model supplies the local inward/outward fields, including their
propagation phases. In the Mie-factor phasor convention these are
:math:`e_-=F_{\mathrm{inward}}^*` and
:math:`e_+=F_{\mathrm{outward}}^*`; they are substituted into :math:`Q` above.
The factors :math:`\delta_e,\delta_c` come from the paper's Appendix B.

Using the SI prefactor from the paper's Eq. (3.1) and our phasor conventions
gives the simulated bead-induced reflection:

.. math::

   \boxed{t=\Delta\Gamma_{\mathrm{sim}}=(\kappa_b Q)^*},\qquad
   \kappa_b=\frac{i\epsilon_0\omega\alpha_0}{4P_{\mathrm{in}}},\qquad
   \alpha_0=4\pi r_b^3\frac{\epsilon_b-1}{\epsilon_b+2}.

Here :math:`\omega=2\pi\nu` and :math:`P_{\mathrm{in}}=1\,\mathrm W`.
The conjugations convert between the solver, Mie and recorded-S22 phasor
conventions; :math:`\epsilon_0` is explicit because this :math:`\alpha_0`
has units of volume. Step 6 fits the common complex scale and adds the affine
backgrounds to compare with measured S22.

6. Profile the backgrounds and common complex scale
---------------------------------------------------

For each thread, let :math:`X_j=[1,u]` and
:math:`P_j=I-X_jX_j^+`, where + denotes the pseudoinverse. At every frequency:

.. math::

   c_{\nu}=\frac{\sum_j(P_jt_{j\nu})^\dagger(P_j\Gamma_{j\nu})}
                  {\sum_j\|P_jt_{j\nu}\|^2},\qquad
   b_{j\nu}=X_j^+(\Gamma_{j\nu}-c_{\nu}t_{j\nu}),\qquad
   \widehat\Gamma_{j\nu}=c_{\nu}t_{j\nu}+X_jb_{j\nu}.

One :math:`c_{\nu}` is shared by all four threads. The baseline-subtracted signal
and its fitted prediction are

.. math::

   \Delta\Gamma_{j\nu}=\Gamma_{j\nu}-X_jb_{j\nu},\qquad
   \widehat{\Delta\Gamma}_{j\nu}=c_{\nu}t_{j\nu}.

Geometry minimizes

.. math::

   r_{\nu}^2=\frac{\sum_j\|c_{\nu}P_jt_{j\nu}-P_j\Gamma_{j\nu}\|^2}
                    {\sum_j\|P_j\Gamma_{j\nu}\|^2},\qquad
   \mathcal L=\frac1{|\mathcal F_{\mathrm{fit}}|}
                     \sum_{\nu\in\mathcal F_{\mathrm{fit}}}r_{\nu}^2.

SLSQP enforces the geometry constraints. Frequencies have equal normalized
weight; samples within a frequency have equal absolute weight, so stronger
threads dominate. Individual-thread residuals use the same c. The notebook's
optional fit is one explicit start. The current displayed geometry has been
reoptimized against measured S22 using the 35-bin profile-quality selection.
A small pooled residual does not establish a unique physical geometry.

7. Calculate boost and diagnostic fields
----------------------------------------

The 3D booster model computes the receive-coupled geometry response
:math:`\beta^2_{\mathrm{recv},g}` for the fixed 1-W beam. Since the reflection
template is quadratic in field, its fitted amplitude factor is
:math:`\sqrt{|c_{\nu}|}` and its power factor is :math:`|c_{\nu}|`:

.. math::

   \boxed{\beta^2(\nu)=|c_{\nu}|\,\beta^2_{\mathrm{recv},g}(\nu)},\qquad
   |E|_{\mathrm{plot}}=\sqrt{|c_{\nu}|}\,|e_++e_-|.

Maps are horizontal sections at thread midpoint z, not sloped sampling paths.
Blank pixels follow aperture/clearance masks; exterior T0 extends beyond the disk.
Line-response plots instead show the finite-bead proxy
:math:`\sqrt{|c_{\nu}Q|}`. A 401-bin bank is needed to recompute the full 20–22 GHz
spectrum. The local fitting bank contains all 35 quality-selected bins;
``data/3d/bank_full401/`` supplies the dense evaluation. These banks reuse
unchanged propagation operators at the configured basis, grid and frequency
resolution; disk interfaces and fields are solved anew for each geometry.
The fitting wrapper rejects an incomplete bank.

8. Form conditional uncertainty bands
-------------------------------------

Code: :codefile:`src/analysis_3d_exports.py`.
Hold geometry, beam, materials and calibration fixed. Demean the complex
residuals :math:`e_j=\Gamma_j-\widehat\Gamma_j` per thread/frequency and resample circular
blocks of eight adjacent chord samples. The same sampled indices are used
across frequencies. Each draw updates only the common scale:

.. math::

   c_{\nu}^{(b)}=c_{\nu}+
      \frac{\sum_j(P_jt_{j\nu})^\dagger e_{j\nu}^{(b)}}
           {\sum_j\|P_jt_{j\nu}\|^2},\qquad
   \beta^{2(b)}_{\nu}=|c_{\nu}^{(b)}|\,\beta^2_{\mathrm{recv},g,\nu}.

Default: 1,000 draws, seed 20261025. Stored pointwise quantiles are
2.5%, 16%, 50%, 84% and 97.5%; plots shade 16–84%. These are conditional
residual-resampling bands, not total or simultaneous confidence intervals.
They exclude geometry, beam/phase/registration, materials, absolute calibration
and model discrepancy. Several saved fitted coordinates reach their bounds.
