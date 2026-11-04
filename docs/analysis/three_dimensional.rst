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

The local displayed results have been updated to measured S22 by recomputing
the common scale, affine backgrounds, conditional bands and field amplitudes
at the saved geometry. **Geometry has not been reoptimized after removing the
template correction.** Its original 33-bin fitting mask is preserved separately
from the new raw-data profile-quality mask.

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

The geometry fit compares measured complex S22 with a predicted bead-induced
reflection plus the fitted backgrounds in Step 6. The inverse problem is
recovering geometry; predicting the bead response for each trial geometry is
the **forward measurement model**.

From the paper to the implemented response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Egge et al., JCAP 04 (2024) 005
<https://doi.org/10.1088/1475-7516/2024/04/005>`_ relate the reflection change
to the fields with and without the bead. Starting from Eq. (3.1) of the
`arXiv PDF <https://arxiv.org/pdf/2311.13359#page=6>`_, write

.. math::

   \Delta\Gamma=-iC_b\mathcal Q,\qquad
   C_b=\frac{\epsilon_0\omega\alpha_0}{4P_{\mathrm{in}}},\qquad
   \mathcal Q=\frac{\epsilon_b-1}{\alpha_0}
       \int_{V_b}\mathbf E_b\cdot\mathbf E\,dV,

.. math::

   \alpha_0=4\pi r_b^3\frac{\epsilon_b-1}{\epsilon_b+2},\qquad
   \omega=2\pi\nu.

Here :math:`\mathbf E_b` is the field inside the bead and :math:`\mathbf E`
is the field without it. This definition of :math:`\alpha_0` has units of
volume, so the SI prefactor includes :math:`\epsilon_0` explicitly.

The sphere response is linear in the exciting field. Expanding the bilinear
integral for two counterpropagating waves therefore gives two self terms and
one cross term, as in the paper's Eq. (5.6). At the bead centre, the propagation
phases are already included in the local directional fields: for example,
:math:`e_+=A_+e^{-ikz}` and :math:`e_-=A_-e^{+ikz}` give
:math:`e_+^2=A_+^2e^{-2ikz}` and :math:`e_+e_-=A_+A_-`.
Using the Mie factors in the repository's bead-response convention,

.. math::

   Q=\delta_e(e_+^2+e_-^2)+2\delta_c e_+e_-.

The self and cross factors are computed by :codefile:`src/bead_response.py`;
the 3D model applies this local two-wave approximation to its scalar fields.
The solver and Mie calculation use opposite phasor conventions. The code sets
:math:`e_-=F_{\mathrm{inward}}^*` and
:math:`e_+=F_{\mathrm{outward}}^*`. Conjugating fields and Mie factors together
expresses the same response in the solver/recorded-reflection convention:

.. math::

   \mathcal Q=Q^*
   =\delta_e^*(F_{\mathrm{inward}}^2+F_{\mathrm{outward}}^2)
      +2\delta_c^*F_{\mathrm{inward}}F_{\mathrm{outward}}.

Substitution into the reflection relation above gives the exact form used by
``finite_bead_field`` in :codefile:`src/receive_field.py` and ``evaluate`` in
:codefile:`src/analysis_3d.py`:

.. math::

   \boxed{
   t_{ji\nu}=\Delta\Gamma_{\mathrm{sim},ji\nu}
      =-iC_bQ_{ji\nu}^*
      =\bigl[\kappa_b(\nu)Q_{ji\nu}\bigr]^*,
   \qquad \kappa_b=iC_b.}

The implementation uses :math:`P_{\mathrm{in}}=1\,\mathrm W`. This rearranges
the field--reflection relation to predict reflection from simulated fields;
the common complex coefficient :math:`c_\nu` in Step 6 then scales that prediction.

Why fit reflection instead of a reconstructed field?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the point-bead limit, :math:`\delta_e=\delta_c=1`, so
:math:`Q=(e_++e_-)^2` and a square root recovers the total field up to sign.
For a finite bead,

.. math::

   Q=\delta_e E^2+2(\delta_c-\delta_e)e_+e_-,\qquad E=e_++e_-.

Thus the measured response alone does not determine the unperturbed field:
the incoming/outgoing mixture is also needed. If :math:`B_j` denotes the affine
background, the alternative measurement-to-field inverse would be

.. math::

   E_{\mathrm{eff,meas}}=\pm\sqrt{\frac{(\Gamma-B_j)^*}{\kappa_b}},\qquad
   E_{\mathrm{eff,model}}^2=\frac{(c_\nu t)^*}{\kappa_b}=c_\nu^*Q.

Even with unit normalization this recovers :math:`\pm\sqrt Q`, a finite-bead
field proxy. Correcting it to the unperturbed field requires model information;
the background must also be known or fitted. The field-proxy plots display
the magnitude of this quantity.

Fitting S22 also avoids square-root branch choices, nonlinear noise propagation
near field nodes and a separate background subtraction before fitting. A
field-space comparison is possible with consistent bead response, branches and
uncertainties; it adds no information. The current fit therefore retains measured
S22, with no empirical reflection-template correction, and fits its affine
backgrounds jointly with the physical response.

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
optional fit is one explicit start; the historical saved geometry was selected
from multiple starts using the former preprocessing. Current displayed curves
refit scale and backgrounds at that geometry using measured S22.
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
spectrum. The existing local fitting bank contains the historical 33 bins;
rebuild it with the current ``config/3d_bank.toml`` before optimizing over all
35 raw-data quality-selected bins. The fitting wrapper rejects an incomplete bank.

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
