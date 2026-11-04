1D analysis
===========

**Spider traces → transverse integrals → geometry extension → 1D results.**

Notebook: :notebook:`sbp_1d_analysis.ipynb`.
This page explains the SBP preparation and geometry extensions. The
`existing 1D analysis pipeline
<https://github.com/nabil-2/spider-bead-pull-ai/blob/main/5_Analysis_Boostfactor_Determination_Spider_ET.ipynb>`_
already covers longitudinal amplitude fitting, the finite-bead response,
longitudinal field integration, boost calculation and Monte Carlo uncertainty
propagation. Those operations are treated as a **black box** here.

The backend uses functions copied from that notebook and the MADBead routines
it calls: :codefile:`src/existing_1d_functions.jl`, :codefile:`src/run_1d.jl`
and ``src/madbead/``. :codefile:`src/fit_1D.py` adapts numeric inputs and outputs.
Our raw-to-transverse-integral logic stays in :codefile:`src/sbp_analysis.py`.

Raw data are filtered to **20–22 GHz before preprocessing**. The current 1D
figures use this pipeline with 100 Monte Carlo draws per retained frequency.
All four geometry cases use T0–T3; the full bead-pull reference enters only
the comparison plots after fitting.

1. Prepare the spider traces
----------------------------

Code: :codefile:`src/analysis_band.py`, :codefile:`src/sbp_analysis.py`.

Keep only raw columns satisfying :math:`20\leq\nu/\mathrm{GHz}\leq22`, then join
calibrated positions to traces by ``trace_index``. Average repeated physical
positions. For each thread of length L, estimate the complex background from its
first/last 25 mm:

.. math::

   (b_0,b_1)=\arg\min_{b_0,b_1}\sum_{s\in\mathrm{edges}}
      |\Gamma(s,\nu)-b_0-b_1s/L|^2,\qquad
   \Delta\Gamma(s,\nu)=\Gamma(s,\nu)-b_0-b_1s/L.

Set :math:`\Delta\Gamma=0` in those edge windows. This preparation uses neither
time gating nor the 3D drift correction.

2. Reconstruct the transverse integrals
---------------------------------------

Fit one centre per thread, shared across the retained frequencies, by maximizing
complex reflection symmetry:

.. math::

   s_c=\arg\max_c
   \frac{\int\sum_\nu\operatorname{Re}
      [\Delta\Gamma(s,\nu)^*\Delta\Gamma(2c-s,\nu)]\,ds}
        {\int\sum_\nu|\Delta\Gamma(s,\nu)|^2\,ds}.

Use linear interpolation with zero extension. Fold the measured diameter about
that centre, assuming rotational symmetry of its complex response:

.. math::

   \Delta\Gamma_r(r,\nu)=\tfrac12
       [\Delta\Gamma(s_c-r,\nu)+\Delta\Gamma(s_c+r,\nu)],\qquad
   R_{\mathrm{int}}=\min(s_c-25\,\mathrm{mm},L-25\,\mathrm{mm}-s_c).

Unwrap :math:`\theta=\arg\Delta\Gamma_r` in radius and align its centre branch
across frequency. Apply the bead conversion as a black box and integrate:

.. math::

   E_{\mathrm{conv}}=C_b(\nu)\sqrt{|\Delta\Gamma_r|}\,e^{i\theta/2},\qquad
   E_T(z_j,\nu)=2\pi\int_0^{R_{\mathrm{int}}}rE_{\mathrm{conv}}(r,\nu)\,dr.

The integration uses the measured radial grid and the trapezoidal rule; its last
sample is forced to zero at the off-beam boundary. No Gaussian or Bessel profile
is fitted. The result is ``integrated_fields.npz``, with T0–T3, their z coordinates
and frequencies. The longitudinal fitting input is :math:`y_{j\nu}=|E_T(z_j,\nu)|`.

For the bead response, see J. Egge et al., *Experimental determination of axion
signal power of dish antennas and dielectric haloscopes using the reciprocity
approach*, `JCAP 04 (2024) 005
<https://doi.org/10.1088/1475-7516/2024/04/005>`_, Section 3 and Appendix B.

3. Add the geometry adjustment cases
------------------------------------

**Disk-gap fitting already exists in the existing 1D analysis pipeline.** The
SBP extension controls which geometry coordinates are free, adds thread-z
adjustments, and shares the geometry across the entire 20–22 GHz band.

Let :math:`g=(d_1,d_2,d_3,z_0,z_1,z_2,z_3)`, where d denotes clear air gaps and
z denotes thread positions relative to the mirror. These start from the measured
nominal geometry in :doc:`index`.

.. list-table:: Four independent geometry cases
   :header-rows: 1
   :widths: 20 60 20

   * - Notebook value
     - Free geometry
     - Number
   * - ``fixed``
     - None
     - 0
   * - ``gaps``
     - Three disk gaps
     - 3
   * - ``positions``
     - Four thread z coordinates, T0–T3
     - 4
   * - ``both``
     - Disk gaps and thread z coordinates
     - 7

In the existing notebook, the gaps are named ``d_v_1``–``d_v_3``. In the current
Python adapter they are supplied as ``ModelConfig.gap_widths_m``; thread positions
are supplied as ``z_from_mirror_m``. Thus changing a gap updates the disk surfaces,
while changing a thread z updates where its measured integral is compared with
the model. Disk and thread tilts are outside this 1D extension.

For each trial g, the 1D response calculation fits its frequency-dependent
amplitude and returns predicted measurement magnitudes :math:`\widehat y_{j\nu}(g)`.
The geometry driver minimizes

.. math::

   \mathcal L_{\mathrm{geom}}(g)=\frac1{N_\nu}\sum_\nu
     \frac{\sum_{j=0}^3[\widehat y_{j\nu}(g)-y_{j\nu}]^2}
          {\sum_{j=0}^3y_{j\nu}^2}
     +\lambda\sum_{k\in\mathrm{free}}
       \left(\frac{g_k-g_{k0}}{b_k}\right)^2.

Bounded least squares starts at the nominal geometry. Defaults are
:math:`|g_k-g_{k0}|\leq b_k=0.5` mm and :math:`\lambda=0.001`.
The penalty expresses a preference for measured positions, not an uncertainty.
Zero-signal frequencies do not constrain geometry. Clearance checks keep every
whole bead in its original vacuum region over the allowed geometry range.

The final geometry is evaluated to obtain the returned fields and boost curve.
The amplitude, field-integration and boost-conversion mathematics remain inside
the black box. Only spider measurements and nominal setup inputs enter this fit;
the full bead-pull reference is used for subsequent comparisons.

4. Existing-pipeline outputs and SBP adaptations
------------------------------------------------

The existing pipeline rejects frequency outliers, runs the Monte Carlo amplitude
fits, integrates the model fields and returns boost values and their saved
uncertainties. These calculations stay inside the black box. Defaults inherited
from the notebook are **100 draws**, a **5% assumed ET error**, and its bead and
material input uncertainties.

Only the adapters needed for the measured spider scan are changed:

* The input is our four-thread ET dataset, rather than a sweep of reduced-grid
  configurations. No full-reference measurements enter the spider fit.
* The shared geometry optimization in step 3 replaces the notebook's
  per-frequency disk-gap optimization. At the fitted geometry, the original
  Monte Carlo routine fits the amplitude for each retained frequency.
* The inapplicable too-few-z prior fallback is omitted. Full-grid configuration
  ranking is also omitted; the full reference is compared after fitting.

The supplied input uncertainties are propagated by the existing Monte Carlo
routine. The shared geometry optimizer does not provide an additional fitted
covariance; its bounds and penalty are not uncertainty estimates.

The numeric outputs keep the retained frequencies, fitted parameters, fields,
boost values/errors, Monte Carlo draws and rejected-frequency masks. Plots match
frequencies explicitly after rejection. Their labels remain editable in the
notebook, and no normalized shape-comparison plots are generated.

Current measurement run
-----------------------

Run ``run_20261025_141003`` was reprocessed on 4 November 2026, using all
401 acquired frequencies in 20–22 GHz, T0–T3 and 100 Monte Carlo draws.
The existing outlier filter rejected no frequencies. The four fits ran in
parallel on ``max-itt002``; numerical analysis took 157 s including startup.

.. list-table:: Results at 20.825 GHz
   :header-rows: 1
   :widths: 35 35 30

   * - Geometry case
     - :math:`\beta^2` with propagated uncertainty
     - Relative magnitude residual
   * - Fixed
     - :math:`294\pm167`
     - 21.2%
   * - Disk gaps
     - :math:`239\pm208`
     - 22.8%
   * - Thread z
     - :math:`319\pm183`
     - 21.2%
   * - Both
     - :math:`256\pm221`
     - 23.3%

These are conditional uncertainties from the existing pipeline's input priors,
not an added geometry-fit covariance. The geometry minimizes the full-band
objective, rather than the residual at this one frequency. Several fitted
coordinates reach their allowed 0.5 mm shifts; the residuals and broad
uncertainties remain substantial. Exact shifts are saved in
:codefile:`config/analysis_1d_provenance.json` and shown in the notebook.
