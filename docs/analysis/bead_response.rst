Bead response
=============

The bead response is treated as a **black box**. Its theory and finite-size
corrections are described in the paper cited below; this documentation records
only the inputs, outputs and their use in the analysis.

Inputs and outputs
------------------

.. list-table:: Response interface
   :header-rows: 1
   :widths: 30 70

   * - Quantity
     - Description
   * - Inputs
     - Frequency, bead radius/permittivity and incident-power convention;
       directional model fields when evaluating the finite-size response
   * - Field conversion
     - A complex factor :math:`C_b(\nu)` converting a continuous square root of
       the measured reflection change into the finite-bead field estimate
   * - Model response
     - A complex quadratic response :math:`Q=\mathcal R_b(e_+,e_-;\nu)` and
       reflection conversion factor :math:`\kappa_b(\nu)` for the model template

The configured bead has radius 1.465 mm and relative permittivity 9.23; the
incident-power convention is 1 W. These are this repository's inputs.

Where it enters the pipelines
-----------------------------

* **Verification and raw 1D reconstruction:** apply
  :math:`E_{\mathrm{conv}}=C_b(\nu)\sqrt{\Delta\Gamma}` using continuous branches.
* **1D fit:** compare measured magnitudes with :math:`\sqrt{|Q|}`.
* **3D fit:** use the complex response in the reflection template, with the
  phasor/conjugation convention specified in :doc:`three_dimensional`.

The finite-bead estimate is distinct from the field without the bead,
:math:`e_++e_-`. The response routines do not determine the directional fields;
those are supplied by the field model.

Code: :codefile:`src/bead_response.py` and ``bead_factors`` in
:codefile:`src/fit_1D.py`; the raw-field conversion is in
:codefile:`src/verification_core.py` and :codefile:`src/sbp_analysis.py`.

Reference
---------

J. Egge et al., *Experimental determination of axion signal power of dish
antennas and dielectric haloscopes using the reciprocity approach*,
`JCAP 04 (2024) 005 <https://doi.org/10.1088/1475-7516/2024/04/005>`_.
Open-access version: `arXiv:2311.13359v2 <https://arxiv.org/abs/2311.13359v2>`_.
See **Section 3 and Appendix B** for the bead-pull relation and finite-sphere
response calculation.
