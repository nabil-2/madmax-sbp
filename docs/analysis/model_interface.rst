3D model interface
==================

**BoostFractor is treated as a black box.** This documentation does not derive
its basis functions, field propagation, diffraction, interfaces or emission
calculation. It describes the input/output contract needed by this
repository's measurement fit.

Inputs
------

.. list-table:: Supplied to the model adapter
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Meaning
   * - Frequency
     - The selected measured frequencies, in Hz
   * - Geometry
     - Three disk near-face positions, two slopes per disk, thickness and radius
   * - Materials
     - Disk permittivity/loss and the mirror assumption
   * - Illumination
     - The independently measured six-profile input beam, normalized to 1 W;
       its phase and plane registration are assumptions
   * - Evaluation coordinates
     - Actual measured x/y and fitted pointwise z along each thread
   * - Numerical settings
     - Propagation-bank grid, modal truncation and interpolation settings

The measured spider S22 samples are **not** supplied as an input illumination.
They remain the observations in the external fit. Beam construction, thread
geometry and the fit equations are given in :doc:`three_dimensional`.

Outputs
-------

* Directional receive fields at each bead position, normalized to the fixed
  incident beam. The external finite-bead operator converts these into a
  complex reflection template.
* Geometry-only receive-coupled power boost. The external fitted response
  scale turns this into the reported conditional boost.
* Total emitted power boost, a different observable from receive coupling.
* Field maps and numerical diagnostics, including reflected power and mirror
  termination residuals.

Implementation boundary
-----------------------

The bank is built by :codefile:`src/build_bank.jl` using BoostFractor.jl commit
``acf4f491fe633ab05c3b8abbafe6d6288e42aba0``. This repository's adapters are
:codefile:`src/receive_field.py`, :codefile:`src/tilted_disks.py` and
:codefile:`src/exterior_field.py`. They evaluate the scalar model for the fitted
disk surfaces and sloping paths; the implementation is not a vector Maxwell
solver. Their internal field-model mathematics is outside this documentation.

The fitting bank has 120 modes, a 128×128 grid and 33 selected frequencies.
Recomputing the dense spectrum requires a 401-frequency bank. CPU/CUDA are bank
construction options; Python performs the geometry optimization. Numerical
resolution changes are convergence checks, not statistical uncertainty estimates.
