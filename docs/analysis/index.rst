Analysis
========

The two fits use different observables. Neither fits a freely chosen boost
spectrum: :math:`\beta^2` is calculated from the fitted field/model response.
For 1D, the existing 1D analysis pipeline is treated as a black box covering
longitudinal fitting, field integration and boost calculation; the 1D page
documents the SBP input preparation and geometry extension. Its backend uses
the original notebook/MADBead routines with the necessary SBP adapters.
The 3D booster model and the bead response are also
treated as black boxes.

.. list-table:: Choose a pipeline
   :header-rows: 1
   :widths: 20 40 40

   * -
     - 1D
     - Full-tilt 3D
   * - Measurement entering the fit
     - Magnitudes of reconstructed transverse integrals :math:`|E_T(z,\nu)|`
     - Measured complex :math:`\Gamma` at the actual chord points
   * - Transverse assumption
     - Radial continuation of each measured diameter
     - Scalar diffraction model with an independent measured input beam
   * - Geometry
     - Fixed, gaps, thread z, or both
     - Disk z/slopes and two z endpoints per thread

.. toctree::
   :maxdepth: 1

   one_dimensional
   three_dimensional
   full_reference

Coordinates and defaults
------------------------

The analysis coordinate :math:`z=0` is the mirror; positive z points towards the
antenna. Positions below are nominal **near-face reference positions**, not
disk centres. The 3D analysis adds the measured surface departures to these
reference planes.

.. list-table:: Corrected nominal geometry
   :header-rows: 1
   :widths: 45 55

   * - Quantity
     - Value
   * - Mirror–D1, D1–D2, D2–D3 clear gaps
     - 7.803, 8.853, 8.855 mm
   * - Disk near faces D1–D3
     - 7.803, 17.656, 27.511 mm
   * - Thread z, T0–T3
     - 32.011, 23.0835, 13.2295, 3.9015 mm
   * - Disk thickness / radius
     - 1 mm / 150 mm
   * - Disk permittivity / loss tangent
     - 9.3 / :math:`10^{-5}`
   * - Alumina bead diameter / permittivity
     - 2.93 mm / 9.23

The scan logs supply measured **x/y**; their original z column is not a registered
booster coordinate. The supplied geometry determines nominal analysis z.
Both pipelines discard raw frequencies outside 20–22 GHz before preprocessing.
The saved 1D results and rendered figures use the existing 1D analysis pipeline,
with the frequency cut applied before transverse reconstruction, and 100 Monte
Carlo draws. The 3D results are unchanged by this 1D rerun. Out-of-band saved
inputs are rejected by both updated pipelines.

Units and plot exports
----------------------

Local scalar fields are in V/m; :math:`E_T=\int E\,dA` is in V m.
The reference incident-power convention is 1 W. A finite-bead field proxy is
distinct from the field without the bead.

The notebooks reproduce all plots and save PNG/PDF files to the ignored
``plots/1d/`` and ``plots/3d/full_tilt/`` trees. Titles, axes, legends, colorbars
and annotation templates are editable in their plot cells. The notebooks are
committed with the rendered figures; the exports and large data are not.


The :notebook:`sbp_3d_analysis.ipynb` also shows raw dB traces, the prescribed
incident-beam profile, profile-quality selection, complex-plane fits, fitted
backgrounds, field spectra on the sloped paths, and spatial residual maps.
Each plot cell embeds its PNG in the saved notebook and exports PNG/PDF files;
viewing these rendered outputs requires no running kernel.
