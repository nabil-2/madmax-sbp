Full bead-pull comparison
=========================

The figures below are existing notebook exports, copied unchanged. Saved
absolute normalizations differ; they have not been rescaled to agree.
Click any plot to view it at full resolution.

Full bead pull vs 1D
--------------------

The spider result uses the existing 1D analysis pipeline after restricting the
raw scan to 20–22 GHz. The full bead-pull curves retain their saved values and
do not enter the spider reconstruction or fit.

.. Exported by sbp_1d_analysis.ipynb from
   plots/1d/full_bead_comparison/both/24_field_comparison_resonance.png

.. figure:: /_static/figures/full_reference/1d_field_absolute.png
   :alt: Full and spider 1D transverse field integrals at 20.820 GHz in their saved absolute units.
   :width: 100%
   :target: ../_static/figures/full_reference/1d_field_absolute.png

   **Electric field at 20.820 GHz.** Top: measured transverse field integrals
   and finite-bead model fits. Bottom: model field integrals without the bead.
   Both panels retain saved absolute values. The spider fit adjusts disk gaps
   and thread z positions.

.. Exported by sbp_1d_analysis.ipynb from
   plots/1d/full_bead_comparison/both/22_boost_comparison.png

.. figure:: /_static/figures/full_reference/1d_boost_both.png
   :alt: Full bead-pull boost and the spider 1D fit with both disk gaps and thread z positions adjusted, from 20 to 22 GHz.
   :width: 100%
   :target: ../_static/figures/full_reference/1d_boost_both.png

   **Boost factor.** Full bead pull against the spider 1D fit with both disk
   gaps and thread z positions adjusted. Shading shows the saved full-scan
   one-standard-deviation uncertainty and the spider Monte Carlo uncertainty
   from the existing 1D analysis pipeline (100 draws).
   The axes span 20–22 GHz, with the full reference covering 20.5–21.5 GHz.

Full bead pull vs 3D
--------------------

The spider curves use measured S22 without the former additive template
correction. Scale, backgrounds and conditional bands are reevaluated at the
saved 3D geometry; the geometry has not been reoptimized after that removal.

.. Exported by sbp_3d_analysis.ipynb from
   plots/3d/full_tilt/full_bead_comparison/fields_full_reference_20p825.png

.. figure:: /_static/figures/full_reference/3d_field.png
   :alt: Full-reference model field integral versus full-tilt spider 3D field integrals at the T1 to T3 midpoint sections, at 20.825 GHz.
   :width: 100%
   :target: ../_static/figures/full_reference/3d_field.png

   **Electric field at 20.825 GHz.** The full-reference curve uses its saved
   1D fit. Spider points integrate the fitted full-tilt 3D field over horizontal
   sections at the T1–T3 midpoints where the full aperture lies in air. Both
   show model field integrals without the bead, rather than raw measurements.

.. Exported by sbp_3d_analysis.ipynb from
   plots/3d/full_tilt/full_bead_comparison/boost_full_reference.png

.. figure:: /_static/figures/full_reference/3d_boost.png
   :alt: Full bead-pull boost and full-tilt spider 3D boost, with uncertainty bands, on axes from 20 to 22 GHz.
   :width: 100%
   :target: ../_static/figures/full_reference/3d_boost.png

   **Boost factor.** Full bead pull against the full-tilt spider 3D fit.
   Shading shows the saved full-scan one-standard-deviation uncertainty and
   the conditional spider 16–84% band. The full reference covers
   20.5–21.5 GHz; the axes span 20–22 GHz.
