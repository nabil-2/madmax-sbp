# Measured disk surfaces

These six files are copied unchanged from `madmax-omc/data/raw/disk-measurements/`.
Their SHA-256 checksums are recorded in `manifest.json`. The user confirmed that
the spider assembly uses the same disk order and coordinate axes.

| Booster position from the mirror | Measured disk |
| --- | --- |
| D1 | 2 |
| D2 | 3 |
| D3 | 1 |

Each 401 × 401 map contains surface heights in millimetres at 2.5 mm spacing.
Row and column 200 (one-based) lie on the disk axis. Rows follow x, columns y.
Face A faces the mirror, and face B faces the antenna. In the analysis coordinate
z, increasing from the mirror towards the antenna, the measured departures are
`h_near = -A` and `h_far = +B`, matching the OMC convention.

The measured departures are added to the fitted reference-face positions and
rigid disk slopes. Neither the map mean nor a best-fit plane is removed. The
nominal reference-plane separation remains 1 mm; actual thickness varies with
the two measured faces. Shape maps are fixed during fitting.
