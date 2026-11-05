3D analysis
===========

**Measured complex chords + independent beam measurement → geometry fit →
receive-coupled boost → conditional uncertainty.**

For each trial geometry, a scalar wave model supplies the local inward and
outward electric fields along the measured bead paths. The current forward solver
uses BoostFractor propagation operators with Python interface and field
reconstruction code. Equivalence to the native library for the full tilted,
measured-surface case has not been established.

1. Select the measurement band
------------------------------

Let :math:`\Gamma_{ji}(\nu)` denote the measured complex reflection at point
i on thread j. Retain frequencies in

.. math::

   \mathcal F=\{\nu:20\leq\nu/\mathrm{GHz}\leq22\}.

2. Fix the endpoint background and select geometry-fit frequencies
------------------------------------------------------------------

For each thread, sort the retained valid samples by their measured position.
Let :math:`\mathcal R_j` contain exactly the first five and last five samples.
These ten off-beam measurements alone determine the complex background:

.. math::

   X_j=[\mathbf 1,\mathbf u_j],\qquad
   u_{ji}=s_{ji}/(0.1\,\mathrm m),\qquad
   b_{j\nu}=(X_j[\mathcal R_j])^+\Gamma_{j\nu}[\mathcal R_j].

The signed coordinate :math:`s` is measured from the recorded scan midpoint.
The two complex coefficients are an offset and a slope. Interpolate the fitted
straight line to every measured point, and subtract it:

.. math::

   \Gamma_{\mathrm{bg},j\nu}=X_jb_{j\nu},\qquad
   \Delta\Gamma_{j\nu}=\Gamma_{j\nu}-\Gamma_{\mathrm{bg},j\nu}.

Hold this background fixed for every subsequent profile and geometry fit.
No illuminated measurement, predicted field or model residual enters its
estimation. Retain the measured residual noise at the reference points.

Frequency selection retains the original, separate Gaussian diagnostic on the
raw reflection:

.. math::

   \Gamma(u,\nu)\approx q_0+q_1u+
       A\exp\{[-2/w^2+i\eta](u-u_c)^2\}.

For each trial centre :math:`u_c`, width :math:`w` and curvature :math:`\eta`,
solve the complex diagnostic coefficients :math:`q_0,q_1,A` by least squares.
These nuisance terms are used only to score profile quality; they are **not**
the physical background, are never subtracted from the geometry-fit data and
do not replace the endpoint-only background above.
Bounds remain :math:`u_c\in[-1.1,1.1]`, :math:`w\in[0.2,2.2]`,
:math:`\eta\in[-8,8]`. Accept a converged profile fit if no parameter is within
:math:`10^{-3}` of its bounds and the residual RMS divided by :math:`|A|`
is below 0.20.

The geometry-fit frequencies remain those accepted on all three interior
threads **T1–T3**. T0 quality does not select frequencies, but measurements
from **all four threads** enter the geometry fit at every selected frequency.
The original 35-bin frequency selection is retained. It is calculated
independently of the fixed endpoint background used in the physical fit.

3. Supply the independent incident beam
---------------------------------------

Interpolate the six independent transmission-beam (:math:`S_{31}`) Gaussian
magnitude profiles within :math:`\mathcal F`. Centre each fitted Gaussian on
the common disk axis, :math:`x=y=0`, retaining its measured amplitude and
width. Combine these centred profiles at the outer reference face. At each
:math:`\nu`:

.. math::

   g(\mathbf r)=\sum_{\ell=1}^{6}A_\ell
       e^{-|\mathbf r|^2/w_\ell^2},\qquad
   N=\sqrt{\int_{\mathbb R^2}|g(\mathbf r)|^2\,\mathrm d^2\mathbf r}.

Set :math:`E_{\mathrm{in}}=\sqrt{2Z_0P_{\mathrm{in}}}\,g/N`, with
:math:`P_{\mathrm{in}}=1\,\mathrm W` over the whole transverse plane.
Here :math:`Z_0=\sqrt{\mu_0/\epsilon_0}\simeq376.7303\,\Omega` is the
vacuum wave impedance.
Assume flat phase at the outer reference face and hold the beam parameters
fixed during the geometry fit. This prescribes a centred beam; its alignment
is not fitted to the bead measurements.

4. Include measured disk shapes and parameterize the geometry
-------------------------------------------------------------

Use transverse coordinates :math:`x,y` and axial coordinate :math:`z`, with
:math:`z=0` at the mirror and positive toward the antenna. Index :math:`d`
labels the three disks, :math:`j` the four threads and :math:`i` a scan point.

.. math::

   z_{d,\mathrm{near}}(x,y)
      =z_{d0}+\Delta z_d+s_{dx}x+s_{dy}y+h_{d,\mathrm{near}}(x,y),

.. math::

   z_{d,\mathrm{far}}(x,y)
      =z_{d0}+\Delta z_d+1\,\mathrm{mm}
       +s_{dx}x+s_{dy}y+h_{d,\mathrm{far}}(x,y).

Here :math:`z_{d0}` is the nominal near-face reference-plane position,
:math:`\Delta z_d` its fitted axial shift, and :math:`s_{dx},s_{dy}` the
dimensionless rigid slopes along x and y (approximately the tilt angles in
radians). Near/far denote the faces closer to/farther from the mirror.

The fixed height functions :math:`h_{d,\mathrm{near}},h_{d,\mathrm{far}}`
are bilinear interpolations of the measured surfaces in the same x/y frame.
For D1–D3 counted from the mirror, use measured disks 2, 3, 1 respectively.
Face A faces the mirror and face B the antenna; with measured heights
:math:`A_d,B_d`, the signed departures are
:math:`h_{d,\mathrm{near}}=-A_d` and :math:`h_{d,\mathrm{far}}=B_d`.
The measurements have 2.5 mm spacing; their heights are converted from
millimetres to metres. Their offsets and shapes are retained. Thus the local
thickness is :math:`1\,\mathrm{mm}+A_d(x,y)+B_d(x,y)`.
Measured shapes stay fixed during the fit and enter the interface propagation
phases together with the rigid slopes.

.. math::

   v_i=\frac{s_i-s_{\min}}{s_{\max}-s_{\min}},\qquad
   z_{ji}=(1-v_i)(z_{j0}+\Delta z_{jA})+
                    v_i(z_{j0}+\Delta z_{jB}).

The signed distance :math:`s_i` runs along each thread's scan, with retained
limits :math:`s_{\min},s_{\max}`; :math:`v_i` is its fractional position
from endpoint A (0) to B (1). The nominal thread height is :math:`z_{j0}`,
and :math:`\Delta z_{jA},\Delta z_{jB}` are its independently fitted endpoint
shifts, giving the bead's axial position :math:`z_{ji}`.

Recorded x/y stay fixed. A/B are the retained scan endpoints, not mechanical
supports; paths are straight, without sag. There are **17** shared parameters:
three disk shifts, six disk slopes and eight endpoint shifts.

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
The conjugations convert between the field, Mie and recorded-S22 phasor
conventions; :math:`\epsilon_0` is explicit because this :math:`\alpha_0`
has units of volume. Step 6 fits the common complex scale and adds the affine
backgrounds to compare with measured S22.

Apply the same disk-radius signal aperture to **all four threads**, including
T0: the simulated bead-induced response is zero for
:math:`x_{ji}^2+y_{ji}^2>R^2`, with :math:`R=150\,\mathrm{mm}`.
The propagation inside this aperture remains appropriate to each thread's
location. This is a modelling assumption; physical exterior fields may exist.
All retained measured points, including those outside the aperture, remain in
the least-squares objective. Only the ten designated endpoint samples enter
the background estimate.
The horizontal electric-field and predicted-reflection maps show this same
circular aperture for T0–T3.

6. Fit the geometry, backgrounds and common complex scale
---------------------------------------------------------

At a fixed trial geometry and frequency :math:`\nu`, collect each thread's
:math:`n_j` measured values and simulated bead responses from Step 5 into
vectors :math:`\Gamma_{j\nu},t_{j\nu}\in\mathbb C^{n_j}`. The same scalar
positions :math:`u_{ji}` as in Step 2 form the vector :math:`\mathbf u_j`
and the :math:`n_j\times2` background matrix

.. math::

   X_j=[\mathbf1,\mathbf u_j]
      =\begin{pmatrix}
         1 & u_{j1}\\
         \vdots & \vdots\\
         1 & u_{jn_j}
       \end{pmatrix},\qquad
   b_{j\nu}=\begin{pmatrix}b_{0,j\nu}\\b_{1,j\nu}\end{pmatrix}.

Thus :math:`X_jb_{j\nu}` is a constant plus a linear trend along the thread.
Fit this background using only the first and last five valid points on each
thread, as in Step 2, and hold it fixed. Fit one common complex scale
:math:`c_\nu` by minimizing

.. math::

   S_\nu(c_\nu)
      =\sum_j\|\Gamma_{j\nu}-c_\nu t_{j\nu}-X_jb_{j\nu}\|^2.

All sums over j include **T0–T3**. Let :math:`E_j\in\mathbb R^{10\times n_j}`
select the first five and last five valid samples. Linear least squares on
these ten off-beam measurements gives

.. math::

   b_{j\nu}=(E_jX_j)^+E_j\Gamma_{j\nu},\qquad
   P_j=I-X_j(E_jX_j)^+E_j,

where + denotes the pseudoinverse and :math:`I` the :math:`n_j\times n_j`
identity matrix. The operator :math:`P_j` subtracts the affine background
estimated exclusively from the endpoints. It does not fit the background to
the illuminated samples or to the model residual. For these scans, the
reference points lie outside the disk signal aperture, so
:math:`E_jt_{j\nu}=0` and :math:`P_jt_{j\nu}=t_{j\nu}`. This leaves the
reduced objective

.. math::

   S_\nu(c_\nu)=\sum_j\|P_j\Gamma_{j\nu}-c_\nu P_jt_{j\nu}\|^2.

Minimizing with respect to the complex scale gives the normal equation and
its solution:

.. math::

   \sum_j(P_jt_{j\nu})^\dagger
      (P_j\Gamma_{j\nu}-c_\nu P_jt_{j\nu})=0,

.. math::

   c_\nu=\frac{\sum_j(P_jt_{j\nu})^\dagger(P_j\Gamma_{j\nu})}
                  {\sum_j\|P_jt_{j\nu}\|^2}.

Here :math:`\dagger` denotes the conjugate transpose. Combining this scale
with the fixed endpoint background gives the fitted response and the
baseline-subtracted signal:

.. math::

   \widehat\Gamma_{j\nu}=c_\nu t_{j\nu}+X_jb_{j\nu},\qquad
   \Delta\Gamma_{j\nu}=\Gamma_{j\nu}-X_jb_{j\nu},\qquad
   \widehat{\Delta\Gamma}_{j\nu}=c_\nu t_{j\nu}.

For every trial geometry, recalculate the scale with this analytical solution
while keeping the endpoint backgrounds fixed. Choose the geometry by
minimizing the normalized residual averaged over the selected frequencies:

.. math::

   r_{\nu}^2=\frac{\sum_j\|c_{\nu}P_jt_{j\nu}-P_j\Gamma_{j\nu}\|^2}
                    {\sum_j\|P_j\Gamma_{j\nu}\|^2},\qquad
   \mathcal L=\frac1{|\mathcal F_{\mathrm{fit}}|}
                     \sum_{\nu\in\mathcal F_{\mathrm{fit}}}r_{\nu}^2.

Frequencies have equal normalized weight; samples within a frequency have
equal absolute weight. Individual-thread residuals use the same :math:`c_\nu`.

Minimize :math:`\mathcal L` subject to the following bounds and clearance
constraints. Reference-plane gap changes are bounded by ±2 mm, rigid slope
components by ±5 mrad and endpoint shifts by ±2 mm. Continuous-path
constraints require one bead radius of clearance from the measured curved
faces and the mirror. A bilinear face restricted to a straight chord is
quadratic within each measurement cell. Bounds on its three quadratic
Bernstein coefficients and on the surface gradient guarantee clearance
throughout that cell, including between scan points. Apply these conservative
constraints to the chord clipped to radius :math:`R+r_b`, where
:math:`R=150\,\mathrm{mm}` is the disk radius and
:math:`r_b=1.465\,\mathrm{mm}` the bead radius. The finite rim is conservatively
extended for this check.

7. Calculate boost and diagnostic fields
----------------------------------------

The 3D booster model computes the receive-coupled geometry response
:math:`\beta^2_{\mathrm{recv},g}` for the fixed 1-W beam. Since the reflection
template is quadratic in field, its fitted amplitude factor is
:math:`\sqrt{|c_{\nu}|}` and its power factor is :math:`|c_{\nu}|`:

.. math::

   \boxed{\beta^2(\nu)=|c_{\nu}|\,\beta^2_{\mathrm{recv},g}(\nu)},\qquad
   |E|=\sqrt{|c_{\nu}|}\,|e_++e_-|.

The corresponding finite-bead field proxy is :math:`\sqrt{|c_{\nu}Q|}`.

8. Form conditional uncertainty bands
-------------------------------------

Hold geometry, measured surface shapes, beam, materials and calibration fixed.
Demean the complex residuals per thread/frequency and resample circular blocks
of eight adjacent chord samples. Use the same sampled indices across
frequencies. For each of 1,000 draws, re-estimate the background using only the
same first/last-five reference points, then update the common response scale.

Equivalently, for resampled residual :math:`e_j^{(b)}`, remove the baseline
perturbation inferred exclusively from the reference points:

.. math::

   \delta d_j^{(b)}=e_j^{(b)}
      -X_j(X_j[\mathcal R_j])^+e_j^{(b)}[\mathcal R_j],

.. math::

   c_\nu^{(b)}=c_\nu+
      \frac{\sum_j t_{j\nu}^{\dagger}\delta d_{j\nu}^{(b)}}
           {\sum_j\|t_{j\nu}\|^2},\qquad
   \beta^{2(b)}_\nu=|c_\nu^{(b)}|\,\beta^2_{\mathrm{recv},g,\nu}.

Take pointwise quantiles of :math:`\beta^{2(b)}_\nu`; the reported interval is
16–84%. These conditional intervals include residual-induced variation of the
endpoint background and response scale. They exclude uncertainty in geometry,
measured surface shapes, beam, materials, absolute calibration and model assumptions.
