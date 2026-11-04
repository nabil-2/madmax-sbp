3D analysis
===========

**Measured complex chords + independent beam measurement → geometry fit →
receive-coupled boost → conditional uncertainty.**

For each trial geometry, a scalar wave model supplies the local inward and
outward electric fields along the measured bead paths.

1. Select the measurement band
------------------------------

Let :math:`\Gamma_{ji}(\nu)` denote the measured complex reflection at point
i on thread j. Retain frequencies in

.. math::

   \mathcal F=\{\nu:20\leq\nu/\mathrm{GHz}\leq22\}.

2. Select the geometry-fit frequencies
--------------------------------------

Preliminary fits to each measured chord use dimensionless
:math:`u=s/(0.1\,\mathrm m)` and

.. math::

   \Gamma(u,\nu)=b_0+b_1u+A\exp\{[-2/w^2+i\eta](u-u_c)^2\}.

For each trial :math:`(u_c,w,\eta)`, solve :math:`(b_0,b_1,A)` by complex least
squares, then optimize the real/imaginary residuals. Bounds are
:math:`u_c\in[-1.1,1.1]`, :math:`w\in[0.2,2.2]`, :math:`\eta\in[-8,8]`.
Accept a converged profile fit if no parameter is within
:math:`10^{-3}` of its bounds, and :math:`\mathrm{RMS}(\Gamma-\widehat\Gamma)/|A|<0.20`.

Define :math:`\mathcal F_{\mathrm{fit}}\subseteq\mathcal F` as the frequencies
whose profile fits are accepted on all three interior threads **T1–T3**.
T0's profile quality does not enter this frequency-selection rule.
At every selected frequency, the geometry-fit objective in Step 6 includes
the measured complex reflection from **all four threads, T0–T3**. Thus T0
constrains the geometry even when its preliminary profile fit is rejected.

3. Supply the independent incident beam
---------------------------------------

Interpolate the six independent transmission-beam (:math:`S_{31}`) Gaussian
magnitude profiles within :math:`\mathcal F`. At each :math:`\nu`:

.. math::

   g(\mathbf r)=\sum_{\ell=1}^{6}A_\ell
       e^{-|\mathbf r-\mathbf r_\ell|^2/w_\ell^2},\qquad
   N=\sqrt{\int_{\mathbb R^2}|g(\mathbf r)|^2\,\mathrm d^2\mathbf r}.

Set :math:`E_{\mathrm{in}}=\sqrt{2Z_0P_{\mathrm{in}}}\,g/N`, with
:math:`P_{\mathrm{in}}=1\,\mathrm W` over the whole transverse plane.
Here :math:`Z_0=\sqrt{\mu_0/\epsilon_0}\simeq376.7303\,\Omega` is the
vacuum wave impedance.
Assume flat phase at the outer reference face and hold the beam parameters
fixed during the geometry fit.

4. Parameterize disks and actual bead paths
-------------------------------------------

Use transverse coordinates :math:`x,y` and axial coordinate :math:`z`, with
:math:`z=0` at the mirror and positive toward the antenna. Index :math:`d`
labels the three disks, :math:`j` the four threads and :math:`i` a scan point.

.. math::

   z_{d,\mathrm{near}}(x,y)=z_{d0}+\Delta z_d+s_{dx}x+s_{dy}y,\qquad
   z_{d,\mathrm{far}}=z_{d,\mathrm{near}}+1\,\mathrm{mm},

Here :math:`z_{d0}` is the nominal near-face position at :math:`x=y=0`,
:math:`\Delta z_d` its fitted axial shift, and :math:`s_{dx},s_{dy}` the
dimensionless face slopes along x and y (approximately the tilt angles in
radians). Near/far denote the faces closer to/farther from the mirror.

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

Central gap changes are bounded by ±2 mm, slope components by ±5 mrad and
endpoint shifts by ±2 mm. Continuous-path constraints require at least one
bead radius of normal clearance from neighboring face planes and the mirror.
Face-plane checks cover the chord clipped to radius :math:`R+r_b`, where
:math:`R=150\,\mathrm{mm}` is the disk radius and
:math:`r_b=1.465\,\mathrm{mm}` the bead radius; they are conservative near
the finite disk rim.

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

Minimize :math:`\mathcal L` over geometries satisfying the constraints in
Step 4. Frequencies have equal normalized weight; samples within a frequency
have equal absolute weight. Individual-thread residuals use the same
:math:`c_\nu`.

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

Hold geometry, beam, materials and calibration fixed. Demean the complex
residuals :math:`e_j=\Gamma_j-\widehat\Gamma_j` per thread/frequency and resample circular
blocks of eight adjacent chord samples. The same sampled indices are used
across frequencies. Each draw updates only the common scale:

.. math::

   c_{\nu}^{(b)}=c_{\nu}+
      \frac{\sum_j(P_jt_{j\nu})^\dagger e_{j\nu}^{(b)}}
           {\sum_j\|P_jt_{j\nu}\|^2},\qquad
   \beta^{2(b)}_{\nu}=|c_{\nu}^{(b)}|\,\beta^2_{\mathrm{recv},g,\nu}.

Repeat for 1,000 draws and take pointwise quantiles of
:math:`\beta^{2(b)}_\nu`; the reported interval is 16–84%.
These conditional intervals exclude uncertainty in geometry, beam, materials,
absolute calibration and model discrepancy.
