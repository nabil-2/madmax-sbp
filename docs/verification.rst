Verification
============

**Full grid → gated reflection → limited lines → Gaussian field → transverse
integral → comparison with the full grid.**

Notebook: :notebook:`sbp_verification.ipynb`.
Runner: :codefile:`src/sbp_verification.py`.
Numerical steps: :codefile:`src/verification_core.py`.

1. Read the grids and choose samples
------------------------------------

Each HDF5 file contains complex ``s11_map[x,y,z,frequency]`` and the
``vna_config`` / ``measurement_config`` JSON attributes. Grid coordinates are
converted from millimetres to metres. The default example selects nine bins
near 19.5 GHz from the original 18–24 GHz grids.

Gaps are ordered by acquisition z. The legacy plane-selection starts correspond
to smallest/largest/smallest/fifth-smallest **absolute z** for gaps 1–4;
absolute z runs opposite to the acquisition coordinate. The notebook prints the
actual selected indices. The full reference keeps every acquired z plane.

2. Time gate before selecting frequencies
-----------------------------------------

On the complete acquired frequency axis, form

.. math::

   s(t)=\operatorname{fftshift}\{\operatorname{IFFT}_f S(f)\},\qquad
   S_g(f)=\operatorname{FFT}_t\{\operatorname{ifftshift}[W(t)s(t)]\}.

For each z plane, find the largest central-grid time-domain magnitude, add
55 ns, and fit those times linearly in z. Correct the gate centre off axis:

.. math::

   t_c(x,y,z)=\frac{2}{c_0}
   \sqrt{[c_0 t_0(z)/2]^2+x^2+y^2}.

Writing :math:`d=|t-t_c|`, the gate has a flat core of width
:math:`w=180\,\mathrm{ns}` and cosine ramps of
:math:`e=0.2w=36\,\mathrm{ns}` on **each** side:

.. math::

   W=\begin{cases}
   1,&d\leq w/2,\\
   \tfrac12[1+\cos(\pi(d-w/2)/e)],&w/2<d\leq w/2+e,\\
   0,&d>w/2+e.
   \end{cases}

The requested frequency subset is taken after this transform. Disable
``TIME_GATING`` to bypass the gate.

3. Select lines and subtract the background
-------------------------------------------

Default line equations are :math:`y=ax+b`, with
:math:`(a,b)=(0.15,0.080),(-0.15,0.040),(0.15,0),(-0.15,-0.040)`;
b is in metres. Retain points satisfying :math:`|y-ax-b|\leq3\,\mathrm{mm}`.
One-line mode uses the central line, two-line mode the two middle lines, and
three/four-line modes the first three/four lines.

Sort samples along each line's unit direction. At every z and frequency, fit
a complex affine background to the first and last two samples:

.. math::

   (b_0,b_1)=\arg\min\sum_{i\in\mathrm{edges}}
   |S_{g,i}-b_0-b_1s_i|^2,\qquad
   \Delta\Gamma_i=S_{g,i}-b_0-b_1s_i.

For the full grid, the same operation uses the first/last two x samples of
each y row, with x replacing the line coordinate s.

4. Take the continuous square root and convert to field
-------------------------------------------------------

The code unwraps the principal square-root phase with period :math:`\pi`
along x then y for the full grid, or along the retained line coordinate:

.. math::

   q=\sqrt{|\Delta\Gamma|}\,
   \exp\{i\operatorname{unwrap}_{\pi}[\arg\sqrt{\Delta\Gamma}]\},
   \qquad
   E_{\mathrm{conv}}=
   \sqrt{\frac{4P_{\mathrm{in}}}{i\omega\epsilon_0\alpha_0}}\;q.

Here :math:`\omega=2\pi f`, :math:`P_{\mathrm{in}}=1\,\mathrm W`, and
:math:`\alpha_0=4\pi r_b^3(\epsilon_b-1)/(\epsilon_b+2)` uses
:math:`r_b=1.465\,\mathrm{mm}`, :math:`\epsilon_b=9.23`.
This is the finite-bead field estimate, without longitudinal Mie deconvolution.

5. Optional full-reference alignment
------------------------------------

With the default ``REFERENCE_ALIGNMENT=True``, let :math:`q_i` be a line's
square roots and :math:`F_i` the full-grid square roots at those points.
For each line, plane and frequency:

.. math::

   \sigma_i=\begin{cases}1&\operatorname{Re}(q_iF_i^*)\geq0\\-1&\text{otherwise},\end{cases}
   \quad v_i=\sigma_iq_i,\quad
   a=\frac{\sum_i v_i^*F_i}{\sum_i|v_i|^2},\quad
   q_i\leftarrow av_i.

Only finite values enter these sums. At overlapping line samples the last
selected line supplies the stored value. This mode **uses full-grid information**;
set the flag to false for a line-only reconstruction.

6. Fit the transverse field
---------------------------

The actual fit uses the fundamental circular Gaussian with quadratic phase:

.. math::

   G(x,y)=A\exp(-r^2/s^2)\exp[i(Kr^2+\phi)],\qquad
   r^2=(x-x_0)^2+(y-y_0)^2.

Higher Hermite weights in the general helper are fixed to zero in this workflow;
there is no Bessel-mode fit here. A complex scale is eliminated analytically,
with its magnitude capped at :math:`10^4`:

.. math::

   a=\frac{\sum_iG_i^*E_i}{\sum_i|G_i|^2},\qquad
   \mathcal L=\frac1N\sum_i|E_i-aG_i|^2+\mathcal R,

.. math::

   \mathcal R={}&5\!\times\!10^{-4}(K/800)^2
   +3\!\times\!10^{-3}\sum_{v=x,y}[(v_0-v_{\mathrm{mid}})/h_v]^2\\
   &+10^{-4}[(s-s_{\mathrm{mid}})/h_s]^2
   +10^{-4}(|a|/10^4)^2.

The centres are bounded by the sampled rectangle, :math:`8\leq s\leq120` mm,
:math:`|K|\leq800\,\mathrm{m^{-2}}`, and :math:`|\phi|\leq\pi`.
Midpoints and half-spans define :math:`v_{\mathrm{mid}},h_v,s_{\mathrm{mid}},h_s`.
L-BFGS-B fits each plane/frequency independently. The returned parameters absorb
:math:`|a|` into A and :math:`\arg a` into :math:`\phi`.

7. Integrate and compare
------------------------

.. math::

   E_T^{\mathrm{full}}=\sum_{x,y}E_{\mathrm{conv}}\,\Delta x\Delta y,
   \qquad
   E_T^{\mathrm{reduced}}=\int\!\int G_{\mathrm{fitted}}(x,y)\,dx\,dy.

The reduced integral uses a 50×50 trapezoidal grid over the original rectangular
extent; the full reference retains its original grid sum. Failed fits are masked.
Results have units V m and are matched by acquisition z before comparison:

.. math::

   \varepsilon_{\mathrm{complex}}=
   \frac{\|E_T^{\mathrm{reduced}}-E_T^{\mathrm{full}}\|_2}
        {\|E_T^{\mathrm{full}}\|_2},\qquad
   \varepsilon_{|E|}=\frac{\big||E_T^{\mathrm{reduced}}|-|E_T^{\mathrm{full}}|\big|}
                              {|E_T^{\mathrm{full}}|}.

The summary reports the complex error, median/90th-percentile magnitude errors
and optimizer success; the error plot evaluates the complex norm per frequency.
Invalid/zero denominators are excluded. No statistical uncertainty or boost fit
is produced by verification.

Output: ``outputs/sbp_verification/ET_results.npz`` with
``ETs_full``, ``ETs_reduced``, ``z_full``, ``z_reduced`` and ``frequencies``.
ET axes are ``(z, frequency)``; acquisition-relative z is not the mirror coordinate.

Diagnostic plots
----------------

The notebook includes the original sampling, per-line background/square-root,
reference-alignment, fitted/full complex-field, magnitude-residual, Gaussian
component, complex-plane integral and per-z frequency plots. Summary field,
integral and error plots are retained. Every figure has editable text arguments
and saves PNG/PDF under ``plots/verification/L3_Z1_G4/reference_assisted/`` for
the default configuration; line-only alignment uses a separate folder.

The three mode-check figures are illustrations of

.. math::

   G_w=G\sum_{p,q=0}^1\hat w_{pq}
       H_p((x-x_0)/s)H_q((y-y_0)/s),\qquad
   H_0=1,\quad H_1(v)=2v,\quad \|\hat w\|_2=1.

They do not add free modes to the measurement fit. The default three-line
notebook contains **19 rendered figures**, which are committed in the notebook;
the generated export folder remains ignored by Git.
