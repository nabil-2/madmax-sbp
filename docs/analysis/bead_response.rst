Bead response
=============

The bead calibration and quadratic response are applied by this repository,
outside the BoostFractor black box. Code: :codefile:`src/bead_response.py`
and ``bead_factors`` in :codefile:`src/fit_1D.py`.

Static conversion and directional response
------------------------------------------

The volume polarizability and 1-W reflection coefficient are

.. math::

   \alpha_0=4\pi r_b^3\frac{\epsilon_b-1}{\epsilon_b+2},\qquad
   \kappa_b=\frac{i\omega\epsilon_0\alpha_0}{4P_{\mathrm{in}}}.

For :math:`r_b=1.465` mm and :math:`\epsilon_b=9.23`, two directional field
components give the finite-size quadratic

.. math::

   Q(e_+,e_-)=\delta_e(e_+^2+e_-^2)+2\delta_c e_+e_-.

The measured magnitude proxy is :math:`\sqrt{|Q|}`. The field without the bead
is :math:`e_++e_-`; replacing Q by that field squared would discard the finite
bead response. The 1D fit compares magnitudes; the 3D fit uses the full complex Q
with its stated phasor conversion.

Six-term finite-sphere calculation
----------------------------------

These are the implemented Mie-series formulas, not BoostFractor internals.
Let :math:`x=\omega r_b/c_0`, :math:`m=\sqrt{\epsilon_b}`,
:math:`h_n=j_n+i y_n` be outgoing spherical Hankel functions, and
:math:`j_{-1}(x)=\cos(x)/x`. Define

.. math::

   a=\frac{m^2+2}{2x(m^2-1)},\qquad
   K_n=(2n+1)a[j_{n-1}(x)j_n(mx)-m j_n(x)j_{n-1}(mx)],

.. math::

   L_n=a\{(n+1)[j_{n-2}(x)j_{n-1}(mx)-m j_{n-1}(x)j_{n-2}(mx)]
       +n[j_n(x)j_{n+1}(mx)-m j_{n+1}(x)j_n(mx)]\}.

For compactness write
:math:`J_n(t)=-j_{n-1}(t)+j_{n+1}(t)`,
:math:`H_n(t)=h_{n-1}(t)-h_{n+1}(t)` and
:math:`N_n=J_n(x)h_n(x)+j_n(x)H_n(x)`. Then

.. math::

   c_n=\frac{N_n}{mJ_n(mx)h_n(x)+j_n(mx)H_n(x)},

.. math::

   d_n=\frac{mxN_n}{mxJ_n(mx)h_n(x)+j_n(mx)
               [m^2xH_n(x)+(m^2-1)h_n(x)]},

.. math::

   \delta_e=\sum_{n=1}^{6}(-1)^n(K_nc_n-L_nd_n),\qquad
   \delta_c=\sum_{n=1}^{6}(K_nc_n+L_nd_n).

Verification and raw transverse reconstruction use the static conversion
:math:`E_{\mathrm{conv}}=\sqrt{\Delta\Gamma/\kappa_b}` with continuous branches.
They do not independently separate the two directional waves. That separation
comes from the longitudinal or 3D field model.
