"""Finite alumina-sphere response, preserving the original Mie convention."""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

C = 299792458.0


def jn(n, x):
    return np.cos(x) / x if n == -1 else spherical_jn(n, x)


def hn(n, x):
    return jn(n, x) + 1j * spherical_yn(n, x)


def bead_factors(freq, radius=0.00293 / 2, eps=9.23, order=6):
    """Port of MADBead bead_finite_size.jl, without Julia dependencies."""
    x = 2 * np.pi * np.asarray(freq) / C * radius
    m = np.sqrt(eps)
    delta, deltac = 0j * x, 0j * x
    for n in range(1, order + 1):
        a = (m * m + 2) / (2 * x * (m * m - 1))
        kn = (
            (2 * n + 1)
            * a
            * (jn(n - 1, x) * jn(n, m * x) - m * jn(n, x) * jn(n - 1, m * x))
        )
        ln = a * (
            (n + 1)
            * (jn(n - 2, x) * jn(n - 1, m * x) - m * jn(n - 1, x) * jn(n - 2, m * x))
            + n * (jn(n, x) * jn(n + 1, m * x) - m * jn(n + 1, x) * jn(n, m * x))
        )
        cn = (
            hn(n, x) * (-jn(n - 1, x) + jn(n + 1, x))
            + jn(n, x) * (hn(n - 1, x) - hn(n + 1, x))
        ) / (
            m * (-jn(n - 1, m * x) + jn(n + 1, m * x)) * hn(n, x)
            + jn(n, m * x) * (hn(n - 1, x) - hn(n + 1, x))
        )
        dn = (
            m
            * x
            * (
                (-jn(n - 1, x) + jn(n + 1, x)) * hn(n, x)
                + jn(n, x) * (hn(n - 1, x) - hn(n + 1, x))
            )
            / (
                m * x * (-jn(n - 1, m * x) + jn(n + 1, m * x)) * hn(n, x)
                + jn(n, m * x)
                * (
                    m * m * x * hn(n - 1, x)
                    + (m * m - 1) * hn(n, x)
                    - m * m * x * hn(n + 1, x)
                )
            )
        )
        delta += (-1) ** n * (kn * cn - ln * dn)
        deltac += kn * cn + ln * dn
    return delta, deltac
