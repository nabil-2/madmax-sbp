

export calc_δ_e_mie,calc_δ_c_mie

"""
Bead factor δ_e^+ (Eq. 14.8 in J.Egge's Thesis) for bead
permittivity ϵ, bead radius R, at frequency f, and expansion order N.
"""
function calc_δ_e_mie(ϵ,R;f=22e9,N=3)
    c_const = 299792458
    k = 2π*f/c_const
    x = k*R
    m = sqrt(ϵ)
    δ_e = 0.0 + 0.0im
    for n in 1:1:N
       δ_e += (-1)^n * (Kn(n,x,m)*cn(n,x,m) - Ln(n,x,m)*dn(n,x,m))
    end
    return δ_e
end
"""
Bead factor δ_c^+ (Eq. 14.11 in J.Egge's Thesis) for bead
permittivity ϵ, bead radius R, at frequency f, and expansion order N.
"""
function calc_δ_c_mie(ϵ,R;f=22e9,N=3)
    c_const = 299792458
    k = 2π*f/c_const
    x = k*R
    m = sqrt(ϵ)
    δ_c = 0.0 + 0.0im
    for n in 1:1:N
       δ_c += (Kn(n,x,m)*cn(n,x,m) + Ln(n,x,m)*dn(n,x,m))
    end
    return δ_c
end


"""
Spherical Bessel function of the first kind of order n at x
"""
function jn(n,x)
    return sphericalbesselj(n,x)
end

"""
Spherical Hankel functions of order n at x
"""
function hn(n,x)
    return sphericalbesselj(n,x) + 1im*sphericalbessely(n,x)
end


"""
Mie factor cn (Eq.14.5 in J.Egge's Thesis)
"""
function cn(n,x,m)
        num = (hn(n,x)*(-jn(n-1,x)+jn(n+1,x))+jn(n,x)*(hn(n-1,x)-hn(n+1,x)))
        denom = (m*(-jn(n-1,m*x)+jn(n+1,m*x))*hn(n,x)+jn(n, m*x)*(hn(n-1,x)-hn(n+1,x)))
        return num/denom
end
"""
Mie factor dn (Eq.14.6 in J.Egge's Thesis)
"""
function dn(n,x,m)
    num = (m*x*((-jn(n-1,x)+jn(n+1, x))*hn(n, x)+jn(n,x)*(hn(n-1,x)-hn(n+1, x))))
    denom = (m*x*(-jn(n-1,m*x)+jn(n+1,m*x))*hn(n,x)+jn(n,m*x)*(m^2*x*hn(n-1,x)+(-1 + m^2)*hn(n,x)-m^2*x*hn(n+1,x)))
    return num/denom
end

"""
Mie factor Kn (Eq.14.9 in J.Egge's Thesis)
"""
function Kn(n,x,m)
    A = (m^2+2)/(2*x*(m^2-1))
    B = jn(n-1,x)*jn(n,m*x)
    C = m*jn(n,x)*jn(n-1,m*x)
    value = (2*n+1)*A*(B-C)
    return value
end
"""
Mie factor Ln (Eq.14.10 in J.Egge's Thesis)
"""
function Ln(n,x,m)
    A = (m^2+2)/(2*x*(m^2-1))
    B = jn(n-2,x)*jn(n-1,m*x)
    C = m*jn(n-1,x)*jn(n-2,m*x)
    D = jn(n,x)*jn(n+1,m*x)
    E = m*jn(n+1,x)*jn(n,m*x)
    value = A*((n+1)*(B-C)+n*(D-E))
    return value
end
