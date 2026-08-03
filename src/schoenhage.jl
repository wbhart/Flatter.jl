const SCHOENHAGE_DEFAULT_CUTOFF = 200

@inline function _schoenhage_bitlength(x::BigInt)
    return iszero(x) ? 1 : ndigits(abs(x); base=2)
end

@inline function _schoenhage_identity()
    return BigInt[1 0; 0 1]
end

"""
    _schoenhage_transform_form(a, b, c, U)

Apply the integral change of variables `U` to the binary quadratic form

    a*x^2 + b*x*y + c*y^2.

If the columns of a lattice basis are transformed by `U`, this returns the
coefficients of the transformed form.
"""
function _schoenhage_transform_form(
    a::BigInt,
    b::BigInt,
    c::BigInt,
    U::Matrix{BigInt},
)
    u = U[1, 1]
    v = U[1, 2]
    w = U[2, 1]
    z = U[2, 2]

    alpha = a*u*u + b*u*w + c*w*w
    beta = 2*a*u*v + b*(u*z + v*w) + 2*c*w*z
    gamma = a*v*v + b*v*z + c*z*z

    return alpha, beta, gamma
end

"""Multiply two `2 x 2` `BigInt` matrices without a general matrix kernel."""
function _schoenhage_mul2x2(A::Matrix{BigInt}, B::Matrix{BigInt})
    c11 = A[1, 1]*B[1, 1] + A[1, 2]*B[2, 1]
    c12 = A[1, 1]*B[1, 2] + A[1, 2]*B[2, 2]
    c21 = A[2, 1]*B[1, 1] + A[2, 2]*B[2, 1]
    c22 = A[2, 1]*B[1, 2] + A[2, 2]*B[2, 2]
    return BigInt[c11 c12; c21 c22]
end

"""
Update the accumulated transformation after one simple reduction step.

For a low step, `t` has already had its sign reversed, as in the original
algorithm.
"""
function _schoenhage_update_transform!(
    U::Matrix{BigInt},
    t::BigInt,
    is_low_step::Bool,
)
    if is_low_step
        U[1, 2] += U[1, 1]*t
        U[2, 2] += U[2, 1]*t
    else
        U[1, 1] -= U[1, 2]*t
        U[2, 1] -= U[2, 2]*t
    end

    return U
end

function _schoenhage_swap_columns!(U::Matrix{BigInt})
    u11 = U[1, 1]
    u21 = U[2, 1]
    U[1, 1] = U[1, 2]
    U[2, 1] = U[2, 2]
    U[1, 2] = u11
    U[2, 2] = u21
    return U
end

"""
Perform one reduction step above `s = 2^m`.

The form is represented by `(a, b, c)`, corresponding to the symmetric
matrix `[a b/2; b/2 c]`. The returned `t` is the value used to update the
accumulated transformation matrix. For a low step its sign is reversed.
"""
function _schoenhage_simple_step(
    a::BigInt,
    b::BigInt,
    c::BigInt,
    m::Int,
)
    d = b*b - 4*a*c
    s = BigInt(1) << m
    four_s_squared = 4*s*s

    if a < c
        z = d + 4*a*s
        r = if z < four_s_squared
            2*s
        else
            isqrt(z - 1) + 1
        end

        q = fld(b - r, 2*a)
        new_b = b - 2*a*q
        new_c = c - b*q + a*q*q

        # The transformation update uses the negative of the quotient for a
        # low step.
        return a, new_b, new_c, -q, true
    else
        z = d + 4*c*s
        r = if z < four_s_squared
            2*s
        else
            isqrt(z - 1) + 1
        end

        q = fld(b - r, 2*c)
        new_a = a - b*q + c*q*q
        new_b = b - 2*c*q

        return new_a, new_b, c, q, false
    end
end

"""Return whether `(a, b, c)` is minimal above `2^m`."""
function _schoenhage_is_minimal(
    a::BigInt,
    b::BigInt,
    c::BigInt,
    m::Int,
)
    if a <= 0 || _schoenhage_bitlength(a) <= m ||
       b <= 0 || _schoenhage_bitlength(b) <= m ||
       c <= 0 || _schoenhage_bitlength(c) <= m
        return false
    end

    t = a - b + c
    if t <= 0 || _schoenhage_bitlength(t) <= m
        return true
    end

    t = b - 2*a
    if t > 0 && _schoenhage_bitlength(t) > m + 1
        return false
    end

    t = b - 2*c
    if t > 0 && _schoenhage_bitlength(t) > m + 1
        return false
    end

    return true
end

"""Classical sequence of simple steps, used below the recursion cutoff."""
function _schoenhage_nonrecursive(
    a::BigInt,
    b::BigInt,
    c::BigInt,
    m::Int,
)
    U = _schoenhage_identity()

    while true
        a, b, c, t, is_low_step = _schoenhage_simple_step(a, b, c, m)
        iszero(t) && break
        _schoenhage_update_transform!(U, t, is_low_step)
    end

    return a, b, c, U
end

"""Recursive core of Schoenhage reduction for positive binary forms."""
function _schoenhage_recursive(
    a::BigInt,
    b::BigInt,
    c::BigInt,
    m::Int,
    cutoff::Int,
)
    a_len = _schoenhage_bitlength(a)
    b_len = _schoenhage_bitlength(b)
    c_len = _schoenhage_bitlength(c)
    n = max(a_len, b_len, c_len) - m

    if n < cutoff
        return _schoenhage_nonrecursive(a, b, c, m)
    end

    U = _schoenhage_identity()
    alpha = BigInt(0)
    beta = BigInt(0)
    gamma = BigInt(0)

    # R1. If one coefficient is already too small, no high-part recursion is
    # useful at this level.
    if min(a_len, b_len, c_len) <= m + 2
        alpha, beta, gamma = a, b, c
    else
        # R2. Possibly remove a common block of low bits.
        if m <= n
            m_prime = m
            p = 0
            a0 = BigInt(0)
            b0 = BigInt(0)
            c0 = BigInt(0)
        else
            m_prime = n
            p = m + 1 - n
            modulus = BigInt(1) << p

            a0 = mod(a, modulus)
            b0 = mod(b, modulus)
            c0 = mod(c, modulus)

            a = div(a, modulus)
            b = div(b, modulus)
            c = div(c, modulus)

            a_len = _schoenhage_bitlength(a)
            b_len = _schoenhage_bitlength(b)
            c_len = _schoenhage_bitlength(c)
        end

        # R3--R4. Reduce the leading part to approximately half the current
        # precision.
        h = m_prime + div(n, 2)
        if min(a_len, b_len, c_len) <= h
            U = _schoenhage_identity()
        else
            a, b, c, U = _schoenhage_recursive(a, b, c, h, cutoff)
            a_len = _schoenhage_bitlength(a)
            b_len = _schoenhage_bitlength(b)
            c_len = _schoenhage_bitlength(c)
        end

        # R5. Continue with simple steps until the coefficients fit below the
        # half-size bound, or until minimality at the lower level is reached.
        going_to_r8 = false
        while max(a_len, b_len, c_len) > h
            if _schoenhage_is_minimal(a, b, c, m_prime)
                alpha, beta, gamma = a, b, c
                going_to_r8 = true
                break
            end

            a, b, c, t, is_low_step =
                _schoenhage_simple_step(a, b, c, m_prime)
            _schoenhage_update_transform!(U, t, is_low_step)

            a_len = _schoenhage_bitlength(a)
            b_len = _schoenhage_bitlength(b)
            c_len = _schoenhage_bitlength(c)
        end

        if !going_to_r8
            # R6--R7. Finish the lower-level reduction and compose the two
            # transformation matrices.
            a, b, c, U_prime =
                _schoenhage_recursive(a, b, c, m_prime, cutoff)
            alpha, beta, gamma = a, b, c
            U = _schoenhage_mul2x2(U, U_prime)
        end

        # R8. Restore the discarded low parts after applying the same change
        # of variables to them.
        if p > 0
            a0, b0, c0 = _schoenhage_transform_form(a0, b0, c0, U)
            alpha = (alpha << p) + a0
            beta = (beta << p) + b0
            gamma = (gamma << p) + c0
        end
    end

    a, b, c = alpha, beta, gamma

    # R9. Complete minimal reduction at the requested level.
    while !_schoenhage_is_minimal(a, b, c, m)
        a, b, c, t, is_low_step = _schoenhage_simple_step(a, b, c, m)
        iszero(t) && break
        _schoenhage_update_transform!(U, t, is_low_step)
    end

    return a, b, c, U
end

"""
    schoenhage(G; cutoff=200) -> R, U

Reduce a positive-definite integral `2 x 2` Gram matrix using Schoenhage's
recursive reduction algorithm for positive binary quadratic forms.

The input may be any `AbstractMatrix` with integer entries. It is converted to
`BigInt`. The result consists of two `Matrix{BigInt}` objects satisfying

    R == transpose(U) * G * U,

where `U` is unimodular. The reduced Gram matrix satisfies

    R[1, 1] <= R[2, 2]
    2*abs(R[1, 2]) <= R[1, 1].

The function is intended to remain internal to the containing module; it need
not be exported. The `cutoff` is the bit-size crossover to the nonrecursive
sequence of simple steps.
"""
function schoenhage(
    G::AbstractMatrix{<:Integer};
    cutoff::Integer=SCHOENHAGE_DEFAULT_CUTOFF,
)
    size(G) == (2, 2) ||
        throw(DimensionMismatch("schoenhage requires a 2 x 2 Gram matrix"))
    cutoff >= 2 || throw(ArgumentError("cutoff must be at least 2"))
    cutoff <= typemax(Int) || throw(ArgumentError("cutoff is too large"))

    G0 = BigInt[
        G[1, 1] G[1, 2]
        G[2, 1] G[2, 2]
    ]

    G0[1, 2] == G0[2, 1] ||
        throw(ArgumentError("the Gram matrix must be symmetric"))

    a = G0[1, 1]
    b = 2*G0[1, 2]
    c = G0[2, 2]

    determinant = a*c - G0[1, 2]*G0[1, 2]
    if a <= 0 || determinant <= 0
        throw(ArgumentError("the Gram matrix must be positive definite"))
    end

    a_input, b_input, c_input = a, b, c

    # The recursive algorithm assumes a positive middle coefficient. Negating
    # the second basis vector changes only its sign.
    flipped_b = b < 0
    if flipped_b
        b = -b
    end

    if b > 0
        a, b, c, U = _schoenhage_recursive(a, b, c, 0, Int(cutoff))
    else
        U = _schoenhage_identity()
    end

    # Convert minimality above 1 to the usual Lagrange reduction conditions.
    if c < a
        a, c = c, a
        _schoenhage_swap_columns!(U)
    end

    if abs(b) > a
        # One low step with quotient 1. At this stage the theory guarantees
        # that no further size-reduction step is required.
        b -= a
        c -= b
        b -= a

        U[1, 2] -= U[1, 1]
        U[2, 2] -= U[2, 1]

        if c < a
            a, c = c, a
            _schoenhage_swap_columns!(U)
        end
    end

    if !(a <= c && abs(b) <= a)
        error("internal error: Schoenhage reduction did not produce a reduced form")
    end

    if flipped_b
        U[2, 1] = -U[2, 1]
        U[2, 2] = -U[2, 2]
    end

    iseven(b) || error("internal error: transformed Gram entry is not integral")
    h = div(b, 2)
    R = BigInt[a h; h c]

    # These inexpensive 2 x 2 checks also guard against mistakes in the
    # transformation bookkeeping.
    check_a, check_b, check_c =
        _schoenhage_transform_form(a_input, b_input, c_input, U)
    if check_a != a || check_b != b || check_c != c
        error("internal error: transformation matrix does not match the reduced form")
    end

    det_U = U[1, 1]*U[2, 2] - U[1, 2]*U[2, 1]
    abs(det_U) == 1 || error("internal error: transformation is not unimodular")

    return R, U
end
