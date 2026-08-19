# fused_qr_size_reduction.jl
#
# Simultaneous QR factorization and size reduction of an integer basis.
#
# Given an integer basis B (m x n, columns are basis vectors), compute a
# unimodular U (n x n) and a floating point R such that B*U is size reduced and
# R is the R factor of B*U.
#
# The two operations have to be fused. The R factor of an UNREDUCED basis is
# catastrophically ill conditioned -- that is exactly what makes the basis
# unreduced -- so factoring first and reducing afterwards loses the information
# the reduction needs. Reducing column i against its predecessors BEFORE
# generating its own Householder reflector keeps R well scaled as it is built.
#
# This is a port of flatter's `FusedQRSizeRedImpl::Columnwise`
# (problems/fused_qr_sizered/columnwise.cpp). flatter's other implementations in
# that directory are not ported here:
#
#   * `ColumnwiseDouble` is the same algorithm on hardware doubles, with
#     machinery to carry a global power-of-two exponent shift so that R does not
#     overflow. BigFloat's exponent range makes that unnecessary, and this
#     implementation throws rather than silently producing Inf if a float type
#     too narrow for the input is requested.
#   * `LazyRefine` handles bases with already-reduced prefixes and needs a
#     driver that produces them.
#   * `Iterated` and `SeysenRefine` are unreachable from flatter's dispatcher.
#
# The output is in the same packed compact-WY-compatible layout as
# `householder` and `householder_block`: the upper trapezoid of R is the R
# factor, and below the diagonal column j holds the tail of the Householder
# vector v_j whose leading entry is an implicit one. Use `triu` for R alone, and
# `compact_wy_from_reflectors` to obtain the T factor that `apply_Qt!` wants.

const FUSED_QR_DEADBAND = 0.51
const FUSED_QR_MAX_PASSES = 64

# ---------------------------------------------------------------------------
# Single Householder reflector primitives
# ---------------------------------------------------------------------------
#
# `householder.jl` factors a whole matrix; the fused algorithm has to interleave
# reflector generation with integer column operations, so it needs the two
# LAPACK level primitives separately. These follow dlarfg and dlarf exactly, and
# produce the same packed layout `householder.jl` documents, so the outputs of
# the two are interchangeable.

"""
    _fused_larfg!(R, column, m) -> tau

Generate the Householder reflector that annihilates `R[column+1:m, column]`.

On return `R[column, column]` holds the resulting diagonal entry `beta` and
`R[column+1:m, column]` holds the tail of `v`, whose leading entry is an
implicit one, so that `H = I - tau*v*v'` satisfies `H*x = beta*e_1`.

The sign of `beta` is chosen opposite to the leading entry, which is what keeps
`alpha - beta` away from cancellation.
"""
function _fused_larfg!(R::AbstractMatrix{S}, column::Int, m::Int) where {S<:AbstractFloat}
    alpha = R[column, column]

    tail_norm_squared = zero(S)
    for i in (column + 1):m
        tail_norm_squared += R[i, column] * R[i, column]
    end

    if iszero(tail_norm_squared)
        return zero(S)
    end

    beta = sqrt(alpha * alpha + tail_norm_squared)
    signbit(alpha) || (beta = -beta)

    tau = (beta - alpha) / beta
    scale = inv(alpha - beta)
    for i in (column + 1):m
        R[i, column] *= scale
    end
    R[column, column] = beta

    return tau
end

"""
    _fused_larf_col!(R, target, reflector, tau, m)

Apply `H = I - tau*v*v'` to `R[reflector:m, target]` in place, where `v` is read
from column `reflector` of `R` with its implicit leading one.
"""
function _fused_larf_col!(R::AbstractMatrix{S}, target::Int, reflector::Int,
                          tau::S, m::Int) where {S<:AbstractFloat}
    iszero(tau) && return nothing

    inner = R[reflector, target]
    for i in (reflector + 1):m
        inner += R[i, reflector] * R[i, target]
    end
    inner *= tau

    R[reflector, target] -= inner
    for i in (reflector + 1):m
        R[i, target] -= inner * R[i, reflector]
    end

    return nothing
end

"""
    compact_wy_from_reflectors(factors, tau; reflectors = length(tau)) -> T

Build the compact-WY factor `T` from a packed factorization and its `tau`
vector, so that `Q = I - V*T*transpose(V)` with `V` read from `factors`.

`householder` and `householder_block` return `T` directly; the fused algorithm
generates reflectors one at a time and so produces `tau` instead. This converts
between the two, which is what lets a fused factorization be handed to
[`apply_Qt!`](@ref) and hence to the relative size reduction kernels.

The recurrence is the standard one: with `w = transpose(V[:, 1:j-1]) * V[:, j]`,

    T[1:j-1, j] = -tau[j] * T[1:j-1, 1:j-1] * w
    T[j, j]     = tau[j]
"""
function compact_wy_from_reflectors(factors::AbstractMatrix{S},
                                    tau::AbstractVector{S};
                                    reflectors::Integer = length(tau)) where {S<:AbstractFloat}
    m = size(factors, 1)
    k = Int(reflectors)
    k <= length(tau) || throw(DimensionMismatch(
        "asked for $k reflectors but tau has $(length(tau)) entries"))

    T = zeros(S, k, k)
    inner = Vector{S}(undef, max(k, 1))

    for j in 1:k
        if j > 1
            # inner[c] = <v_c, v_j>, using the implicit unit leading entries.
            # v_c is zero above row c and v_j is zero above row j, with j > c,
            # so the sum runs from row j.
            for c in 1:(j - 1)
                total = factors[j, c]                 # v_j[j] = 1
                for r in (j + 1):m
                    total += factors[r, c] * factors[r, j]
                end
                inner[c] = total
            end
            for a in 1:(j - 1)
                total = zero(S)
                for b in a:(j - 1)                     # T is upper triangular
                    total += T[a, b] * inner[b]
                end
                T[a, j] = -tau[j] * total
            end
        end
        T[j, j] = tau[j]
    end

    return T
end

# ---------------------------------------------------------------------------
# The fused sweep
# ---------------------------------------------------------------------------

# Is R[row, column] size reduced against the diagonal entry R[row, row]?
#
# flatter's three tier test, which avoids a division in the common cases:
# an entry at least as large as the diagonal is certainly not reduced, an entry
# below a quarter of it certainly is, and only the band between needs the actual
# quotient compared against the deadband.
#
# flatter evaluates that last comparison in double, having converted both
# operands out of MPFR. This implementation compares in S. The result differs
# only for entries within a double's rounding error of the threshold, where
# either answer is acceptable: a spurious reduction applies a valid transvection
# and a missed one is caught on the next pass.
@inline function _fused_is_reduced(entry::S, diagonal::S, deadband::S) where {S<:AbstractFloat}
    absolute_entry = abs(entry)
    absolute_diagonal = abs(diagonal)
    absolute_entry >= absolute_diagonal && return false
    4 * absolute_entry < absolute_diagonal && return true
    return absolute_entry / absolute_diagonal <= deadband
end

# Reduce column `column` against columns 1 .. column-1, returning
# (any_correction_applied, largest multiplier exponent).
function _fused_reduce_column!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                               R::AbstractMatrix{S}, column::Int,
                               m::Int, deadband::S) where {T<:Integer, S<:AbstractFloat}
    applied = false
    largest = typemin(Int)

    @inbounds for row in (column - 1):-1:1
        diagonal = R[row, row]
        iszero(diagonal) && throw(ArgumentError(
            "R has a zero diagonal entry at index $row; B is rank deficient, " *
            "or the working precision is too low"))

        _fused_is_reduced(R[row, column], diagonal, deadband) && continue
        applied = true

        quotient = round(R[row, column] / diagonal)
        iszero(quotient) && continue
        multiplier = T(quotient)
        largest = max(largest, safe_exponent(quotient))

        # Exact updates. B is dense so the whole column moves; U and R are
        # upper triangular in this region, so rows above `row` are structurally
        # zero and need not be touched.
        for k in 1:m
            B[k, column] -= multiplier * B[k, row]
        end
        for k in 1:row
            U[k, column] -= multiplier * U[k, row]
            R[k, column] -= quotient * R[k, row]
        end
    end

    return applied, largest
end

function _fused_columnwise!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                            R::AbstractMatrix{S}, tau::AbstractVector{S};
                            deadband::S, max_passes::Int) where {T<:Integer, S<:AbstractFloat}
    m, n = size(B)

    _set_identity!(U)
    for j in 1:n, i in 1:m
        R[i, j] = _to_float(S, B[i, j])
    end
    fill!(tau, zero(S))

    for column in 1:n
        previous = typemax(Int)
        passes = 0

        while true
            passes += 1
            passes <= max_passes || throw(ErrorException(
                "column $column did not reduce in $max_passes passes; the " *
                "working precision is too low for this basis"))

            applied, largest = _fused_reduce_column!(B, U, R, column, m, deadband)

            # Nothing was out of bounds, so the column is reduced and its float
            # coordinates are trustworthy.
            applied || break

            # Stagnation guard. flatter compares multiplier MAGNITUDES against
            # an absolute slack of prec/2, which is dimensionally odd: for large
            # multipliers the slack vanishes and for small ones it always fires.
            # This compares exponents instead, matching the guard in
            # `relative_size_reduction.jl`: if a pass did not produce smaller
            # multipliers than the last, the precision is exhausted and further
            # passes cannot help. See Notes.md.
            largest >= previous && break
            previous = largest

            # Re-orthogonalize. The float updates above accumulate error, so the
            # column is rebuilt from the EXACT integer column and the previous
            # reflectors re-applied. This is what stops error compounding across
            # passes, exactly as in the relative size reduction kernels.
            for i in 1:m
                R[i, column] = _to_float(S, B[i, column])
            end
            for j in 1:(column - 1)
                _fused_larf_col!(R, column, j, tau[j], m)
            end
        end

        # Generate this column's reflector and push it through the columns to
        # the right, so that the next column arrives already orthogonalized
        # against everything before it.
        if column < m
            tau[column] = _fused_larfg!(R, column, m)
            for target in (column + 1):n
                _fused_larf_col!(R, target, column, tau[column], m)
            end
        end
    end

    return B, U, R, tau
end

# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------

"""
    fused_qr_precision(B) -> Int

Default working precision for a fused factorization of `B`, from the package's
existing policy: [`householder_precision`](@ref) on the row count, with the
largest entry bit length standing in for the log of the condition number.

Pessimistic, as flatter's `get_initial_precision` is. Supply `precision`
explicitly when a better bound is known.
"""
function fused_qr_precision(B::AbstractMatrix{<:Integer})
    largest = 0
    for value in B
        iszero(value) && continue
        largest = max(largest, ndigits(value; base = 2))
    end
    return householder_precision(size(B, 1), Float64(largest))
end

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

"""
    fused_qr_size_reduction!(B, U; kwargs...) -> (B, U, R, tau)

Size reduce the integer basis `B` in place while building its QR factorization,
and overwrite `U` with the unimodular transform applied, so that on return the
new `B` equals the old `B` times `U`.

`B` is `m x n` with `m >= n` and columns as basis vectors; `U` is `n x n` with
the same integer element type. `U` is unimodular, though not triangular: column
operations run left to right but each column is reduced against all its
predecessors, so `U` is unit upper triangular only when no reduction crosses a
column boundary.

Returns the packed factorization in the layout used throughout this package:
the upper trapezoid of `R` is the R factor of the reduced basis, and below the
diagonal column `j` holds the tail of the Householder vector `v_j` with an
implicit leading one. Use `triu(R)` for the R factor alone, and
[`compact_wy_from_reflectors`](@ref) to build the `T` that [`apply_Qt!`](@ref)
needs.

Keyword arguments:

  * `float_type`  -- element type of `R`, `BigFloat` by default. `Float64` is
                     accepted but will throw on any basis whose entries exceed
                     its exponent range; flatter's `ColumnwiseDouble` handles
                     that case with a global exponent shift and is not ported.
  * `precision`   -- working precision when `float_type` is `BigFloat`.
                     Defaults to [`fused_qr_precision`](@ref).
  * `deadband`    -- threshold above which an entry counts as unreduced.
                     Default `$(FUSED_QR_DEADBAND)`, as in flatter.
  * `max_passes`  -- cap on reduction passes per column.

# Reduction quality

Each column is reduced against the Gram-Schmidt frame of its predecessors to
within `deadband` rather than exactly `1/2`, because of the deadband itself and
because a pass stops once the multipliers stop shrinking. The exact invariants,
true on every path, are that the returned `B` equals the original times `U` and
that `U` is unimodular.

Precision must cover the dynamic range of the basis: reducing column `i` needs
the coordinates of that column in the frame of columns `1:i-1` to be known to
better than the diagonal entries there. A basis whose profile spans more bits
than the working precision will stop early with a valid but poorly reduced
result, or raise if no column pass makes progress.
"""
function fused_qr_size_reduction!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
                                  float_type::Type{<:AbstractFloat} = BigFloat,
                                  precision::Union{Nothing, Integer} = nothing,
                                  deadband::Real = FUSED_QR_DEADBAND,
                                  max_passes::Integer = FUSED_QR_MAX_PASSES) where {T<:Integer}
    m, n = size(B)
    m >= n || throw(DimensionMismatch(
        "expected at least as many rows as columns, got $m x $n"))
    (iszero(m) || iszero(n)) && throw(ArgumentError("B must be non-empty"))
    size(U) == (n, n) || throw(DimensionMismatch(
        "U must be $n x $n, got $(size(U))"))
    deadband > 0 || throw(ArgumentError("deadband must be positive"))
    max_passes >= 1 || throw(ArgumentError("max_passes must be at least one"))

    if float_type === BigFloat
        bits = precision === nothing ? fused_qr_precision(B) : Int(precision)
        # BigFloat arithmetic takes its result precision from the global
        # default, not from the operands, so the whole computation has to run
        # inside the block, not just the allocations. See Notes.md.
        return with_precision(bits) do
            R = Matrix{BigFloat}(undef, m, n)
            tau = Vector{BigFloat}(undef, n)
            _fused_columnwise!(B, U, R, tau;
                               deadband = BigFloat(deadband),
                               max_passes = Int(max_passes))
            (B, U, R, tau)
        end
    end

    S = float_type
    R = Matrix{S}(undef, m, n)
    tau = Vector{S}(undef, n)
    _fused_columnwise!(B, U, R, tau;
                       deadband = S(deadband), max_passes = Int(max_passes))
    return B, U, R, tau
end

"""
    fused_qr_size_reduction(B; kwargs...) -> (B_reduced, U, R, tau)

Non-mutating form of [`fused_qr_size_reduction!`](@ref). `B` is left untouched.
"""
function fused_qr_size_reduction(B::AbstractMatrix{T}; kwargs...) where {T<:Integer}
    reduced = Matrix{T}(B)
    U = Matrix{T}(undef, size(B, 2), size(B, 2))
    return fused_qr_size_reduction!(reduced, U; kwargs...)
end
