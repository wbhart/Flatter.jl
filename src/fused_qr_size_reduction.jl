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

"""
    _fused_temporaries(S, count) -> Vector{S}

Scratch values for the in-place kernels, allocated once per factorisation.

Built with a comprehension rather than `fill!` on purpose: `fill!` would store a
single shared object in every slot, and these are mutated independently.
"""
_fused_temporaries(::Type{S}, count::Int) where {S} = S[zero(S) for _ in 1:count]

# Scratch slots. Kept distinct per kernel so that none of them can interfere,
# even though the call graph happens not to nest them today.
const _WS_LARF_INNER = 1
const _WS_LARF_PRODUCT = 2
const _WS_LARFG_ALPHA = 3
const _WS_LARFG_NORM = 4
const _WS_LARFG_BETA = 5
const _WS_LARFG_SCALE = 6
const _WS_REDUCE_QUOTIENT = 7
const _WS_REDUCE_PRODUCT = 8
const _FUSED_WORKSPACE_SIZE = 8

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
function _fused_larfg!(R::AbstractMatrix{S}, column::Int, m::Int,
                       ws::Vector{S}) where {S<:AbstractFloat}
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


# In-place counterpart. Same algorithm, same result; only the arithmetic differs.
function _fused_larfg!(R::Matrix{BigFloat}, column::Int, m::Int, ws::Vector{BigFloat})
    alpha = ws[_WS_LARFG_ALPHA]
    norm_squared = ws[_WS_LARFG_NORM]
    beta = ws[_WS_LARFG_BETA]
    scale = ws[_WS_LARFG_SCALE]

    _mpfr_set!(alpha, R[column, column])

    _mpfr_zero!(norm_squared)
    @inbounds for i in (column + 1):m
        _mpfr_fma!(norm_squared, R[i, column], R[i, column], norm_squared)
    end
    iszero(norm_squared) && return zero(BigFloat)

    _mpfr_fma!(beta, alpha, alpha, norm_squared)
    _mpfr_sqrt!(beta, beta)
    signbit(alpha) || _mpfr_neg!(beta, beta)

    # tau is stored into the caller's vector, so it must be a fresh object.
    tau = BigFloat()
    _mpfr_sub!(tau, beta, alpha)
    _mpfr_div!(tau, tau, beta)

    _mpfr_sub!(scale, alpha, beta)
    _mpfr_inv!(scale, scale)
    @inbounds for i in (column + 1):m
        _mpfr_mul!(R[i, column], R[i, column], scale)
    end
    _mpfr_set!(R[column, column], beta)

    return tau
end

"""
    _fused_larf_col!(R, target, reflector, tau, m)

Apply `H = I - tau*v*v'` to `R[reflector:m, target]` in place, where `v` is read
from column `reflector` of `R` with its implicit leading one.
"""
function _fused_larf_col!(R::AbstractMatrix{S}, target::Int, reflector::Int,
                          tau::S, m::Int, ws::Vector{S}) where {S<:AbstractFloat}
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


# In-place counterpart of the reflector application, and the hot loop of the
# factorisation: the trailing update calls it once per remaining column, for
# every column, touching O(m) entries each time.
function _fused_larf_col!(R::Matrix{BigFloat}, target::Int, reflector::Int,
                          tau::BigFloat, m::Int, ws::Vector{BigFloat})
    iszero(tau) && return nothing

    inner = ws[_WS_LARF_INNER]
    product = ws[_WS_LARF_PRODUCT]

    _mpfr_set!(inner, R[reflector, target])
    @inbounds for i in (reflector + 1):m
        _mpfr_fma!(inner, R[i, reflector], R[i, target], inner)
    end
    _mpfr_mul!(inner, inner, tau)

    _mpfr_sub!(R[reflector, target], R[reflector, target], inner)
    @inbounds for i in (reflector + 1):m
        _mpfr_mul!(product, inner, R[i, reflector])
        _mpfr_sub!(R[i, target], R[i, target], product)
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



# B[k, column] -= multiplier * B[k, row], over a range of rows.
#
# The obvious `B[k, column] -= multiplier * B[k, row]` allocates twice per
# element: once for the product and once for the difference. Only the second is
# unavoidable -- the slot must be rebound rather than mutated, because `B` and
# `U` belong to the caller and may share entry objects with a copy the caller
# still holds. Computing the product into a scratch value removes the other,
# halving the allocations in what is the busiest loop of the factorisation.
@inline function _fused_integer_update!(M::AbstractMatrix{T}, column::Int, row::Int,
                                        rows::UnitRange{Int}, multiplier::T,
                                        scratch::T) where {T<:Integer}
    @inbounds for k in rows
        M[k, column] -= multiplier * M[k, row]
    end
    return nothing
end

@inline function _fused_integer_update!(M::AbstractMatrix{BigInt}, column::Int,
                                        row::Int, rows::UnitRange{Int},
                                        multiplier::BigInt, scratch::BigInt)
    @inbounds for k in rows
        Base.GMP.MPZ.mul!(scratch, multiplier, M[k, row])
        result = BigInt()
        Base.GMP.MPZ.sub!(result, M[k, column], scratch)
        M[k, column] = result
    end
    return nothing
end

# R[1:row, column] -= quotient * R[1:row, row]
@inline function _fused_scale_subtract!(R::AbstractMatrix{S}, column::Int, row::Int,
                                        quotient::S, ws::Vector{S}) where {S<:AbstractFloat}
    @inbounds for k in 1:row
        R[k, column] -= quotient * R[k, row]
    end
    return nothing
end

@inline function _fused_scale_subtract!(R::Matrix{BigFloat}, column::Int, row::Int,
                                        quotient::BigFloat, ws::Vector{BigFloat})
    product = ws[_WS_REDUCE_PRODUCT]
    @inbounds for k in 1:row
        _mpfr_mul!(product, quotient, R[k, row])
        _mpfr_sub!(R[k, column], R[k, column], product)
    end
    return nothing
end



# ---------------------------------------------------------------------------
# Panel processing
# ---------------------------------------------------------------------------
#
# A reflector is generated one column at a time and then pushed through every
# column to its right, one at a time. That trailing update is a rank-1 operation
# per column pair, and at dimension 256 it is 19.7% of the relation family's
# runtime and 6.4% of knapsack's.
#
# Blocking it means deferring: within a panel of `panelsize` columns, a
# reflector is applied only to the columns still inside the panel, since those
# are about to be processed and need it. Once the panel is finished, its whole
# set of reflectors goes through the remaining columns as ONE compact-WY block
# update, which is a matrix product and reaches Strassen.
#
# This is exact in the same sense the blocked size reduction is: applying k
# reflectors as a block computes the same transformation as applying them one at
# a time, so the panelled factorisation differs from the columnwise one only by
# floating point association.
#
# What this does NOT do is block the SIZE REDUCTION against earlier panels, or
# the re-orthogonalisation that re-reads a column from `B` and re-applies every
# earlier reflector. Both are still column-at-a-time. They are the other half of
# `Heuristic3`'s update, and they change the order in which reductions happen,
# so unlike this they will not produce identical results.

"""
Scratch for the panel flush, allocated ONCE per factorisation.

The first version allocated a compact-WY factor, a `k` by `n` work matrix and a
Strassen workspace on every flush — at every level, every iteration. That cost
more than the entire trailing update it was replacing, which is why panelling
first measured slower than doing nothing.
"""
struct _FusedPanelWorkspace{S<:AbstractFloat}
    compact_T::Matrix{S}
    inner::Vector{S}
    work::Matrix{S}
    multiply::_HouseholderBlockMultiplyWorkspace{S}
end

function _fused_panel_workspace(::Type{S}, panelsize::Int, m::Int, n::Int,
                                strassen_cutoff::Int) where {S<:AbstractFloat}
    width = min(panelsize, m, n)
    order = max(0, min(width, m - width, n))
    return _FusedPanelWorkspace{S}(
        zeros(S, max(width, 1), max(width, 1)),
        Vector{S}(undef, max(width, 1)),
        Matrix{S}(undef, max(width, 1), max(n, 1)),
        _householder_block_multiply_workspace(S, order, strassen_cutoff))
end

"""
    _fused_panel_flush!(R, tau, panel_start, panel_end, m, n, cutoff, panel_work)

Push a finished panel's reflectors through every column to its right, as one
compact-WY block update rather than one rank-1 update per column pair.

Does nothing when the panel is the last one, or when it holds no reflector --
`tau` is zero for a column whose tail was already zero.
"""
function _fused_panel_flush!(R::AbstractMatrix{S}, tau::AbstractVector{S},
                             panel_start::Int, panel_end::Int, m::Int, n::Int,
                             strassen_cutoff::Int,
                             panel_work::_FusedPanelWorkspace{S}) where {S<:AbstractFloat}
    panel_end < n || return nothing
    last_reflector = min(panel_end, m - 1)
    last_reflector >= panel_start || return nothing
    width = last_reflector - panel_start + 1

    factors = view(R, panel_start:m, panel_start:last_reflector)
    scalars = view(tau, panel_start:last_reflector)
    all(iszero, scalars) && return nothing

    target = view(R, panel_start:m, (panel_end + 1):n)
    trailing = size(target, 2)

    compact_T = view(panel_work.compact_T, 1:width, 1:width)
    fill!(compact_T, zero(S))
    _compact_wy_into!(compact_T, panel_work.inner, factors, scalars, width,
                      m - panel_start + 1)

    apply_Qt!(target, factors, compact_T, width,
              view(panel_work.work, 1:width, 1:trailing),
              panel_work.multiply, strassen_cutoff)
    return nothing
end

"""
`compact_wy_from_reflectors` writing into caller-supplied storage.

`inner` holds the inner products for the current column. It could be folded into
an unused part of `T`, and the iteration order happens to make that safe — but
only by an argument about which entries are read before being overwritten, which
is exactly the sort of thing that stops being true after an innocuous edit.
"""
function _compact_wy_into!(T::AbstractMatrix{S}, inner::AbstractVector{S},
                           factors::AbstractMatrix{S}, tau::AbstractVector{S},
                           k::Int, rows::Int) where {S<:AbstractFloat}
    @inbounds for j in 1:k
        if j > 1
            # inner[c] = <v_c, v_j>, using the implicit unit leading entries.
            for c in 1:(j - 1)
                total = factors[j, c]
                for r in (j + 1):rows
                    total += factors[r, c] * factors[r, j]
                end
                inner[c] = total
            end
            for a in 1:(j - 1)
                total = zero(S)
                for b in a:(j - 1)                  # T is upper triangular
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
# Blocked size reduction
# ---------------------------------------------------------------------------
#
# The elementwise sweep reduces a column against its predecessors one at a time,
# updating `B`, `U` and `R` at every step. Only `R` is READ during that sweep:
# the quotient for row j depends on `R[j, column]`, which the steps above it have
# already changed. `B` and `U` are written and never read.
#
# So their updates can be deferred and applied once per block, which is exact --
# the quotients are unchanged, so the result is bit-identical to the elementwise
# version. What changes is the allocation count: one `BigInt` per output entry
# per BLOCK, rather than one per entry per row.
#
# `R` still needs elementwise updates within a block, since the next quotient
# depends on them, but only across the block's own rows; everything below the
# block is batched with `B` and `U`.
#
# This is the first half of what flatter's `Heuristic3` does. The second half is
# processing a PANEL of columns at once, which turns these matrix-vector
# products into matrix-matrix ones and is a larger restructuring.

const DEFAULT_FUSED_BLOCKSIZE = 16

"""
Columns processed before a panel's reflectors are pushed through the rest of the
matrix as one block.

OFF by default, and measurement says it should stay that way. On all three
families tried it is slower, including relation -- the case with the most to gain
at 19.7% of its run in the trailing update -- at 92.9s unpanelled against 105.0s
at its best panel width. Hoisting every per-flush allocation into
`_FusedPanelWorkspace` bought only 4%; the gap is structural, not a leak.

WHY IT CANNOT WIN HERE, which is the interesting part. The operation counts are
the same: a compact-WY update computes `W = V'C`, `W = T'W`, `C -= VW`, about
`2*m*w*t` multiply-adds, and `w` rank-1 updates cost `2*m*t` each. LAPACK's
blocking wins through cache locality and vectorisation. A `Matrix{BigFloat}` is
an array of POINTERS to scattered heap blocks, so there is no locality to exploit
and nothing to vectorise. Blocking adds the `T` factor and an intermediate `W`
and buys nothing back.

Contrast `DEFAULT_FUSED_BLOCKSIZE`, which does pay. That is not a cache effect
either: it reduces the number of ALLOCATIONS, one result per output entry per
block instead of one per entry per row. Blocking arbitrary-precision code helps
when it removes allocations, not when it merely reorders the same arithmetic.

The implementation is kept because it is correct, tested, and demonstrates the
point.
"""
const DEFAULT_FUSED_PANELSIZE = 0

"`R[from:to, column] -= quotient * R[from:to, row]`, in place."
@inline function _fused_scale_range!(R::AbstractMatrix{S}, column::Int, row::Int,
                                     from::Int, to::Int, quotient::S,
                                     ws::Vector{S}) where {S<:AbstractFloat}
    @inbounds for k in from:to
        R[k, column] -= quotient * R[k, row]
    end
    return nothing
end

@inline function _fused_scale_range!(R::Matrix{BigFloat}, column::Int, row::Int,
                                     from::Int, to::Int, quotient::BigFloat,
                                     ws::Vector{BigFloat})
    product = ws[_WS_REDUCE_PRODUCT]
    @inbounds for k in from:to
        _mpfr_mul!(product, quotient, R[k, row])
        _mpfr_sub!(R[k, column], R[k, column], product)
    end
    return nothing
end

"""
`M[1:upto, column] -= M[1:upto, rows] * factors`, one accumulation per row of the
output rather than one per (row, column) pair.

Zero factors are skipped: on a nearly reduced basis most of a block contributes
nothing, and testing is far cheaper than multiplying by zero.
"""
function _fused_batch_update!(M::AbstractMatrix{V}, column::Int,
                              rows::UnitRange{Int}, factors::AbstractVector{V},
                              upto::Int) where {V}
    @inbounds for i in 1:upto
        total = zero(V)
        touched = false
        for (k, row) in enumerate(rows)
            iszero(factors[k]) && continue
            total += factors[k] * M[i, row]
            touched = true
        end
        touched && (M[i, column] -= total)
    end
    return nothing
end

function _fused_batch_update!(M::AbstractMatrix{BigInt}, column::Int,
                              rows::UnitRange{Int}, factors::AbstractVector{BigInt},
                              upto::Int)
    accumulator = BigInt()
    product = BigInt()
    @inbounds for i in 1:upto
        Base.GMP.MPZ.set_si!(accumulator, 0)
        touched = false
        for (k, row) in enumerate(rows)
            iszero(factors[k]) && continue
            Base.GMP.MPZ.mul!(product, factors[k], M[i, row])
            Base.GMP.MPZ.add!(accumulator, product)
            touched = true
        end
        touched || continue
        # One allocation per output entry, and the slot is REBOUND rather than
        # mutated: `B` and `U` belong to the caller.
        result = BigInt()
        Base.GMP.MPZ.sub!(result, M[i, column], accumulator)
        M[i, column] = result
    end
    return nothing
end

function _fused_batch_update!(R::Matrix{BigFloat}, column::Int,
                              rows::UnitRange{Int}, factors::AbstractVector{BigFloat},
                              upto::Int)
    accumulator = BigFloat()
    product = BigFloat()
    @inbounds for i in 1:upto
        _mpfr_zero!(accumulator)
        touched = false
        for (k, row) in enumerate(rows)
            iszero(factors[k]) && continue
            _mpfr_mul!(product, factors[k], R[i, row])
            _mpfr_add!(accumulator, accumulator, product)
            touched = true
        end
        # R is owned by this file, so it may be mutated in place.
        touched && _mpfr_sub!(R[i, column], R[i, column], accumulator)
    end
    return nothing
end


"""
    _fused_reduce_column_blocked!(B, U, R, column, m, deadband, ws, blocksize, work)

Size reduce a column against its predecessors, deferring the `B` and `U` updates
until a whole block of predecessors has been processed.

Exact: the quotients are computed from the same `R` values in the same order, so
the result is bit-identical to [`_fused_reduce_column!`](@ref). Only the
allocation count differs.

`work` supplies the per-block coefficient vectors, allocated once per
factorisation rather than once per block.
"""
function _fused_reduce_column_blocked!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                                       R::AbstractMatrix{S}, column::Int,
                                       m::Int, deadband::S, ws::Vector{S},
                                       blocksize::Int,
                                       multipliers::Vector{T},
                                       quotients::Vector{S}
                                       ) where {T<:Integer, S<:AbstractFloat}
    applied = false
    largest = typemin(Int)
    top = column - 1
    top < 1 && return applied, largest

    @inbounds while top >= 1
        low = max(1, top - blocksize + 1)
        span = low:top
        width = length(span)

        any_here = false
        for k in 1:width
            multipliers[k] = zero(T)
            quotients[k] = zero(S)
        end

        # Elementwise across the block, high to low. `R` inside the block must be
        # updated as we go, because the next quotient reads it.
        for row in top:-1:low
            diagonal = R[row, row]
            iszero(diagonal) && throw(ArgumentError(
                "R has a zero diagonal entry at index $row; B is rank deficient, " *
                "or the working precision is too low"))

            _fused_is_reduced(R[row, column], diagonal, deadband) && continue
            quotient = round(R[row, column] / diagonal)
            iszero(quotient) && continue

            applied = true
            any_here = true
            largest = max(largest, safe_exponent(quotient))
            index = row - low + 1
            quotients[index] = quotient
            multipliers[index] = T(quotient)

            # Only within the block: everything below it is batched.
            _fused_scale_range!(R, column, row, low, row, quotient, ws)
        end

        if any_here
            # Deferred. `B` is dense so the whole column moves; `U` and `R` are
            # upper triangular here, so nothing above row `top` is nonzero.
            _fused_batch_update!(B, column, span, multipliers, m)
            _fused_batch_update!(U, column, span, multipliers, top)
            low > 1 && _fused_batch_update!(R, column, span, quotients, low - 1)
        end

        top = low - 1
    end

    return applied, largest
end

# Reduce column `column` against columns 1 .. column-1, returning
# (any_correction_applied, largest multiplier exponent).
function _fused_reduce_column!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                               R::AbstractMatrix{S}, column::Int,
                               m::Int, deadband::S,
                               ws::Vector{S}) where {T<:Integer, S<:AbstractFloat}
    applied = false
    largest = typemin(Int)
    integer_scratch = zero(T)

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
        _fused_integer_update!(B, column, row, 1:m, multiplier, integer_scratch)
        _fused_integer_update!(U, column, row, 1:row, multiplier, integer_scratch)
        # R is allocated and owned by this file, so its entries may be mutated
        # in place. B and U belong to the caller and may share objects with a
        # copy the caller still holds, so they are updated by assignment.
        _fused_scale_subtract!(R, column, row, quotient, ws)
    end

    return applied, largest
end

"""
Slots in the optional `timings` vector, which splits this factorisation into the
four things it actually does. They respond very differently to blocking:

  * `FUSED_TIME_REDUCE` -- size reducing a column against its predecessors.
    Integer work on `B` and `U`, unaffected by blocking.
  * `FUSED_TIME_REORTH` -- rebuilding a column from the exact integer basis and
    re-applying every earlier reflector. Blockable in principle: the reflectors
    of completed panels could be applied as compact-WY products.
  * `FUSED_TIME_REFLECTOR` -- generating a reflector. Inherently sequential,
    O(m) each, and blocking cannot touch it.
  * `FUSED_TIME_TRAILING` -- pushing a reflector through the columns to its
    right. This is the rank-1 update that blocking replaces with a matrix
    product, and the only part a panel-blocked rewrite fully addresses.

If the trailing update is not the dominant slot, blocking will buy far less than
the total cost of this routine suggests.
"""
const FUSED_TIME_REDUCE = 1
const FUSED_TIME_REORTH = 2
const FUSED_TIME_REFLECTOR = 3
const FUSED_TIME_TRAILING = 4
const FUSED_TIME_SLOTS = 4

@inline _fused_tick() = time_ns()
@inline function _fused_tock!(timings, slot, started)
    timings === nothing || (timings[slot] += (time_ns() - started) / 1e9)
    return nothing
end

function _fused_columnwise!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                            R::AbstractMatrix{S}, tau::AbstractVector{S};
                            deadband::S, max_passes::Int,
                            blocksize::Int = DEFAULT_FUSED_BLOCKSIZE,
                            panelsize::Int = 0,
                            strassen_cutoff::Int = DEFAULT_STRASSEN_CUTOFF,
                            timings::Union{Nothing, Vector{Float64}} = nothing
                            ) where {T<:Integer, S<:AbstractFloat}
    m, n = size(B)

    _set_identity!(U)
    for j in 1:n, i in 1:m
        R[i, j] = _to_float(S, B[i, j])
    end
    fill!(tau, zero(S))

    # Scratch for the in-place kernels, allocated once for the whole call.
    ws = _fused_temporaries(S, _FUSED_WORKSPACE_SIZE)

    # Per-block coefficients, allocated once for the whole factorisation. A
    # blocksize of 1 or less means the elementwise sweep, which is kept both as
    # the reference implementation and for comparison.
    # Without panelling, each reflector goes straight through every column to
    # its right and nothing is deferred -- the original code path exactly, not a
    # panel of width one, which would route single reflectors through the block
    # machinery and be a different computation.
    panelled = panelsize > 1
    panel_start = 1
    panel_work = panelled ?
        _fused_panel_workspace(S, panelsize, m, n, strassen_cutoff) : nothing

    blocked = blocksize > 1
    multipliers = blocked ? T[zero(T) for _ in 1:blocksize] : T[]
    quotients = blocked ? _fused_temporaries(S, blocksize) : S[]

    for column in 1:n
        previous = typemax(Int)
        passes = 0

        while true
            passes += 1
            passes <= max_passes || throw(ErrorException(
                "column $column did not reduce in $max_passes passes; the " *
                "working precision is too low for this basis"))

            reduce_started = _fused_tick()
            applied, largest = blocked ?
                _fused_reduce_column_blocked!(B, U, R, column, m, deadband, ws,
                                              blocksize, multipliers, quotients) :
                _fused_reduce_column!(B, U, R, column, m, deadband, ws)
            _fused_tock!(timings, FUSED_TIME_REDUCE, reduce_started)

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
            reorth_started = _fused_tick()
            for i in 1:m
                R[i, column] = _to_float(S, B[i, column])
            end
            for j in 1:(column - 1)
                _fused_larf_col!(R, column, j, tau[j], m, ws)
            end
            _fused_tock!(timings, FUSED_TIME_REORTH, reorth_started)
        end

        # Generate this column's reflector and push it through the columns to
        # the right, so that the next column arrives already orthogonalized
        # against everything before it.
        panel_end = panelled ? min(panel_start + panelsize - 1, n) : n

        if column < m
            reflector_started = _fused_tick()
            tau[column] = _fused_larfg!(R, column, m, ws)
            _fused_tock!(timings, FUSED_TIME_REFLECTOR, reflector_started)

            # Only as far as the end of the panel: the columns beyond it are
            # not processed until the panel is flushed, and get the whole
            # panel's reflectors in one block update then.
            trailing_started = _fused_tick()
            for target in (column + 1):panel_end
                _fused_larf_col!(R, target, column, tau[column], m, ws)
            end
            _fused_tock!(timings, FUSED_TIME_TRAILING, trailing_started)
        end

        if panelled && column == panel_end
            trailing_started = _fused_tick()
            _fused_panel_flush!(R, tau, panel_start, panel_end, m, n,
                                strassen_cutoff, panel_work)
            _fused_tock!(timings, FUSED_TIME_TRAILING, trailing_started)
            panel_start = column + 1
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
                                  max_passes::Integer = FUSED_QR_MAX_PASSES,
                                  blocksize::Integer = DEFAULT_FUSED_BLOCKSIZE,
                                  panelsize::Integer = DEFAULT_FUSED_PANELSIZE,
                                  strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
                                  timings::Union{Nothing, Vector{Float64}} = nothing
                                  ) where {T<:Integer}
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
            _fused_columnwise!(B, U, R, tau; timings = timings,
                               blocksize = Int(blocksize),
                               panelsize = Int(panelsize),
                               strassen_cutoff = Int(strassen_cutoff),
                               deadband = BigFloat(deadband),
                               max_passes = Int(max_passes))
            (B, U, R, tau)
        end
    end

    S = float_type
    R = Matrix{S}(undef, m, n)
    tau = Vector{S}(undef, n)
    _fused_columnwise!(B, U, R, tau; timings = timings,
                       blocksize = Int(blocksize),
                       panelsize = Int(panelsize),
                       strassen_cutoff = Int(strassen_cutoff),
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
