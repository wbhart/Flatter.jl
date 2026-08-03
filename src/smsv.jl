# smsv.jl
#
# Column scaling for LLL reduction of a floating-point lattice basis.
#
# This is Algorithm 2 of
#
#     Saruchi, I. Morel, D. Stehle, G. Villard,
#     "LLL reducing with the most significant bits", ISSAC 2014.
#
# following the implementation in `recursive_generic.cpp` of the `flatter`
# lattice reduction library (K. Ryan, N. Heninger, "Fast Practical Lattice
# Reduction through Iterated Compression", CRYPTO 2023).
#
# The input is a lattice basis in columns-as-vectors form, represented as a
# columnwise floating-point matrix
#
#     B = M_B * E_B,    M_B integral,    E_B = diag(2^e_j).
#
# The algorithm computes a diagonal scaling D = diag(2^d_i) with d
# non-decreasing, such that
#
#   * the integer matrix C = M_B E_B D^-1 has entries of small bit-size, and
#   * any transformation U that LLL-reduces C yields a transformation
#     D^-1 U D that is unimodular and reduces B itself.
#
# Note on the correspondence with the paper: `recursive_generic.cpp` fuses
# Algorithm 2's per-column block scaling with the uniform truncation of
# Algorithm 1 into a single shift vector. That fusion is reproduced here, but
# the two components are returned separately as `shifts` (Algorithm 2's D) and
# `truncation` (Algorithm 1's rounding), so callers can distinguish them.

using LinearAlgebra: I

# --------------------------------------------------------------------------
# Representation
# --------------------------------------------------------------------------

"""
    ScaledBasis(mantissa, exponent)

A columnwise floating-point lattice basis in columns-as-vectors form. Column
`j` of the represented basis is `mantissa[:, j] * 2^exponent[j]`; that is, the
basis is `mantissa * Diagonal(2 .^ exponent)`.

`mantissa` is `m x n` with `m >= n`, and its columns must be linearly
independent.
"""
struct ScaledBasis
    mantissa::Matrix{BigInt}
    exponent::Vector{Int}

    function ScaledBasis(mantissa::AbstractMatrix{<:Integer},
                         exponent::AbstractVector{<:Integer})
        m, n = size(mantissa)
        m >= n > 0 ||
            throw(ArgumentError("expected an m x n mantissa with m >= n >= 1, got $m x $n"))
        length(exponent) == n ||
            throw(ArgumentError("expected $n column exponents, got $(length(exponent))"))
        return new(Matrix{BigInt}(mantissa), Vector{Int}(exponent))
    end
end

"""
    ScaledBasis(mantissa)

An integral basis, with every column exponent equal to zero.
"""
ScaledBasis(mantissa::AbstractMatrix{<:Integer}) =
    ScaledBasis(mantissa, zeros(Int, size(mantissa, 2)))

Base.size(basis::ScaledBasis) = size(basis.mantissa)
Base.size(basis::ScaledBasis, dimension::Integer) = size(basis.mantissa, dimension)

"""
    is_integral(basis) -> Bool

Whether every column exponent is nonnegative, so that the represented basis has
integer entries.
"""
is_integral(basis::ScaledBasis) = all(>=(0), basis.exponent)

"""
    integer_matrix(basis) -> Matrix{BigInt}

The represented basis as an exact integer matrix. Requires `is_integral(basis)`.
"""
function integer_matrix(basis::ScaledBasis)
    is_integral(basis) ||
        throw(ArgumentError("the basis has negative column exponents and is not integral"))
    m, n = size(basis)
    result = Matrix{BigInt}(undef, m, n)
    for j in 1:n
        shift = basis.exponent[j]
        for i in 1:m
            result[i, j] = basis.mantissa[i, j] << shift
        end
    end
    return result
end

"""
    float_matrix(basis, precision) -> Matrix{BigFloat}

The represented basis rounded to `BigFloat` at `precision` bits. Each column is
scaled by its exponent exactly; the only rounding is in the mantissa entries.
"""
function float_matrix(basis::ScaledBasis, precision::Integer)
    precision >= 2 || throw(ArgumentError("precision must be at least 2 bits"))
    m, n = size(basis)
    result = Matrix{BigFloat}(undef, m, n)
    setprecision(BigFloat, Int(precision)) do
        for j in 1:n, i in 1:m
            result[i, j] = ldexp(BigFloat(basis.mantissa[i, j]), basis.exponent[j])
        end
    end
    return result
end

"""
    apply_transform(basis, V) -> ScaledBasis

Apply the column transformation `V` to `basis`, returning `basis * V` exactly.

The result has a uniform column exponent, equal to the minimum exponent of the
input, because `Diagonal(2^e) * V` is integral once the common factor
`2^min(e)` is pulled out.
"""
function apply_transform(basis::ScaledBasis, V::AbstractMatrix{<:Integer})
    m, n = size(basis)
    size(V) == (n, n) ||
        throw(DimensionMismatch("expected a $n x $n transformation, got $(size(V))"))
    base = minimum(basis.exponent)
    lifted = Matrix{BigInt}(undef, n, n)
    for j in 1:n, i in 1:n
        lifted[i, j] = BigInt(V[i, j]) << (basis.exponent[i] - base)
    end
    return ScaledBasis(basis.mantissa * lifted, fill(base, n))
end

# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

# log2|x| for a BigFloat, via the significand/exponent split rather than a
# direct logarithm. This mirrors flatter's use of mpfr_get_d_2exp and keeps the
# result in Float64 range even when x is astronomically large or small.
function _log2_abs(x::BigFloat)
    iszero(x) && return -Inf
    significand, exponent = frexp(x)
    return Float64(exponent) + log2(abs(Float64(significand)))
end

# Per-column power-of-two normalisers, so the matrix handed to the QR has
# balanced columns. Since right multiplication by a positive diagonal commutes
# through QR (B = QR implies BD = Q(RD), which is the R-factor by uniqueness),
# the exponents can be added back to the profile afterwards, exactly.
function _column_normalisers(mantissa::Matrix{BigInt})
    m, n = size(mantissa)
    result = zeros(Int, n)
    for j in 1:n
        largest = 0
        for i in 1:m
            value = mantissa[i, j]
            iszero(value) || (largest = max(largest, ndigits(value; base=2)))
        end
        result[j] = largest
    end
    return result
end

"""
    profile(basis, precision) -> Vector{Float64}

Steps 1-2 of Algorithm 2: run Householder's algorithm on the represented basis
at `precision` bits and return `log2 |r_ii|` for each `i`, where `R` is the
computed R-factor.

Only the diagonal of `R` is used, so the compact-WY factor is discarded.
Throws if the computed R-factor is singular, since Algorithm 2 assumes the
input has full column rank.

The `blocked' parameter chooses whether the Householder-QR attempts to use a
blocked implementation (with Strassen) for n^omega with omega < 3 complexity.
"""
function profile(basis::ScaledBasis, precision::Integer; blocked::Bool=true)
    precision >= 2 || throw(ArgumentError("precision must be at least 2 bits"))
    m, n = size(basis)
    normalisers = _column_normalisers(basis.mantissa)

    balanced = Matrix{BigFloat}(undef, m, n)
    factors = setprecision(BigFloat, Int(precision)) do
        for j in 1:n, i in 1:m
            balanced[i, j] = ldexp(BigFloat(basis.mantissa[i, j]), -normalisers[j])
        end
        (blocked ? householder_block(balanced) : householder(balanced)).factors
    end

    result = Vector{Float64}(undef, n)
    for i in 1:n
        value = _log2_abs(factors[i, i])
        isfinite(value) || throw(ArgumentError(
            "the R-factor has a zero at diagonal position $i; the basis is " *
            "rank deficient, or the precision of $precision bits is too low"))
        result[i] = value + normalisers[i] + basis.exponent[i]
    end
    return result
end

"""
    profile_spread(prof) -> Float64

The difference between the largest and smallest entries of a profile. This is
flatter's stand-in for `log2 cond(R)`; see [`householder_precision`](@ref).
"""
profile_spread(prof::AbstractVector{<:Real}) = Float64(maximum(prof) - minimum(prof))

# --------------------------------------------------------------------------
# Precision policies
# --------------------------------------------------------------------------

"""
    householder_precision(m, log2_cond) -> Int

Step 1 of Algorithm 2: the precision `p = 10 + ceil(log2(m^3.5 * chi))` at which
the R-factor of the input is estimated, where `log2_cond >= log2 chi` bounds
`log2 cond(R)`.
"""
function householder_precision(m::Integer, log2_cond::Real)
    isfinite(log2_cond) || throw(ArgumentError("log2_cond must be finite"))
    return max(53, 10 + ceil(Int, 3.5 * log2(max(m, 1)) + max(log2_cond, 0.0)))
end

"""
    lll_precision(spread, n; aggressive=false) -> Int

The working precision to truncate the scaled basis to before handing it to an
LLL reduction, as a function of the profile spread.

Theorem 1 of the paper gives `2 log2 cond(R) + n(1 + eps) + O(log m)` bits; the
default here is flatter's non-aggressive form `2 * spread + 30 + 2n`, with the
profile spread substituted for `log2 cond(R)`. With `aggressive=true` this drops
to `spread + 30`, matching flatter's `FLATTER_AGGRESSIVE_PREC` setting.

Note that substituting the spread for `log2 cond(R)` is a heuristic. Lemma 12 of
the paper justifies it for size-reduced upper triangular bases with balanced
diagonal, not in general.
"""
function lll_precision(spread::Real, n::Integer; aggressive::Bool=false)
    isfinite(spread) || throw(ArgumentError("spread must be finite"))
    bounded = max(Float64(spread), 0.0)
    return aggressive ? ceil(Int, bounded) + 30 :
                        ceil(Int, 2 * bounded) + 30 + 2 * Int(n)
end

# A pessimistic standing estimate of log2 cond(R), used when the caller supplies
# no bound. This follows flatter's `get_initial_precision`, which uses the
# largest entry bit-length, widened here by the spread of the column exponents.
function _estimate_log2_cond(basis::ScaledBasis)
    largest = 0
    for value in basis.mantissa
        iszero(value) && continue
        largest = max(largest, ndigits(value; base=2))
    end
    exponent_spread = isempty(basis.exponent) ? 0 :
                      maximum(basis.exponent) - minimum(basis.exponent)
    return Float64(largest + exponent_spread)
end

# --------------------------------------------------------------------------
# The scaling itself
# --------------------------------------------------------------------------

"""
    compression_shifts(prof, precision) -> (shifts, truncation, spread)

Steps 3-7 of Algorithm 2, in the fused form used by flatter's
`get_shifts_for_compression`.

Given a profile, return

  * `shifts`, the per-column exponents `d_i` of the scaling `D = diag(2^d_i)`,
    before truncation, as a non-decreasing integer vector starting at zero;
  * `truncation`, the additional uniform shift that brings the largest column
    down to `precision` bits (this is Algorithm 1's rounding, not part of
    Algorithm 2);
  * `spread`, the profile spread that remains after scaling.

Blocks are never formed explicitly. The paper's block boundary test is
`min_{j>=i} r_jj > (8/theta) max_{j<i} r_jj`; here the running maximum and
minimum are compared directly, with the threshold relaxed to a single bit and
the increment taken as `floor(log2(g/2))` rather than `floor(log2(g/4))`. A
column that fails the test simply inherits the previous shift, so `shifts` is
constant within a block and jumps at boundaries.
"""
function compression_shifts(prof::AbstractVector{<:Real}, precision::Integer)
    n = length(prof)
    n > 0 || throw(ArgumentError("the profile must be non-empty"))
    all(isfinite, prof) || throw(ArgumentError("the profile must be finite"))

    maximum_from_left = Vector{Float64}(undef, n)
    minimum_from_right = Vector{Float64}(undef, n)
    maximum_from_left[1] = prof[1]
    minimum_from_right[n] = prof[n]
    for i in 2:n
        maximum_from_left[i] = max(Float64(prof[i]), maximum_from_left[i - 1])
    end
    for i in (n - 1):-1:1
        minimum_from_right[i] = min(Float64(prof[i]), minimum_from_right[i + 1])
    end

    shifts = zeros(Int, n)
    for i in 2:n
        shifts[i] = shifts[i - 1]
        gap = minimum_from_right[i] - maximum_from_left[i - 1]
        gap <= 1 && continue
        shifts[i] += floor(Int, gap - 1)
    end

    spread = maximum_from_left[n] - shifts[n] - minimum_from_right[1]
    truncation = ceil(Int, maximum_from_left[n]) - shifts[n] - Int(precision)
    return shifts, truncation, spread
end

"""
    blocks(shifts) -> Vector{UnitRange{Int}}

The index blocks `I_l` implied by a shift vector: maximal runs of columns
carrying the same shift.
"""
function blocks(shifts::AbstractVector{<:Integer})
    n = length(shifts)
    result = UnitRange{Int}[]
    start = 1
    for i in 2:n
        if shifts[i] != shifts[i - 1]
            push!(result, start:(i - 1))
            start = i
        end
    end
    push!(result, start:n)
    return result
end

# C = M_B E_B D^-1, converted to an integer matrix. Column j is shifted by
# exponent[j] - shifts[j]; a negative shift floors, matching flatter's use of
# mpz_div_2exp (which is mpz_fdiv_q_2exp, as is Julia's >> on BigInt).
function _scale_to_integer(basis::ScaledBasis, shifts::Vector{Int})
    m, n = size(basis)
    result = Matrix{BigInt}(undef, m, n)
    for j in 1:n
        shift = basis.exponent[j] - shifts[j]
        if shift >= 0
            for i in 1:m
                result[i, j] = basis.mantissa[i, j] << shift
            end
        else
            for i in 1:m
                result[i, j] = basis.mantissa[i, j] >> (-shift)
            end
        end
    end
    return result
end

"""
    conjugate_transform(U, shifts) -> Matrix{BigInt}

Map a transformation `U` for the scaled matrix to the transformation
`D^-1 U D` for the unscaled one, where `D = diag(2^shifts[i])`.

Entry `(i, j)` is multiplied by `2^(shifts[j] - shifts[i])`. Theorem 2 of the
paper shows that `U` is block-upper-triangular with respect to the blocks of
`shifts`, which is exactly what makes the result integral; this function
verifies that property rather than assuming it, and throws if it fails.
"""
function conjugate_transform(U::AbstractMatrix{<:Integer}, shifts::AbstractVector{<:Integer})
    n = size(U, 1)
    size(U, 2) == n ||
        throw(DimensionMismatch("expected a square transformation, got $(size(U))"))
    length(shifts) == n ||
        throw(DimensionMismatch("expected $n shifts, got $(length(shifts))"))
    issorted(shifts) ||
        throw(ArgumentError("the shifts must be non-decreasing for the conjugation to be integral"))

    result = Matrix{BigInt}(undef, n, n)
    for j in 1:n, i in 1:n
        entry = BigInt(U[i, j])
        if shifts[i] == shifts[j]
            result[i, j] = entry
        elseif shifts[i] > shifts[j]
            iszero(entry) || throw(ArgumentError(
                "the LLL reduction mixed columns across a scaling block: U[$i, $j] " *
                "is nonzero but the conjugation would not be integral. The reducer " *
                "is not well-behaved in the sense of the paper, or the block gaps " *
                "are too small."))
            result[i, j] = entry
        else
            result[i, j] = entry << (shifts[j] - shifts[i])
        end
    end
    return result
end

# --------------------------------------------------------------------------
# LLL adapters
# --------------------------------------------------------------------------

"""
    fplll_reduce(A; delta=0.99, eta=0.51) -> (basis, transform)

Column-convention adapter around `FPLLL.lll`. Takes an integer matrix whose
columns are basis vectors and returns the reduced basis together with `U`
satisfying `basis == A * U`.

`FPLLL` uses vectors-as-rows and left multiplication, so both the basis and the
transformation are transposed on the way through. The `FPLLL` module must be
available at the call site; otherwise pass your own reducer to
[`scale_for_reduction`](@ref) via the `reducer` keyword.
"""
function fplll_reduce(A::AbstractMatrix{<:Integer}; delta::Real=0.99, eta::Real=0.51)
    result = FPLLL.lll(permutedims(Matrix{BigInt}(A)); transform=true, delta=delta, eta=eta)
    result.transform === nothing &&
        throw(ArgumentError("the reducer returned no transformation matrix"))
    return permutedims(result.basis), permutedims(result.transform)
end

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

"""
    ScalingResult

The output of [`scale_for_reduction`](@ref).

Fields:

  * `transform`     -- `D^-1 U D`, the transformation valid for the input basis.
  * `inner`         -- `U`, the transformation the reducer produced for `scaled`.
  * `scaled`        -- `C = M_B E_B D^-1`, truncated to an integer matrix.
  * `reduced`       -- the reduced form of `scaled`, that is `scaled * inner`.
  * `shifts`        -- the exponents of `D`, non-decreasing, starting at zero.
  * `truncation`    -- the uniform truncation folded into the applied scaling.
  * `applied`       -- `shifts .+ truncation`, what was actually divided out.
  * `blocks`        -- the index blocks implied by `shifts`.
  * `profile`       -- `log2 |r_ii|` of the input basis.
  * `spread`        -- the profile spread remaining after scaling.
  * `qr_precision`  -- the precision used for the Householder call.
  * `bits`          -- the precision the scaled matrix was truncated to.
"""
struct ScalingResult
    transform::Matrix{BigInt}
    inner::Matrix{BigInt}
    scaled::Matrix{BigInt}
    reduced::Matrix{BigInt}
    shifts::Vector{Int}
    truncation::Int
    applied::Vector{Int}
    blocks::Vector{UnitRange{Int}}
    profile::Vector{Float64}
    spread::Float64
    qr_precision::Int
    bits::Int
end

"""
    scale_for_reduction(basis; kwargs...) -> ScalingResult
    scale_for_reduction(mantissa, exponent; kwargs...) -> ScalingResult
    scale_for_reduction(mantissa; kwargs...) -> ScalingResult

Algorithm 2 of Saruchi-Morel-Stehle-Villard: LLL-reduce a columnwise
floating-point lattice basis by column scaling.

The basis is given in columns-as-vectors form, either as a [`ScaledBasis`](@ref)
or as a mantissa matrix with optional column exponents.

Keyword arguments:

  * `reducer`      -- the LLL routine to call on the scaled matrix. It receives
                      an integer matrix in column convention and must return
                      `(basis, transform)` with `basis == input * transform`.
                      Defaults to [`fplll_reduce`](@ref).
  * `log2_cond`    -- a bound on `log2 cond(R)` for the input basis, the paper's
                      `log2 chi`. Estimated pessimistically if omitted.
  * `qr_precision` -- override the precision of the Householder call.
  * `bits`         -- override the precision the scaled matrix is truncated to.
  * `aggressive`   -- use flatter's aggressive precision policy. Ignored if
                      `bits` is given.
  * `check`        -- verify unimodularity of the returned transformation.
                      On by default; the check is cubic in `n`.

The returned `transform` is unimodular, and `basis * transform` is reduced.
"""
function scale_for_reduction(basis::ScaledBasis;
                             reducer=fplll_reduce,
                             log2_cond::Union{Nothing, Real}=nothing,
                             qr_precision::Union{Nothing, Integer}=nothing,
                             bits::Union{Nothing, Integer}=nothing,
                             aggressive::Bool=false,
                             check::Bool=true)
    m, n = size(basis)

    # Steps 1-2: estimate the diagonal of the R-factor.
    condition = log2_cond === nothing ? _estimate_log2_cond(basis) : Float64(log2_cond)
    qr_bits = qr_precision === nothing ? householder_precision(m, condition) :
                                         Int(qr_precision)
    prof = profile(basis, qr_bits)

    # Steps 3-7: block detection and scaling, fused with Algorithm 1's rounding.
    # The target precision depends on the spread, which depends on the shifts,
    # so the shifts are computed once with a provisional precision and the
    # truncation is then recomputed from the resulting spread.
    provisional = bits === nothing ?
        lll_precision(profile_spread(prof), n; aggressive=aggressive) : Int(bits)
    shifts, _, spread = compression_shifts(prof, provisional)
    target = bits === nothing ? lll_precision(spread, n; aggressive=aggressive) : Int(bits)
    shifts, truncation, spread = compression_shifts(prof, target)

    applied = shifts .+ truncation
    scaled = _scale_to_integer(basis, applied)

    for j in 1:n
        all(iszero, view(scaled, :, j)) && throw(ArgumentError(
            "column $j of the scaled basis truncated to zero at $target bits; " *
            "increase `bits`, or supply a tighter `log2_cond`"))
    end

    # Step 9: reduce the scaled matrix.
    reduced, inner = reducer(scaled)
    size(inner) == (n, n) || throw(DimensionMismatch(
        "the reducer returned a $(size(inner)) transformation, expected $n x $n"))

    # Step 10, and the D^-1 U D map of Theorem 2. Note that the conjugation uses
    # `shifts`, not `applied`: the uniform truncation commutes with U and cancels.
    transform = conjugate_transform(inner, shifts)

    if check
        determinant = _unimodular_determinant(transform)
        isone(abs(determinant)) || throw(ArgumentError(
            "the conjugated transformation has determinant $determinant and is " *
            "not unimodular"))
    end

    return ScalingResult(transform, Matrix{BigInt}(inner), scaled,
                         Matrix{BigInt}(reduced), shifts, truncation, applied,
                         blocks(shifts), prof, spread, qr_bits, target)
end

scale_for_reduction(mantissa::AbstractMatrix{<:Integer},
                    exponent::AbstractVector{<:Integer}; kwargs...) =
    scale_for_reduction(ScaledBasis(mantissa, exponent); kwargs...)

scale_for_reduction(mantissa::AbstractMatrix{<:Integer}; kwargs...) =
    scale_for_reduction(ScaledBasis(mantissa); kwargs...)

# Fraction-free Bareiss determinant. Exact over the integers and free of the
# division that a generic LU would need.
function _unimodular_determinant(A::AbstractMatrix{<:Integer})
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("a determinant needs a square matrix"))
    n == 0 && return BigInt(1)
    work = Matrix{BigInt}(A)
    sign = 1
    previous = BigInt(1)
    for k in 1:(n - 1)
        if iszero(work[k, k])
            pivot = findfirst(i -> !iszero(work[i, k]), (k + 1):n)
            pivot === nothing && return BigInt(0)
            row = k + pivot
            for j in 1:n
                work[k, j], work[row, j] = work[row, j], work[k, j]
            end
            sign = -sign
        end
        for i in (k + 1):n, j in (k + 1):n
            work[i, j] = (work[i, j] * work[k, k] - work[i, k] * work[k, j]) ÷ previous
        end
        previous = work[k, k]
    end
    return sign * work[n, n]
end
