# relative_size_reduction.jl
#
# Size reduction of one block of lattice vectors against another.
#
# Given B1 (m x n1), whose columns are the reference block, and B2 (m x n2),
# find an integer U (n1 x n2) such that
#
#     B2 <- B2 + B1*U
#
# leaves every column of B2 size reduced against B1's Gram-Schmidt frame: the
# coefficient of b*_j in each column of B2 is at most 1/2 in absolute value, for
# every j <= n1. Columns of B2 are never reduced against each other.
#
# Note the sign convention: U accumulates -c, so this is an addition of B1*U,
# not a subtraction.
#
# This is the counterpart of `size_reduction_triu.jl`, which reduces a single
# triangular block against itself. Three kernels, mirroring flatter's
# `problems/relative_size_reduction`:
#
#   :triangular  B1 is square upper triangular, so it is its own R factor and
#                the whole computation is exact integer arithmetic
#                (triangular.cpp)
#   :orthogonal  a precomputed floating point QR of B1 is supplied; multipliers
#                come from approximate coordinates, corrected by iterative
#                refinement (orthogonal.cpp and orthogonal_double.cpp)
#   :generic     no factorization supplied, so one is computed and the work
#                delegated to :orthogonal (generic.cpp)
#
# flatter splits the orthogonal case into two classes, `Orthogonal` (MPFR) and
# `OrthogonalDouble` (double), because C++ needs distinct types. In Julia one
# kernel parameterised on the float type covers both; pass Float64 factors to
# get the hardware path and BigFloat factors to get the arbitrary precision one.
#
# See Notes.md for the two deliberate divergences from flatter here: the
# refinement loop is implemented for the double path (flatter's is dead code),
# and BigFloat-times-BigInt rounds twice where MPFR rounds once.

# flatter's threshold, slightly wider than 1/2. An entry whose coordinate sits
# exactly at the boundary would otherwise be corrected back and forth between
# passes on rounding noise alone. Costs a little reduction quality and nothing
# else, since any integer multiplier gives a valid unimodular update.
const RELATIVE_SIZE_REDUCTION_DEADBAND = 0.51

# Safety cap on refinement passes. The convergence guard below should stop long
# before this; the cap exists so that a pathological input fails loudly rather
# than hanging.
const RELATIVE_SIZE_REDUCTION_MAX_PASSES = 64

# ---------------------------------------------------------------------------
# Conversion of exact columns to floating point
# ---------------------------------------------------------------------------

# BigFloat has an enormous exponent range, so an integer of any lattice size
# converts without overflow; only the mantissa rounds.
_to_float(::Type{BigFloat}, x::Integer) = BigFloat(x)

# Float64 does not. Converting a 2^2000-sized entry silently yields Inf, which
# would then propagate as NaN through the first division. Fail loudly instead:
# the double kernel is for inputs whose entries fit, exactly as in flatter,
# where `Matrix::copy(r_col, b_col)` has the same limitation.
function _to_float(::Type{S}, x::Integer) where {S<:AbstractFloat}
    value = S(x)
    isfinite(value) || throw(ArgumentError(
        "an entry of B2 has $(ndigits(x; base = 2)) bits and does not fit in " *
        "$S; use a BigFloat factorization for entries this large"))
    return value
end

# ---------------------------------------------------------------------------
# Argument checking
# ---------------------------------------------------------------------------

function _check_relative_arguments(B1::AbstractMatrix, B2::AbstractMatrix,
                                   U::AbstractMatrix)
    m, n1 = size(B1)
    size(B2, 1) == m || throw(DimensionMismatch(
        "B1 has $m rows but B2 has $(size(B2, 1))"))
    size(U) == (n1, size(B2, 2)) || throw(DimensionMismatch(
        "U must be $n1 x $(size(B2, 2)), got $(size(U))"))
    return m, n1, size(B2, 2)
end

function _check_triangular_reference(B1::AbstractMatrix)
    m, n1 = size(B1)
    m == n1 || throw(DimensionMismatch(
        "the triangular kernel needs a square B1, got $m x $n1"))
    for j in 1:n1
        iszero(B1[j, j]) && throw(ArgumentError(
            "B1 has a zero diagonal entry at index $j"))
        for i in (j + 1):m
            iszero(B1[i, j]) || throw(ArgumentError(
                "B1 must be upper triangular; B1[$i,$j] is nonzero"))
        end
    end
    return nothing
end

_zero_transform!(U::AbstractMatrix{T}) where {T} =
    (for index in eachindex(U); U[index] = zero(T); end; U)

# ---------------------------------------------------------------------------
# Triangular kernel
# ---------------------------------------------------------------------------

# When B1 is upper triangular it is its own R factor, so the coordinates of a
# column of B2 in B1's Gram-Schmidt frame are just that column's entries: for
# upper triangular B1 the orthogonalised vectors are b*_j = B1[j,j] * e_j. The
# multipliers are therefore exact rational roundings and no floating point
# appears anywhere.
#
#     for each column j of B2:
#         for row = n1 down to 1:
#             c <- round(B2[row,j] / B1[row,row])
#             U[row,j] <- -c
#             B2[1:row, j] -= c * B1[1:row, row]
#             rescale B2[row,j] by new_shift
#
# The rescaling is applied per entry, immediately after that entry stops
# changing: reductions at row r only touch rows <= r, so each entry is shifted
# exactly once and the still-unshifted entries above are what later iterations
# consume.
function _relative_triangular!(B1::AbstractMatrix{T}, B2::AbstractMatrix{T},
                               U::AbstractMatrix{T}, new_shift::Int) where {T<:Integer}
    n1 = size(B1, 1)
    n2 = size(B2, 2)

    _zero_transform!(U)

    @inbounds for j in 1:n2
        for row in n1:-1:1
            c = _div_round(B2[row, j], B1[row, row])
            if !iszero(c)
                U[row, j] = -c
                for k in 1:row
                    B2[k, j] -= c * B1[k, row]
                end
            end

            # Floor throughout, per Notes.md: flatter uses a truncating shift
            # here and a flooring one in the compression path, and this package
            # uses one convention everywhere.
            if new_shift < 0
                B2[row, j] <<= -new_shift
            elseif new_shift > 0
                B2[row, j] >>= new_shift
            end
        end
    end

    return B2, U
end

# ---------------------------------------------------------------------------
# Orthogonal kernel
# ---------------------------------------------------------------------------

# Reduce one column, whose approximate coordinates in B1's frame are in `r`.
# Returns the largest multiplier exponent seen, or typemin(Int) if no multiplier
# cleared the deadband.
#
# `r` is floating point and approximate; B2 and U are exact and are updated with
# the rounded integer multiplier. That split is what makes refinement work: the
# exact quantities never accumulate float error, so recomputing `r` from B2 at
# the start of the next pass discards the error entirely rather than compounding
# it.
function _reduce_relative_column!(r::AbstractVector{S},
                                  factors::AbstractMatrix{S},
                                  B1::AbstractMatrix{T},
                                  B2::AbstractMatrix{T},
                                  U::AbstractMatrix{T},
                                  column::Int, n1::Int,
                                  deadband::Float64) where {S<:AbstractFloat, T<:Integer}
    m = size(B1, 1)
    largest = typemin(Int)

    @inbounds for row in n1:-1:1
        quotient = r[row] / factors[row, row]

        # Inside the deadband there is nothing worth correcting, and `largest`
        # is deliberately not updated: the convergence test below asks how big
        # the multipliers we actually applied were.
        (-deadband < quotient < deadband) && continue

        largest = max(largest, safe_exponent(quotient))

        rounded = round(quotient)
        multiplier = T(rounded)

        # Approximate update of the coordinates, using the rounded float. This
        # is where flatter would call mpfr_mul_z for a single rounding; see
        # Notes.md.
        for k in 1:row
            r[k] -= rounded * factors[k, row]
        end

        # Exact updates.
        for k in 1:m
            B2[k, column] -= multiplier * B1[k, row]
        end
        U[row, column] -= multiplier
    end

    return largest
end

# The refinement loop.
#
#     until every column has converged:
#         recompute the coordinates of the active columns from the EXACT B2
#         apply Q' to get coordinates in B1's frame
#         reduce each active column, recording the largest multiplier applied
#         a column stops when its multipliers are below unit magnitude, or when
#         they stop shrinking
#
# The second stopping rule is load-bearing, not a belt-and-braces duplicate of
# the first: at low precision a badly conditioned block will keep producing
# unit-or-larger multipliers forever, and only the "no longer shrinking" test
# terminates it. flatter's MPFR kernel has both; its double kernel has neither
# in practice, since the variable driving them is never assigned. See Notes.md.
#
# Columns are processed as a block rather than one at a time as flatter does,
# because applying Q' to many columns at once is a matrix product rather than a
# sequence of matrix-vector products. Converged columns are dropped from the
# block so that the extra passes needed by a slow column are not paid for by
# every column.
function _relative_orthogonal!(B1::AbstractMatrix{T}, B2::AbstractMatrix{T},
                               U::AbstractMatrix{T},
                               factors::AbstractMatrix{S},
                               compact_T::AbstractMatrix{S},
                               R2::Union{Nothing, AbstractMatrix{S}};
                               deadband::Float64,
                               strassen_cutoff::Int,
                               max_passes::Int) where {T<:Integer, S<:AbstractFloat}
    m, n1 = size(B1)
    n2 = size(B2, 2)

    _zero_transform!(U)
    (iszero(n1) || iszero(n2)) && return B2, U

    # Every multiplier is a quotient by a diagonal entry of R, so a rank
    # deficient reference block would produce Inf and then NaN. Fail here,
    # where the cause is still visible.
    for j in 1:n1
        isfinite(factors[j, j]) && !iszero(factors[j, j]) || throw(ArgumentError(
            "the R factor of B1 has a zero or non-finite diagonal entry at " *
            "index $j; B1 is rank deficient, or the factorization precision " *
            "is too low"))
    end

    coordinates = Matrix{S}(undef, m, n2)
    reflector_work = Matrix{S}(undef, n1, n2)
    multiply_work = _householder_block_multiply_workspace(
        S, max(0, min(n1, m - n1, n2)), strassen_cutoff)

    active = collect(1:n2)
    previous = fill(typemax(Int), n2)
    passes = 0

    while !isempty(active)
        passes += 1
        passes <= max_passes || throw(ErrorException(
            "relative size reduction failed to converge in $max_passes passes; " *
            "the working precision is too low for this block"))

        # Recompute from the exact columns. This is what stops float error
        # accumulating across passes.
        for (slot, column) in enumerate(active)
            for i in 1:m
                coordinates[i, slot] = _to_float(S, B2[i, column])
            end
        end

        block = view(coordinates, :, 1:length(active))
        apply_Qt!(block, factors, compact_T, n1,
                  reflector_work, multiply_work, strassen_cutoff)

        still_active = Int[]
        for (slot, column) in enumerate(active)
            largest = _reduce_relative_column!(
                view(block, :, slot), factors, B1, B2, U, column, n1, deadband)

            if R2 !== nothing
                for i in 1:m
                    R2[i, column] = block[i, slot]
                end
            end

            # Julia's `exponent` gives e with |x| in [2^e, 2^(e+1)); MPFR's
            # mpfr_get_exp gives that plus one. flatter's `max_mu_size <= 1` is
            # therefore `largest <= 0` here: stop once every multiplier applied
            # was below unit magnitude.
            largest <= 0 && continue
            largest >= previous[column] && continue

            previous[column] = largest
            push!(still_active, column)
        end
        active = still_active
    end

    return B2, U
end

# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------

"""
    relative_size_reduction_precision(B1) -> Int

The default working precision for a factorization of `B1`, from the package's
existing policy: [`householder_precision`](@ref) applied to the row count, with
the largest entry bit-length standing in for `log2 cond(R)`.

This mirrors flatter's `get_initial_precision`, and is pessimistic. Supply a
`precision` explicitly when a better bound is known; the refinement loop will
cope either way, but a tighter precision costs less per pass.
"""
function relative_size_reduction_precision(B1::AbstractMatrix{<:Integer})
    largest = 0
    for value in B1
        iszero(value) && continue
        largest = max(largest, ndigits(value; base = 2))
    end
    return householder_precision(size(B1, 1), Float64(largest))
end

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

"""
    relative_size_reduction!(B1, B2, U; kwargs...) -> (B2, U, R2)

Size reduce the columns of `B2` against the block `B1`, in place. Overwrites
`U` with the integer transform, so that on return the new `B2` equals the old
`B2` plus `B1*U`. Returns `(B2, U, R2)`.

`B1` is `m x n1`, `B2` is `m x n2` and `U` is `n1 x n2`, all with the same
integer element type. Columns of `B2` are reduced against `B1` only, never
against each other.

`R2` is the coordinate matrix of the reduced `B2` in `B1`'s orthogonal frame:
its first `n1` rows are the size reduced coordinates and its remaining rows are
the component orthogonal to `B1`'s span, which is what a recursive driver hands
to the next level. It is `nothing` when `coordinates = false`.

Keyword arguments:

  * `algorithm`       -- `:auto`, `:triangular`, `:orthogonal` or `:generic`.
  * `factors`, `compact_T` -- a packed compact-WY factorization of `B1` from
                        [`householder`](@ref) or [`householder_block`](@ref).
                        Supplying these selects `:orthogonal`; their element
                        type chooses the arithmetic, `Float64` for the hardware
                        path and `BigFloat` for arbitrary precision.
  * `triangular`      -- assert that `B1` is square upper triangular, selecting
                        the exact `:triangular` kernel.
  * `new_shift`       -- a power-of-two rescaling applied to `B2` on output.
                        Supported by `:triangular` only, matching flatter.
  * `precision`       -- working precision for `:generic`. Defaults to
                        [`relative_size_reduction_precision`](@ref).
  * `coordinates`     -- whether to compute `R2`. Default `true`.
  * `deadband`        -- the threshold above which a coordinate is corrected.
                        Default `$(RELATIVE_SIZE_REDUCTION_DEADBAND)`, as in flatter.
  * `strassen_cutoff` -- forwarded to [`apply_Qt!`](@ref).
  * `max_passes`      -- safety cap on refinement passes.

The `:orthogonal` and `:generic` kernels achieve a size reduction bound of
roughly `deadband` rather than exactly `1/2`, because of the deadband and
because refinement stops once the multipliers fall below unit magnitude. Only
`:triangular` is exact, and it achieves `1/2` strictly. The identity
`B2_new == B2_old + B1*U` is exact on every path.

# Precision requirement

Refinement can only reduce as far as the coordinates can be known. Converting
`B2` to a `p`-bit float carries an absolute error of about `max|B2| * 2^-p`,
which `apply_Qt!` preserves, so a multiplier is known only to
`max|B2| * 2^-p / min|R_jj|`. Reduction to the deadband therefore needs

    p > log2( max|B2| / min|R_jj| )

with some margin, where `R` is the factor of `B1`.

The `max|B2|` in that bound is the magnitude *after* reduction, because only
the component of `B2` lying in `B1`'s span can shrink. When `B2` is close to
that span, reduction shrinks it and each pass sees better coordinates than the
last, which is the regime refinement is for. When `B2` has a large component
orthogonal to `B1`, that component sets a floor which no number of passes can
lower, and the requirement is a hard one.

Under-provisioning precision is not an error. The convergence guard notices
that the multipliers have stopped shrinking and returns, exactly as flatter
does; `U` is still exact and unimodular, and only the reduction quality
suffers. Raise the precision if quality matters.
"""
function relative_size_reduction!(B1::AbstractMatrix{T}, B2::AbstractMatrix{T},
                                  U::AbstractMatrix{T};
                                  algorithm::Symbol = :auto,
                                  factors::Union{Nothing, AbstractMatrix{<:AbstractFloat}} = nothing,
                                  compact_T::Union{Nothing, AbstractMatrix{<:AbstractFloat}} = nothing,
                                  triangular::Bool = false,
                                  new_shift::Integer = 0,
                                  precision::Union{Nothing, Integer} = nothing,
                                  coordinates::Bool = true,
                                  deadband::Real = RELATIVE_SIZE_REDUCTION_DEADBAND,
                                  strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
                                  max_passes::Integer = RELATIVE_SIZE_REDUCTION_MAX_PASSES) where {T<:Integer}
    m, n1, n2 = _check_relative_arguments(B1, B2, U)

    (factors === nothing) == (compact_T === nothing) || throw(ArgumentError(
        "supply both `factors` and `compact_T`, or neither"))

    chosen = algorithm
    if chosen === :auto
        chosen = triangular ? :triangular :
                 factors === nothing ? :generic : :orthogonal
    end

    deadband > 0 || throw(ArgumentError("deadband must be positive"))
    max_passes >= 1 || throw(ArgumentError("max_passes must be at least one"))

    if chosen !== :triangular && !iszero(new_shift)
        throw(ArgumentError(
            "new_shift is supported by the :triangular kernel only, matching " *
            "flatter, where the rescaling is applied in triangular.cpp"))
    end

    if chosen === :triangular
        _check_triangular_reference(B1)
        _relative_triangular!(B1, B2, U, Int(new_shift))
        return B2, U, coordinates ? Matrix{T}(B2) : nothing

    elseif chosen === :orthogonal
        factors === nothing && throw(ArgumentError(
            "the :orthogonal kernel needs `factors` and `compact_T`"))
        S = eltype(factors)
        eltype(compact_T) === S || throw(ArgumentError(
            "`factors` and `compact_T` must have the same element type"))
        size(factors, 1) == m || throw(DimensionMismatch(
            "`factors` has $(size(factors, 1)) rows, but B1 has $m"))
        size(factors, 2) >= n1 || throw(DimensionMismatch(
            "`factors` has too few columns for a $n1 column reference block"))

        R2 = coordinates ? Matrix{S}(undef, m, n2) : nothing
        _run_orthogonal!(B1, B2, U, factors, compact_T, R2,
                         Float64(deadband), Int(strassen_cutoff), Int(max_passes))
        return B2, U, R2

    elseif chosen === :generic
        bits = precision === nothing ? relative_size_reduction_precision(B1) :
                                       Int(precision)
        return with_precision(bits) do
            reference = bigfloat_matrix(B1, bits)
            packed, wy = householder_block(reference; strassen_cutoff = Int(strassen_cutoff))
            R2 = coordinates ? Matrix{BigFloat}(undef, m, n2) : nothing
            _relative_orthogonal!(B1, B2, U, packed, wy, R2;
                                  deadband = Float64(deadband),
                                  strassen_cutoff = Int(strassen_cutoff),
                                  max_passes = Int(max_passes))
            (B2, U, R2)
        end
    end

    throw(ArgumentError("unknown algorithm :$algorithm"))
end

# BigFloat work must run inside a setprecision block, because Julia's BigFloat
# arithmetic takes its result precision from the global default rather than from
# the operands; see Notes.md. Float64 work needs no such wrapper.
function _run_orthogonal!(B1, B2, U, factors::AbstractMatrix{BigFloat},
                          compact_T, R2, deadband, strassen_cutoff, max_passes)
    bits = uniform_precision(factors)
    bits === nothing && throw(ArgumentError(
        "`factors` has entries of differing precision; it was probably written " *
        "to outside a setprecision block"))
    assert_precision(bits, compact_T; names = ("compact_T",))

    return with_precision(bits) do
        _relative_orthogonal!(B1, B2, U, factors, compact_T, R2;
                              deadband = deadband,
                              strassen_cutoff = strassen_cutoff,
                              max_passes = max_passes)
    end
end

_run_orthogonal!(B1, B2, U, factors::AbstractMatrix{<:AbstractFloat},
                 compact_T, R2, deadband, strassen_cutoff, max_passes) =
    _relative_orthogonal!(B1, B2, U, factors, compact_T, R2;
                          deadband = deadband,
                          strassen_cutoff = strassen_cutoff,
                          max_passes = max_passes)

"""
    relative_size_reduction(B1, B2; kwargs...) -> (B2_reduced, U, R2)

Non-mutating form of [`relative_size_reduction!`](@ref). `B1` and `B2` are left
untouched.
"""
function relative_size_reduction(B1::AbstractMatrix{T}, B2::AbstractMatrix{T};
                                 kwargs...) where {T<:Integer}
    reduced = Matrix{T}(B2)
    U = Matrix{T}(undef, size(B1, 2), size(B2, 2))
    return relative_size_reduction!(B1, reduced, U; kwargs...)
end
