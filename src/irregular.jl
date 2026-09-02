# irregular.jl
#
# Reduction of a basis that is not already square upper triangular.
#
# `lattice_reduce!` requires square, upper triangular, nonsingular input, which
# is what the recursion produces internally but not what a caller usually has.
# This lifts that restriction, following flatter's
# `LatticeReductionImpl::Irregular` (problems/lattice_reduction/irregular.cpp):
#
#   * a basis that is triangular in ANY of the four corner orientations is
#     flipped into upper triangular form, reduced, and flipped back;
#   * anything else takes the dense path below.
#
# There are now two dense paths.  `algorithm=:heuristic` follows flatter: a
# non-triangular basis enters phase 1 with unknown condition number and is sent
# to CondUnknown, which discovers a resolvable independent prefix, drives it
# through phase 2, and increases precision until every remaining dependency is
# represented by an exact zero column.  Rank-deficient input is therefore
# supported on the heuristic path.
#
# `algorithm=:teaching` keeps the older QR-and-round route because it is useful
# as an independent baseline and is substantially simpler to study:
#
#   1. QR-factor B and round its R factor to an integer triangular lattice;
#   2. reduce that integer lattice with the recursive driver;
#   3. apply the resulting unimodular transform to B;
#   4. repeat on the improved exact basis.
#
# The teaching dense path still requires full column rank.

const DEFAULT_DENSE_ROUNDS = 4

"""
Fixed-precision float types the dense path will try, cheapest first, before
falling back to `BigFloat`.

Only `Float64` is here by default, because it is all the standard library
offers. A double-double type gives about 106 bits at a small multiple of
`Float64` cost -- far cheaper than `BigFloat` at the same width -- and slots
into the gap this path otherwise jumps. Nothing in the package depends on such a
type; pass one in instead:

    using DoubleFloats
    reduce_basis(B; float_tiers = (Float64, Double64, BigFloat))

Which bases benefit: `householder_precision` is roughly `34.5 + entry_bits` at
dimension 128, so entries of 19 to 71 bits land between `Float64` and a
double-double. Bases with entries smaller than that are already on `Float64`,
and ones much larger need arbitrary precision regardless.
"""
const DEFAULT_FLOAT_TIERS = (Float64,)

# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

"""
    _is_corner_zero(B, left, top) -> Bool

Whether `B` is triangular with its zero corner at the given side.

Walks outward from the chosen corner: for each `i`, the entries nearer the
corner than the `i`th must be zero and the `i`th itself nonzero. `left, top`
selects which of the four corners is tested, so `(true, false)` — bottom left
zero — is ordinary upper triangular.
"""
function _is_corner_zero(B::AbstractMatrix, left::Bool, top::Bool)
    m, n = size(B)
    for i in 0:(m - 1)
        row = top ? m - i : i + 1
        for j in 0:i
            column = left ? j + 1 : n - j
            (1 <= row <= m && 1 <= column <= n) || return false
            if j < i
                iszero(B[row, column]) || return false
            else
                iszero(B[row, column]) && return false
            end
        end
    end
    return true
end

"""
    triangular_orientation(B) -> Union{Nothing, Tuple{Bool, Bool}}

Which row and column reversals bring `B` into upper triangular form, or
`nothing` if no combination does.

Reversing the columns reorders the basis vectors and reversing the rows permutes
the ambient coordinates, so neither changes the lattice — a basis triangular in
any corner orientation is as good as one already upper triangular.
"""
function triangular_orientation(B::AbstractMatrix)
    size(B, 1) == size(B, 2) || return nothing
    _is_corner_zero(B, true, false) && return (false, false)   # already upper
    _is_corner_zero(B, true, true) && return (true, false)     # top left zero
    _is_corner_zero(B, false, false) && return (false, true)   # bottom right
    _is_corner_zero(B, false, true) && return (true, true)     # top right
    return nothing
end

function _flip!(M::AbstractMatrix, flip_rows::Bool, flip_columns::Bool)
    m, n = size(M)
    if flip_rows
        for j in 1:n, i in 1:div(m, 2)
            M[i, j], M[m - i + 1, j] = M[m - i + 1, j], M[i, j]
        end
    end
    if flip_columns
        for j in 1:div(n, 2), i in 1:m
            M[i, j], M[i, n - j + 1] = M[i, n - j + 1], M[i, j]
        end
    end
    return M
end

# ---------------------------------------------------------------------------
# The integer approximation of a floating point R factor
# ---------------------------------------------------------------------------

"""
    to_integer_lattice(R, n) -> Matrix{BigInt}

Round the upper triangle of a floating point R factor to an integer triangular
matrix, at a scale that keeps every diagonal entry nonzero.

A port of flatter's `to_int_lattice`, which appears identically in
`columnwise.cpp` and `iterated.cpp`. The scale is chosen from the spread of the
diagonal: with `spread + n + 10` bits allotted to the largest
diagonal entry, the smallest retains `n + 10`, so none rounds away.

The result is a different lattice from the one `R` came from — it is an
approximation — but reducing it yields a unimodular transform, and a unimodular
transform is valid for any basis. That is what makes this usable.
"""
function to_integer_lattice(R::AbstractMatrix{S}, n::Integer) where {S<:AbstractFloat}
    n = Int(n)
    logs = Vector{Float64}(undef, n)
    for i in 1:n
        logs[i] = _log2_abs(R[i, i])
    end

    largest = maximum(logs)

    # Rank deficiency does not show up as an exact zero. Factorising a singular
    # matrix in floating point leaves the dependent diagonal entry at the noise
    # floor -- around 2^-53 relative to the rest for a Float64 -- and scaling
    # that up would manufacture a nonzero entry out of rounding error. The test
    # that catches it is relative: an entry more than the working precision
    # below the largest carries no information at all.
    available = Base.precision(R[1, 1])
    for i in 1:n
        largest - logs[i] > available - 10 && throw(ArgumentError(
            "diagonal entry $i is $(round(largest - logs[i]; digits = 1)) bits " *
            "below the largest, beyond the $available bits available: the basis " *
            "is rank deficient, or the precision is too low to resolve its " *
            "profile"))
    end

    spread = largest - minimum(logs)
    # Bits allotted to the largest diagonal entry, leaving the smallest with
    # `n + 10`. Named to avoid shadowing `Base.precision`, which is needed just
    # above -- flatter calls this local `prec`.
    allotted = spread + n + 10
    scale = round(Int, allotted - largest)

    result = zeros(BigInt, n, n)
    for j in 1:n, i in 1:j
        result[i, j] = round(BigInt, ldexp(R[i, j], scale))
    end
    for i in 1:n
        iszero(result[i, i]) && throw(ErrorException(
            "diagonal entry $i rounded to zero; the scale calculation is wrong"))
    end
    return result
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

"""
    _dense_precision(B, m) -> Int

Starting precision for the dense path's factorisation.

This uses the QR policy, not the reduction policy. `lll_precision` carries a
`2n` term describing the COMPRESSED basis inside the driver, whose profile spans
about `n` bits after compression; that term has nothing to do with factorising a
raw basis, and on a small-entry lattice it dominates — at dimension 128 with
8-bit entries it asks for 302 bits where 53 suffice, and MPFR cost scales with
precision.

The entry bit-length stands in for the log condition number, which is optimistic
for an ill-conditioned basis. `_dense_approximation` doubles from here when the
factorisation comes back degenerate, so an underestimate costs a retry rather
than a wrong answer.
"""
function _dense_precision(B::AbstractMatrix{<:Integer}, m::Int)
    largest = 0
    for value in B
        iszero(value) || (largest = max(largest, ndigits(value; base = 2)))
    end
    return householder_precision(m, Float64(largest))
end

"""
    _scaled_float_matrix(T, B) -> Matrix{T}

`B` converted to `T`, with a common power of two pulled out so that nothing
overflows the exponent range.

A uniform scale leaves the ratios between diagonal entries of the R factor
untouched, and `to_integer_lattice` chooses its own scale regardless, so the
factor is not shifted by this.
"""
function _scaled_float_matrix(::Type{T}, B::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    largest = 0
    for value in B
        iszero(value) || (largest = max(largest, ndigits(value; base = 2)))
    end
    headroom = max(0, Base.precision(T) - 5)
    shift = max(0, largest - headroom)
    return T[T(value >> shift) for value in B]
end

"Factor at a fixed float type, using LAPACK where the type is a BLAS one."
function _factor_at(::Type{T}, B::AbstractMatrix{<:Integer},
                    n::Int) where {T<:AbstractFloat}
    scaled = _scaled_float_matrix(T, B)
    if T <: LinearAlgebra.BlasFloat
        # `dgeqrf` and friends: the same Householder algorithm with BLAS-3 panel
        # updates. It leaves R in the upper trapezoid, which is all that
        # `to_integer_lattice` reads.
        LinearAlgebra.LAPACK.geqrf!(scaled)
        return to_integer_lattice(scaled, n)
    end
    factors, _ = householder_block(scaled)
    return to_integer_lattice(factors, n)
end

"""
    _dense_approximation(B, n, m) -> Matrix{BigInt}

The integer triangular lattice approximating `B`'s Gram-Schmidt structure,
factorising at the cheapest precision that produces a usable R factor.

A degenerate factor -- a diagonal entry rounding to zero -- means the precision
was too low to resolve the profile, so the precision doubles and it tries again.
That is the same discovery loop flatter's `CondUnknown` runs, in miniature: the
condition number is not known in advance, so it is found by attempting.
"""
function _dense_approximation(B::AbstractMatrix{<:Integer}, n::Int, m::Int;
                              hardware::Bool = true,
                              float_tiers = DEFAULT_FLOAT_TIERS)
    bits = _dense_precision(B, m)
    ceiling = 64 * max(bits, 64)

    # `householder_precision` bottoms out at 53 bits, which is exactly what a
    # `Float64` carries -- and for a small-entry basis that is what it asks for.
    # Running such a factorisation in `BigFloat` pays an allocation and an
    # indirection per operation to compute the same thing, and forgoes BLAS in
    # the leaf multiplications. `float64_matrix` pulls out a common power of two
    # so nothing overflows; a uniform scale leaves the ratios between diagonal
    # entries alone, and `to_integer_lattice` chooses its own scale regardless.
    # Try each fixed-precision tier that is wide enough, cheapest first. Every
    # one of these is dramatically faster than `BigFloat` at the same width --
    # no allocation per operation, and BLAS for the ones LAPACK handles -- so it
    # is worth taking the narrowest that will do.
    #
    # A degenerate factor from a tier means its width could not resolve the
    # profile after all, so the next tier gets a turn and arbitrary precision
    # has the last word.
    if hardware
        for tier in float_tiers
            tier === BigFloat && continue
            Base.precision(tier) >= bits || continue
            try
                return _factor_at(tier, B, n)
            catch problem
                problem isa ArgumentError || rethrow()
            end
        end
    end

    while true
        try
            return with_precision(bits) do
                factors, _ = householder_block(bigfloat_matrix(B, bits))
                to_integer_lattice(factors, n)
            end
        catch problem
            problem isa ArgumentError || rethrow()
            bits *= 2
            bits > ceiling && throw(ArgumentError(
                "could not factorise the basis at any precision up to $ceiling " *
                "bits; it is probably rank deficient, which this path does not " *
                "handle"))
        end
    end
end

"""
    reduce_basis!(B, U; kwargs...) -> (B, U, info)

Reduce an integer basis in place, with no requirement that it be triangular.

`B` is `m x n` with columns as basis vectors and `m >= n`; `U` is `n x n` with
the same element type. On return the new `B` equals the old `B` times `U`, and
`U` is unimodular. This is the entry point to prefer over
[`lattice_reduce!`](@ref) unless the input is known to be upper triangular
already.

`info` reports `path`, one of:

  * `:triangular`  — the basis was already upper triangular;
  * `:reoriented`  — it was triangular in another corner orientation and was
                     flipped into place;
  * `:dense`       — neither, so the QR-and-round route was used;

together with `rounds` (dense path only) and the fields
[`lattice_reduce!`](@ref) returns.

Keyword arguments beyond those of [`lattice_reduce!`](@ref):

  * `algorithm`  -- `:teaching` (the existing reducer) or `:heuristic` (flatter's
                    heuristic phase dispatcher).  Triangular/reoriented input
                    enters Heuristic2 after exact size reduction; dense input
                    enters CondUnknown and is then handed to Heuristic2 as its
                    rank/condition information becomes available.
  * `max_rounds` -- iterations of the dense path. Each round re-factorises an
                    already improved basis, so its integer approximation is a
                    better one; the loop stops early when a round produces the
                    identity. Default `$(DEFAULT_DENSE_ROUNDS)`.

# Rank

The teaching dense path requires full column rank.  The heuristic dense path
uses CondUnknown and supports rank-deficient input, moving exact zero
dependencies to the right of the returned basis.
"""
function reduce_basis!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
                       algorithm::Symbol = :teaching,
                       max_rounds::Integer = DEFAULT_DENSE_ROUNDS,
                       aggressive::Bool = false,
                       telemetry::Union{Nothing, ReductionTelemetry} = nothing,
                       float_tiers = DEFAULT_FLOAT_TIERS,
                       want_profile::Bool = true,
                       kwargs...) where {T<:Integer}
    m, n = size(B)
    algorithm in (:teaching, :heuristic) || throw(ArgumentError(
        "unknown reduction algorithm $algorithm; expected :teaching or :heuristic"))
    m >= n || throw(DimensionMismatch(
        "expected at least as many rows as columns, got $m x $n"))
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    (iszero(m) || iszero(n)) && throw(ArgumentError("B must be non-empty"))

    orientation = triangular_orientation(B)

    if orientation !== nothing
        flip_rows, flip_columns = orientation
        _flip!(B, flip_rows, flip_columns)

        if algorithm === :heuristic
            # flatter's Irregular::solve_triangular performs an exact triangular
            # size reduction before entering phase 2.  This step is required
            # even when Heuristic2 accepts the profile immediately at iteration
            # zero: size reduction does not change the diagonal profile, but it
            # is part of the returned reduced representative.
            U_sr = Matrix{T}(undef, n, n)
            size_reduction_triu!(B, U_sr)

            U_h2 = Matrix{T}(undef, n, n)
            _, _, info = heuristic2_reduce!(B, U_h2; aggressive = aggressive,
                                             telemetry = telemetry, kwargs...)

            # Q*B0*P is first changed by U_sr and then by U_h2, so the
            # transform in the oriented coordinates is U_sr * U_h2.
            combined = strassen(U_sr, U_h2)
            for j in 1:n, i in 1:n
                U[i, j] = combined[i, j]
            end
        else
            _, _, info = lattice_reduce!(B, U; aggressive = aggressive,
                                         want_profile = want_profile,
                                         telemetry = telemetry, kwargs...)
        end

        # Undo the row reversal on the basis, and apply the column reversal to
        # the transform: reducing `Q B P` to `Q B P U'` means the transform for
        # the original basis is `P U'`, which is `U'` with its rows reversed.
        _flip!(B, flip_rows, false)
        _flip!(U, flip_columns, false)

        path = (flip_rows || flip_columns) ? :reoriented : :triangular
        return B, U, (; path, rounds = 0, info...)
    end

    if algorithm === :heuristic
        _, _, info = cond_unknown_reduce!(B, U; aggressive = aggressive,
                                          telemetry = telemetry, kwargs...)
        return B, U, (; path = :dense, rounds = 0, info...)
    end

    return _reduce_dense!(B, U, Int(max_rounds), aggressive, telemetry,
                          float_tiers, want_profile; kwargs...)
end

function _reduce_dense!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                        max_rounds::Int, aggressive::Bool,
                        telemetry::Union{Nothing, ReductionTelemetry},
                        float_tiers, want_profile::Bool;
                        kwargs...) where {T<:Integer}
    m, n = size(B)
    _set_identity!(U)
    last_info = nothing
    rounds = 0

    previous_drop = Inf

    for round in 1:max_rounds
        # This factorisation is separate from the one inside the driver and is
        # paid once per round, so it is timed on its own.
        started = time_ns()
        approximation = _dense_approximation(B, n, m; float_tiers = float_tiers)
        if telemetry !== nothing
            telemetry.time_dense_qr += (time_ns() - started) / 1e9
            telemetry.dense_rounds += 1
        end

        # The approximation is upper triangular, so its profile is just the
        # diagonal -- no factorisation needed. Reading it BEFORE reducing gives
        # the drop this round starts from, which is what the stopping test
        # compares. Asking the driver for a profile instead would make it
        # factorise the reduced basis, and at these dimensions that costs more
        # than the reduction.
        drop = profile_drop([_log2_abs(approximation[i, i]) for i in 1:n])

        step = Matrix{BigInt}(undef, n, n)
        _, _, last_info = lattice_reduce!(approximation, step;
                                          aggressive = aggressive,
                                          want_profile = false,
                                          telemetry = telemetry, kwargs...)
        rounds = round

        applied = T.(step)
        _is_identity(applied) && break

        update_started = time_ns()

        # Any unimodular transform is valid, so this is exact regardless of how
        # good the approximation was; a poor one costs quality, never validity.
        updated = _reduction_mul(Matrix{T}(B), applied)
        for j in 1:n, i in 1:m
            B[i, j] = updated[i, j]
        end
        composed = _reduction_mul(Matrix{T}(U), applied)
        for j in 1:n, i in 1:n
            U[i, j] = composed[i, j]
        end
        # The basis and transform updates: two integer matrix products per
        # round, at full entry width.
        telemetry === nothing ||
            (telemetry.time_matmul += (time_ns() - update_started) / 1e9)

        # Another round re-factorises an improved basis and so approximates it
        # better -- but only while the basis is still improving. Once the
        # profile stops flattening there is nothing left for a further round to
        # find.
        drop >= previous_drop - 1e-6 && break
        previous_drop = drop
    end

    return B, U, (; path = :dense, rounds = rounds, last_info...)
end

function _is_identity(M::AbstractMatrix)
    n = size(M, 1)
    size(M, 2) == n || return false
    for j in 1:n, i in 1:n
        expected = i == j ? 1 : 0
        M[i, j] == expected || return false
    end
    return true
end

"""
    reduce_basis(B; kwargs...) -> (B_reduced, U, info)

Non-mutating form of [`reduce_basis!`](@ref). `B` is left untouched.
"""
function reduce_basis(B::AbstractMatrix{T}; kwargs...) where {T<:Integer}
    reduced = Matrix{T}(B)
    transform = Matrix{T}(undef, size(B, 2), size(B, 2))
    return reduce_basis!(reduced, transform; kwargs...)
end
