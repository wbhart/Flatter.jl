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
# THE DENSE PATH DIVERGES FROM FLATTER. flatter sends a non-triangular basis to
# phase 1, which -- as the comment in `Irregular::solve_rectangular` says -- is
# deliberately routed to `CondUnknown` because the condition number is unknown.
# `CondUnknown` is a substantial piece of machinery: column selection by
# orthogonal component, rank detection, and an iterative precision-doubling
# loop. None of it is ported.
#
# Instead the dense path uses what this package already has. Any unimodular
# transform is valid for any basis, so it suffices to find a good one:
#
#   1. QR-factor B and round its R factor to an integer triangular lattice,
#      which approximates B's Gram-Schmidt structure at a scale chosen so no
#      diagonal entry vanishes;
#   2. reduce that integer lattice with the recursive driver;
#   3. apply the resulting unimodular transform to B;
#   4. repeat, since the second round's factorisation is of an already better
#      basis and so approximates it more usefully.
#
# The cost is that rank-deficient input is detected and reported rather than
# handled -- that is what `CondUnknown` exists for.

const DEFAULT_DENSE_ROUNDS = 4

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
    _dense_approximation(B, n, m) -> Matrix{BigInt}

The integer triangular lattice approximating `B`'s Gram-Schmidt structure,
factorising at the cheapest precision that produces a usable R factor.

A degenerate factor -- a diagonal entry rounding to zero -- means the precision
was too low to resolve the profile, so the precision doubles and it tries again.
That is the same discovery loop flatter's `CondUnknown` runs, in miniature: the
condition number is not known in advance, so it is found by attempting.
"""
function _dense_approximation(B::AbstractMatrix{<:Integer}, n::Int, m::Int;
                              hardware::Bool = true)
    bits = _dense_precision(B, m)
    ceiling = 64 * max(bits, 64)

    # `householder_precision` bottoms out at 53 bits, which is exactly what a
    # `Float64` carries -- and for a small-entry basis that is what it asks for.
    # Running such a factorisation in `BigFloat` pays an allocation and an
    # indirection per operation to compute the same thing, and forgoes BLAS in
    # the leaf multiplications. `float64_matrix` pulls out a common power of two
    # so nothing overflows; a uniform scale leaves the ratios between diagonal
    # entries alone, and `to_integer_lattice` chooses its own scale regardless.
    if hardware && bits <= 53
        try
            scaled, _ = float64_matrix(B)
            # LAPACK's `dgeqrf` rather than this package's generic Householder:
            # same algorithm, but with hand-tuned BLAS-3 panel updates. It
            # overwrites its argument with R in the upper trapezoid, which is
            # exactly the part `to_integer_lattice` reads, so nothing needs
            # extracting.
            LinearAlgebra.LAPACK.geqrf!(scaled)
            return to_integer_lattice(scaled, n)
        catch problem
            # A degenerate factor here means 53 bits could not resolve the
            # profile after all, so fall through and discover the precision the
            # slow way.
            problem isa ArgumentError || rethrow()
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

  * `max_rounds` -- iterations of the dense path. Each round re-factorises an
                    already improved basis, so its integer approximation is a
                    better one; the loop stops early when a round produces the
                    identity. Default `$(DEFAULT_DENSE_ROUNDS)`.

# Rank

The basis must have full column rank. A rank-deficient one is reported rather
than handled; flatter's `CondUnknown` is what deals with that case and is not
ported.
"""
function reduce_basis!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
                       max_rounds::Integer = DEFAULT_DENSE_ROUNDS,
                       aggressive::Bool = false,
                       telemetry::Union{Nothing, ReductionTelemetry} = nothing,
                       kwargs...) where {T<:Integer}
    m, n = size(B)
    m >= n || throw(DimensionMismatch(
        "expected at least as many rows as columns, got $m x $n"))
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    (iszero(m) || iszero(n)) && throw(ArgumentError("B must be non-empty"))

    orientation = triangular_orientation(B)

    if orientation !== nothing
        flip_rows, flip_columns = orientation
        _flip!(B, flip_rows, flip_columns)

        _, _, info = lattice_reduce!(B, U; aggressive = aggressive,
                                     telemetry = telemetry, kwargs...)

        # Undo the row reversal on the basis, and apply the column reversal to
        # the transform: reducing `Q B P` to `Q B P U'` means the transform for
        # the original basis is `P U'`, which is `U'` with its rows reversed.
        _flip!(B, flip_rows, false)
        _flip!(U, flip_columns, false)

        path = (flip_rows || flip_columns) ? :reoriented : :triangular
        return B, U, (; path, rounds = 0, info...)
    end

    return _reduce_dense!(B, U, Int(max_rounds), aggressive, telemetry; kwargs...)
end

function _reduce_dense!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                        max_rounds::Int, aggressive::Bool,
                        telemetry::Union{Nothing, ReductionTelemetry};
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
        approximation = _dense_approximation(B, n, m)
        if telemetry !== nothing
            telemetry.time_dense_qr += (time_ns() - started) / 1e9
            telemetry.dense_rounds += 1
        end

        step = Matrix{BigInt}(undef, n, n)
        _, _, last_info = lattice_reduce!(approximation, step;
                                          aggressive = aggressive,
                                          telemetry = telemetry, kwargs...)
        rounds = round

        applied = T.(step)
        _is_identity(applied) && break

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

        # Another round re-factorises an improved basis and so approximates it
        # better -- but only while the basis is still improving. Once the
        # profile stops flattening there is nothing left for a further round to
        # find, and the factorisation is the most expensive part of this path.
        drop = profile_drop(last_info.profile)
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
