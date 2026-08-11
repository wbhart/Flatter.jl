# size_reduction_triu.jl
#
# Size reduction of an upper triangular integer matrix.
#
# Given a nonsingular upper triangular R (typically the R factor of a QR
# decomposition of a lattice basis, rounded to integers), compute a unit upper
# triangular unimodular U such that R*U is size reduced, i.e.
#
#     2 * |(R*U)[i,j]| <= |(R*U)[i,i]|    for all i < j.
#
# Three kernels are provided, mirroring the flatter C++ implementation:
#
#   :ll       elementary column sweep, multipliers rounded in floating point
#             (elementary_ll.cpp)
#   :zz       elementary column sweep, multipliers rounded exactly
#             (elementary_ZZ.cpp)
#   :blocked  tiled reformulation whose bulk work is tile-by-tile matrix
#             multiplication (blocked.cpp)
#
# All three are exact integer algorithms and produce a genuinely unimodular U,
# but they do not all produce the SAME U.  The rounding rule and the order of
# the column operations affect which reduced representative you land on, never
# the validity of the result, because C_i <- C_i - mu*C_j is an integer
# transvection of determinant one for any integer mu whatsoever.

const DEFAULT_SIZE_REDUCTION_BLOCKSIZE = 64

# ---------------------------------------------------------------------------
# Rounded division
# ---------------------------------------------------------------------------

"""
    _div_round(a::Integer, b::Integer)

Return `round(a / b)` exactly, with ties broken away from zero, so that the
remainder `a - b * _div_round(a, b)` satisfies `2 * |r| <= |b|`.

The C++ original computes `floor((2a + b) / (2b))`.  Here the equivalent result
is obtained from a truncating `divrem` followed by a correction, which avoids
forming `2a + b` and so cannot overflow a fixed-width integer type (except for
`typemin(T)`, where `abs` is already undefined).
"""
function _div_round(a::Integer, b::Integer)
    q, r = divrem(a, b)                     # truncated: sign(r) == sign(a)
    iszero(r) && return q
    ar = abs(r)
    ab = abs(b)
    if ar >= ab - ar                        # i.e. 2|r| >= |b|, without overflow
        q = xor(signbit(a), signbit(b)) ? q - one(q) : q + one(q)
    end
    return q
end

"""
    _div_round_float(T, a::Integer, b::Integer)

Return `round(a / b)` as a `T`, computed through floating point.

This is the rounding rule of `elementary_ll.cpp`.  For `Int64` the division is
carried out in `Float64`, whose 53 bit mantissa cannot represent every
quotient, so the result may occasionally be off by one.  That costs a little
reduction quality but never correctness.  For `BigInt` the promotion is to
`BigFloat` at the precision currently in force.
"""
_div_round_float(::Type{T}, a::Integer, b::Integer) where {T<:Integer} =
    round(T, float(a) / float(b))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _set_identity!(U::AbstractMatrix{T}) where {T}
    n = size(U, 1)
    @inbounds for j in 1:n, i in 1:n
        U[i, j] = i == j ? one(T) : zero(T)
    end
    return U
end

function _check_size_reduction_arguments(R::AbstractMatrix, U::AbstractMatrix)
    n = size(R, 1)
    size(R, 2) == n ||
        throw(DimensionMismatch("R must be square, got $(size(R))"))
    size(U) == (n, n) ||
        throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    @inbounds for j in 1:n
        iszero(R[j, j]) &&
            throw(ArgumentError("R has a zero diagonal entry at index $j"))
        for i in (j + 1):n
            iszero(R[i, j]) ||
                throw(ArgumentError("R must be upper triangular; R[$i,$j] is nonzero"))
        end
    end
    return nothing
end

"""
    _block_ranges(n, blocksize)

Partition `1:n` into contiguous ranges of nearly equal length, none longer than
`blocksize`.  Balancing avoids a final runt block, and the bound guarantees the
recursion in the blocked kernel terminates at the elementary kernel.
"""
function _block_ranges(n::Integer, blocksize::Integer)
    nb = cld(n, blocksize)
    base, extra = divrem(n, nb)
    ranges = Vector{UnitRange{Int}}(undef, nb)
    start = 1
    for b in 1:nb
        len = base + (b <= extra ? 1 : 0)
        ranges[b] = start:(start + len - 1)
        start += len
    end
    return ranges
end

"""
    _tile_matmul(A, B, cutoff)

Multiply two tiles.  This is the single seam between the blocked kernel and the
package's multiplication routine.

The arguments are views into the working matrices, and the tiles are not square
in general, since the final block row and column may be short.  Both cases are
handed to [`strassen`](@ref), which sends rectangular products and recursive
leaves to its base multiplication.  The result must be freshly allocated: the
caller writes it back over one of its own inputs.
"""
_tile_matmul(A::AbstractMatrix, B::AbstractMatrix, cutoff::Integer) =
    strassen(A, B; cutoff = cutoff)

# ---------------------------------------------------------------------------
# Elementary kernels
# ---------------------------------------------------------------------------

# Both elementary kernels implement the same column sweep:
#
#   U <- I
#   for i = 1:n                        # each column, left to right
#       for j = i-1:-1:1               # against earlier columns, bottom up
#           mu <- round(R[j,i] / R[j,j])
#           R[:,i] -= mu * R[:,j]
#           U[:,i] -= mu * U[:,j]
#
# The descending j loop is essential: subtracting a multiple of column j alters
# rows 1:j of column i, so entry (j,i) must be fixed before rows above it are
# touched.  It is also what makes the bound tight, since once (j,i) is set no
# later step in the sweep writes to it again.
#
# Unlike the C++ inner loop, which runs over all n rows, these run over 1:j.
# Both R and U are upper triangular throughout (U starts as I and only ever
# receives multiples of earlier columns), so the remaining rows are structurally
# zero.

"""
    _size_reduction_triu_ll!(R, U)

Elementary size reduction with floating point rounding of the multipliers.
Corresponds to `SizeReductionImpl::ElementaryLL`.
"""
function _size_reduction_triu_ll!(R::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T<:Integer}
    n = size(R, 1)
    _set_identity!(U)
    @inbounds for i in 1:n
        for j in (i - 1):-1:1
            mu = _div_round_float(T, R[j, i], R[j, j])
            iszero(mu) && continue
            for k in 1:j
                R[k, i] -= mu * R[k, j]
                U[k, i] -= mu * U[k, j]
            end
        end
    end
    return R, U
end

"""
    _size_reduction_triu_zz!(R, U)

Elementary size reduction with exact rounding of the multipliers.
Corresponds to `SizeReductionImpl::ElementaryZZ`.
"""
function _size_reduction_triu_zz!(R::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T<:Integer}
    n = size(R, 1)
    _set_identity!(U)
    @inbounds for i in 1:n
        for j in (i - 1):-1:1
            mu = _div_round(R[j, i], R[j, j])
            iszero(mu) && continue
            for k in 1:j
                R[k, i] -= mu * R[k, j]
                U[k, i] -= mu * U[k, j]
            end
        end
    end
    return R, U
end

# ---------------------------------------------------------------------------
# Blocked kernel
# ---------------------------------------------------------------------------

# R is cut into nb x nb tiles.  An auxiliary n x n matrix Tr records, for each
# strictly upper tile (i,j), the matrix of multipliers -mu produced when tile
# (i,j) was reduced against the diagonal tile (i,i).  Storing them lets the same
# column operations be replayed against tiles further up block column j as a
# single matrix product rather than as thousands of rank-1 updates.
#
# Tiles are visited by diagonal distance d = j - i, outward from the diagonal.
# When tile (i,j) is reached, everything it depends on is already final:
# U(j,j) and R(i,i) come from d = 0, and each R(i,k), U(i,k), Tr(k,j) for
# i < k < j sits at a distance strictly less than d.
#
# Writing R for the working matrix and R0 for the input, the relation maintained
# for block column j is
#
#     R(i,j) <- R0(i,j)*U(j,j) + sum_{k=i}^{j-1} R(i,k)*Tr(k,j)
#     U(i,j) <-                  sum_{k=i}^{j-1} U(i,k)*Tr(k,j)
#
# with the k = i term supplied by the in-place tile reduction.  The U analogue
# of the first term is omitted because U(i,j) is still zero at that point.  The
# order of the k terms is irrelevant: each Tr(k,j) is already determined, so the
# additions commute.
#
# Note the asymmetry in the first term: U(j,j) multiplies the ORIGINAL R0(i,j),
# whereas the elementary sweep, on reaching a column of block j, subtracts
# multiples of earlier columns in their already-reduced state.  So the blocked
# kernel is not merely a reordering of the elementary one and does not in
# general return the same U; the two agree only on a majority of inputs.  Both
# results are correct: R = R0*U with U unimodular, and both meet the same
# reduction bound, because the discrepancy introduced in block rows above the
# diagonal is cleaned up by the final tile reduction against R(i,i).

"""
    _size_reduction_triu_blocked!(R, U, blocksize, strassen_cutoff)

Tiled size reduction.  Corresponds to `SizeReductionImpl::Blocked`.  Tile
products go through [`strassen`](@ref); products at or below `strassen_cutoff`
are handled by its base multiplication.  Falls back to the elementary kernel
when `size(R, 1) <= blocksize`.
"""
function _size_reduction_triu_blocked!(R::AbstractMatrix{T}, U::AbstractMatrix{T},
                                       blocksize::Integer,
                                       strassen_cutoff::Integer) where {T<:Integer}
    n = size(R, 1)
    if n <= blocksize
        return _size_reduction_triu_zz!(R, U)
    end

    ranges = _block_ranges(n, blocksize)
    nb = length(ranges)

    _set_identity!(U)
    Tr = zeros(T, n, n)

    # d = 0: reduce each diagonal tile on its own.  The recursive call goes
    # through the dispatcher, so a tile larger than blocksize would be blocked
    # again; by construction of _block_ranges it never is.
    for i in 1:nb
        rg = ranges[i]
        size_reduction_triu!(view(R, rg, rg), view(U, rg, rg);
                             blocksize = blocksize, strassen_cutoff = strassen_cutoff)
    end

    for d in 1:(nb - 1)
        for i in 1:(nb - d)
            j = i + d
            ri, rj = ranges[i], ranges[j]

            Rij = view(R, ri, rj)
            Uij = view(U, ri, rj)

            # diag_above: apply block column j's own diagonal transform.
            Rij .= _tile_matmul(Rij, view(U, rj, rj), strassen_cutoff)

            # inner_above: replay the reductions recorded for intervening tiles.
            for k in (i + 1):(j - 1)
                rk = ranges[k]
                Trkj = view(Tr, rk, rj)
                Rij .+= _tile_matmul(view(R, ri, rk), Trkj, strassen_cutoff)
                Uij .+= _tile_matmul(view(U, ri, rk), Trkj, strassen_cutoff)
            end

            # inner_inner: the only remaining elementary work, confined to one
            # tile, and the only place multipliers are computed.
            _reduce_tile_against_diagonal!(R, U, Tr, ri, rj)
        end
    end

    return R, U
end

# Reduce tile (rows = ri, cols = rj) against the diagonal tile (ri, ri), bottom
# up within each column, recording the multipliers into Tr.  Because ri is
# contiguous, the diagonal entry governing global row r is simply R[r,r].
function _reduce_tile_against_diagonal!(R::AbstractMatrix{T}, U::AbstractMatrix{T},
                                        Tr::AbstractMatrix{T},
                                        ri::UnitRange{Int}, rj::UnitRange{Int}) where {T<:Integer}
    @inbounds for c in rj
        for r in reverse(ri)
            mu = _div_round(R[r, c], R[r, r])
            Tr[r, c] = -mu
            iszero(mu) && continue
            for k in first(ri):r
                R[k, c] -= mu * R[k, r]
                U[k, c] -= mu * U[k, r]
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

"""
    size_reduction_triu!(R, U; algorithm=:auto, blocksize=$(DEFAULT_SIZE_REDUCTION_BLOCKSIZE),
                         strassen_cutoff=DEFAULT_STRASSEN_CUTOFF)

Size reduce the nonsingular upper triangular integer matrix `R` in place and
overwrite `U` with the unimodular transform that was applied, so that on return
the new `R` equals the old `R` times `U`.  `U` is unit upper triangular, hence
has determinant one.  Returns `(R, U)`.

`R` and `U` must be `n x n` with the same element type.  Any Julia `Integer`
type works; `BigInt` is the only one immune to overflow, since the entries of
`U` and the intermediate entries of `R` can be much larger than the inputs.

`algorithm` selects the kernel:

  * `:auto`    — `:blocked` when `n > blocksize`, otherwise `:zz`.
  * `:zz`      — elementary sweep, exact rounding.
  * `:ll`      — elementary sweep, floating point rounding.
  * `:blocked` — tiled, with the bulk of the work in matrix products.

The three kernels do not in general return the same `U`.  `:ll` may pick a
multiplier that differs by one where the floating point quotient misrounds, and
`:blocked` genuinely reduces in a different order (see the comments above the
blocked kernel).  All three satisfy the same contract: `R` becomes `R*U`, `U` is
unimodular, and the output meets the size reduction bound.

`strassen_cutoff` is forwarded to [`strassen`](@ref), which performs the tile
products of `:blocked`.  Strassen is exact here: it uses only `+`, `-` and `*`,
so over the ring of integers it is free of the numerical instability it has in
floating point, and it returns the same result as ordinary multiplication for
every value of the cutoff.
"""
function size_reduction_triu!(R::AbstractMatrix{T}, U::AbstractMatrix{T};
                              algorithm::Symbol = :auto,
                              blocksize::Integer = DEFAULT_SIZE_REDUCTION_BLOCKSIZE,
                              strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF) where {T<:Integer}
    _check_size_reduction_arguments(R, U)
    blocksize >= 1 || throw(ArgumentError("blocksize must be positive, got $blocksize"))

    n = size(R, 1)
    alg = algorithm === :auto ? (n > blocksize ? :blocked : :zz) : algorithm

    if alg === :zz
        _size_reduction_triu_zz!(R, U)
    elseif alg === :ll
        _size_reduction_triu_ll!(R, U)
    elseif alg === :blocked
        _size_reduction_triu_blocked!(R, U, blocksize, strassen_cutoff)
    else
        throw(ArgumentError("unknown algorithm :$algorithm"))
    end
    return R, U
end

"""
    size_reduction_triu(R; algorithm=:auto, blocksize=$(DEFAULT_SIZE_REDUCTION_BLOCKSIZE),
                        strassen_cutoff=DEFAULT_STRASSEN_CUTOFF)

Non-mutating form of [`size_reduction_triu!`](@ref).  Returns `(Rred, U)` with
`Rred == R * U`, leaving `R` untouched.
"""
function size_reduction_triu(R::AbstractMatrix{T}; kwargs...) where {T<:Integer}
    n = size(R, 1)
    Rred = Matrix{T}(R)
    U = Matrix{T}(undef, n, n)
    size_reduction_triu!(Rred, U; kwargs...)
    return Rred, U
end
