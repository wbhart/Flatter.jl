using Base.GMP: MPZ

const DEFAULT_STRASSEN_CUTOFF = 32

# BigInt wants a much lower crossover than machine floats do. For `Float64` the
# leaf multiplication is BLAS, which is fast enough that recursing far is a
# loss; for `BigInt` the leaf is a scalar loop and the entries dominate, so
# trading multiplications for additions keeps paying well past the point where
# it stops paying for floats.
#
# The value below is measured, not guessed; `lattices/profile_lift.jl` has a
# sweep that reproduces the measurement, and it should be re-run if the leaf
# multiplication changes.
const DEFAULT_STRASSEN_CUTOFF_BIGINT = 16

"""
    _default_cutoff(T) -> Int

The recursion cutoff to use for element type `T` when the caller does not give
one. See [`DEFAULT_STRASSEN_CUTOFF_BIGINT`](@ref) for why `BigInt` differs.
"""
_default_cutoff(::Type) = DEFAULT_STRASSEN_CUTOFF
_default_cutoff(::Type{BigInt}) = DEFAULT_STRASSEN_CUTOFF_BIGINT

"""
    strassen_workspace_length(n; cutoff=32)

Return the number of scalar entries required by the reusable workspace for
multiplication of two `n × n` matrices with the padded Strassen algorithm.

For a recursive problem of order `n`, put `t = cld(n, 2)`.  Three `t × t`
temporary blocks are used at that level, and the recursive product uses the
workspace below those blocks.  Thus

    W(n) = 0                              if n <= cutoff,
    W(n) = 3 * cld(n, 2)^2 + W(cld(n, 2)) otherwise.

For a power-of-two order `n` whose leaves all have order `b`, this is exactly
`n^2 - b^2` entries, and is therefore less than `n^2` entries.
"""
function strassen_workspace_length(::Type{T}, n::Integer;
                                   cutoff::Integer=_default_cutoff(T)) where {T}
    return strassen_workspace_length(n; cutoff = cutoff)
end

# NOTE: the element-type-free form below keeps the generic default. Because
# `strassen!` picks its cutoff from the element type, a workspace sized with
# this method and then handed to `strassen!` on a `BigInt` matrix would be too
# SMALL -- the lower cutoff recurses deeper and needs more space. Prefer the
# method above, which takes the type, or pass the same explicit `cutoff` to both.
function strassen_workspace_length(n::Integer; cutoff::Integer=DEFAULT_STRASSEN_CUTOFF)
    n >= 0 || throw(ArgumentError("n must be nonnegative"))
    cutoff >= 1 || throw(ArgumentError("cutoff must be positive"))
    return _workspace_length(Int(n), Int(cutoff))
end

function _workspace_length(n::Int, cutoff::Int)
    n > cutoff || return 0

    t = cld(n, 2)
    q = Base.checked_mul(t, t)
    here = Base.checked_mul(3, q)

    return Base.checked_add(here, _workspace_length(t, cutoff))
end

"""
    strassen(A, B; cutoff=32)

Return `A * B`.

Square products larger than `cutoff` are evaluated with padded Strassen
recursion.  Odd orders are handled directly: at order `n`, each quadrant is
regarded as a zero-padded `cld(n, 2) × cld(n, 2)` matrix.  Rectangular products
and recursive leaves use `LinearAlgebra.mul!`.

If `A` and `B` have different element types, they are converted to their
common promoted type.  The element type must support `zero`, `+`, `-`, and `*`.
"""
function strassen(A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
                  cutoff::Integer=_default_cutoff(promote_type(TA, TB))) where {TA, TB}
    m, k = size(A)
    kb, n = size(B)
    k == kb || throw(DimensionMismatch(
        "A has $k columns, but B has $kb rows"))

    T = promote_type(TA, TB)
    T === Any && throw(ArgumentError(
        "the element types $TA and $TB do not have a concrete promoted type"))

    Ap = TA === T ? A : Matrix{T}(A)
    Bp = TB === T ? B : Matrix{T}(B)
    C = Matrix{T}(undef, m, n)

    return strassen!(C, Ap, Bp; cutoff=cutoff)
end


# ---------------------------------------------------------------------------
# Leaf multiplication
# ---------------------------------------------------------------------------
#
# Strassen's recursion bottoms out here, and rectangular products are handled
# here in full.  For most element types Julia's kernel is the right choice.
#
# `BigInt` is the exception.  `mul!` accumulates with `muladd`, which allocates
# a fresh `BigInt` for the product AND another for the running sum at every step
# of the inner loop -- roughly `2 * m * k * n` allocations for a single leaf.
# Accumulating in place through `MPZ` needs one temporary for the whole call plus
# one `BigInt` per output entry, which is the unavoidable minimum.

@inline _leaf_mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) =
    mul!(C, A, B)

function _leaf_mul!(C::AbstractMatrix{BigInt},
                    A::AbstractMatrix{BigInt},
                    B::AbstractMatrix{BigInt})
    m, inner = size(A)
    n = size(B, 2)

    # One scratch value for the whole call.  `C` is never read: its entries may
    # be undefined references, since a workspace is allocated with `undef` and
    # `BigInt` is a mutable type.
    product = BigInt()

    @inbounds for j in 1:n
        for i in 1:m
            total = BigInt()
            for t in 1:inner
                MPZ.mul!(product, A[i, t], B[t, j])
                MPZ.add!(total, product)
            end
            C[i, j] = total
        end
    end
    return C
end

"""
    strassen!(C, A, B; cutoff=32, workspace=nothing)

Overwrite `C` with `A * B`.

`A`, `B`, and `C` must have the same element type, and `C` must not alias either
input.  To avoid allocating the structural Strassen workspace on each call,
pass a `Vector{T}` whose length is at least
`strassen_workspace_length(n; cutoff=cutoff)` for an `n × n` product.

At each recursive level the implementation uses three padded temporary blocks:
two operands and one product.  Only one recursive product is live at a time,
so the same lower portion of the workspace is reused for all seven products.
"""
function strassen!(C::AbstractMatrix{T},
                   A::AbstractMatrix{T},
                   B::AbstractMatrix{T};
                   cutoff::Integer=_default_cutoff(T),
                   workspace::Union{Nothing, Vector{T}}=nothing) where {T}
    cutoff >= 1 || throw(ArgumentError("cutoff must be positive"))
    c = Int(cutoff)

    Base.require_one_based_indexing(C, A, B)

    m, k = size(A)
    kb, n = size(B)
    k == kb || throw(DimensionMismatch(
        "A has $k columns, but B has $kb rows"))
    size(C) == (m, n) || throw(DimensionMismatch(
        "C has size $(size(C)); expected ($m, $n)"))

    Base.mightalias(C, A) && throw(ArgumentError("C must not alias A"))
    Base.mightalias(C, B) && throw(ArgumentError("C must not alias B"))

    # Strassen is used for arbitrary square orders.  Rectangular products and
    # sufficiently small square products are delegated to Julia's kernel.
    if !(m == k && k == n) || m <= c
        _leaf_mul!(C, A, B)
        return C
    end

    needed = _workspace_length(m, c)
    work = workspace === nothing ? Vector{T}(undef, needed) : workspace
    length(work) >= needed || throw(DimensionMismatch(
        "workspace has length $(length(work)); at least $needed entries are required"))

    _strassen_square!(C, A, B, work, 1, c)
    return C
end

# Set every entry of D to zero.  This is used only for non-leaf output blocks,
# which are subsequently updated by the seven Strassen products.
@inline function _zero!(D::AbstractMatrix{T}) where {T}
    z = zero(T)
    @inbounds for j in axes(D, 2)
        for i in axes(D, 1)
            D[i, j] = z
        end
    end
    return D
end

# D := X, treating entries outside X as zero and discarding entries outside D.
# D may be identical to X.  Other partial overlap is not supported.
@inline function _copy_padded!(D::AbstractMatrix{T},
                               X::AbstractMatrix{T}) where {T}
    mx, nx = size(X)
    z = zero(T)

    @inbounds for j in axes(D, 2)
        for i in axes(D, 1)
            D[i, j] = (i <= mx && j <= nx) ? X[i, j] : z
        end
    end

    return D
end

# D := X + Y, treating missing entries as zero and cropping to the size of D.
# D may be identical to X or Y.  Other partial overlap is not supported.
@inline function _add_padded!(D::AbstractMatrix{T},
                              X::AbstractMatrix{T},
                              Y::AbstractMatrix{T}) where {T}
    mx, nx = size(X)
    my, ny = size(Y)
    z = zero(T)

    @inbounds for j in axes(D, 2)
        for i in axes(D, 1)
            in_x = i <= mx && j <= nx
            in_y = i <= my && j <= ny

            if in_x
                D[i, j] = in_y ? X[i, j] + Y[i, j] : X[i, j]
            else
                D[i, j] = in_y ? Y[i, j] : z
            end
        end
    end

    return D
end

# D := X - Y, treating missing entries as zero and cropping to the size of D.
# D may be identical to X or Y.  Other partial overlap is not supported.
@inline function _sub_padded!(D::AbstractMatrix{T},
                              X::AbstractMatrix{T},
                              Y::AbstractMatrix{T}) where {T}
    mx, nx = size(X)
    my, ny = size(Y)
    z = zero(T)

    @inbounds for j in axes(D, 2)
        for i in axes(D, 1)
            in_x = i <= mx && j <= nx
            in_y = i <= my && j <= ny

            if in_x
                D[i, j] = in_y ? X[i, j] - Y[i, j] : X[i, j]
            else
                D[i, j] = in_y ? -Y[i, j] : z
            end
        end
    end

    return D
end

function _strassen_square!(C::AbstractMatrix{T},
                           A::AbstractMatrix{T},
                           B::AbstractMatrix{T},
                           work::Vector{T},
                           offset::Int,
                           cutoff::Int) where {T}
    n = size(C, 1)

    if n <= cutoff
        _leaf_mul!(C, A, B)
        return C
    end

    t = cld(n, 2)
    q = Base.checked_mul(t, t)
    two_q = Base.checked_mul(2, q)
    three_q = Base.checked_mul(3, q)
    child_offset = Base.checked_add(offset, three_q)

    # Three padded t × t blocks at this level: a product and two operands.
    P = reshape(view(work, offset:(offset + q - 1)), t, t)
    L = reshape(view(work, (offset + q):(offset + two_q - 1)), t, t)
    R = reshape(view(work, (offset + two_q):(child_offset - 1)), t, t)

    # Recursive calls use only the workspace below P, L, and R.

    @views begin
        A11 = A[1:t,       1:t]
        A12 = A[1:t,       t + 1:n]
        A21 = A[t + 1:n,   1:t]
        A22 = A[t + 1:n,   t + 1:n]

        B11 = B[1:t,       1:t]
        B12 = B[1:t,       t + 1:n]
        B21 = B[t + 1:n,   1:t]
        B22 = B[t + 1:n,   t + 1:n]

        C11 = C[1:t,       1:t]
        C12 = C[1:t,       t + 1:n]
        C21 = C[t + 1:n,   1:t]
        C22 = C[t + 1:n,   t + 1:n]

        _zero!(C)

        # M1 = (A11 + A22)(B11 + B22)
        _add_padded!(L, A11, A22)
        _add_padded!(R, B11, B22)
        _strassen_square!(P, L, R, work, child_offset, cutoff)
        _add_padded!(C11, C11, P)
        _add_padded!(C22, C22, P)

        # M2 = (A21 + A22)B11
        _add_padded!(L, A21, A22)
        _copy_padded!(R, B11)
        _strassen_square!(P, L, R, work, child_offset, cutoff)
        _add_padded!(C21, C21, P)
        _sub_padded!(C22, C22, P)

        # M3 = A11(B12 - B22)
        _copy_padded!(L, A11)
        _sub_padded!(R, B12, B22)
        _strassen_square!(P, L, R, work, child_offset, cutoff)
        _add_padded!(C12, C12, P)
        _add_padded!(C22, C22, P)

        # M4 = A22(B21 - B11)
        _copy_padded!(L, A22)
        _sub_padded!(R, B21, B11)
        _strassen_square!(P, L, R, work, child_offset, cutoff)
        _add_padded!(C11, C11, P)
        _add_padded!(C21, C21, P)

        # M5 = (A11 + A12)B22
        _add_padded!(L, A11, A12)
        _copy_padded!(R, B22)
        _strassen_square!(P, L, R, work, child_offset, cutoff)
        _sub_padded!(C11, C11, P)
        _add_padded!(C12, C12, P)

        # M6 = (A21 - A11)(B11 + B12)
        _sub_padded!(L, A21, A11)
        _add_padded!(R, B11, B12)
        _strassen_square!(P, L, R, work, child_offset, cutoff)
        _add_padded!(C22, C22, P)

        # M7 = (A12 - A22)(B21 + B22)
        _sub_padded!(L, A12, A22)
        _add_padded!(R, B21, B22)
        _strassen_square!(P, L, R, work, child_offset, cutoff)
        _add_padded!(C11, C11, P)
    end

    return C
end

