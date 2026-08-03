const DEFAULT_STRASSEN_CUTOFF = 32

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
                  cutoff::Integer=DEFAULT_STRASSEN_CUTOFF) where {TA, TB}
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
                   cutoff::Integer=DEFAULT_STRASSEN_CUTOFF,
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
        mul!(C, A, B)
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
        mul!(C, A, B)
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

