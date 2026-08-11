const DEFAULT_QR_BLOCK_CUTOFF = 32

# Unblocked Householder QR in compact-WY form.

"""
    householder(A)

Compute an unblocked Householder QR factorization of the real floating-point
matrix `A` and return

    (factors = F, T = T)

in compact-WY form.

The matrix `F` has the usual packed QR layout:

  * the upper trapezoidal part is `R`;
  * below the diagonal, column `j` stores the tail of the Householder vector
    `v_j`, whose first nonzero entry is the implicit value one.

If `r = min(size(A)...)`, then `T` is an `r × r` upper triangular matrix such
that, with `V` obtained from the packed Householder vectors,

    Q = I - V * T * transpose(V)

and

    transpose(Q) * A = R.

`A` is not modified.  Arithmetic is carried out at the precision currently in
force for its element type; in particular, this routine works with
`Matrix{BigFloat}`.
"""
function householder(A::AbstractMatrix{S}) where {S<:AbstractFloat}
    factors = Matrix{S}(A)
    r = min(size(factors)...)
    compact_T = Matrix{S}(undef, r, r)
    work = Vector{S}(undef, r)

    householder!(factors, compact_T, work)
    return (factors = factors, T = compact_T)
end

"""
    householder!(A, compact_T[, work])

Overwrite `A` with an unblocked packed Householder QR factorization and fill
`compact_T` with the upper triangular compact-WY factor.  The optional `work`
vector must have length at least `min(size(A)...)`.

This is the allocation-controlled kernel intended for use as the base case of
a later blocked or recursive QR implementation.  It is valid to pass ordinary
one-based matrix views for `A` and `compact_T`.
"""
function householder!(
    A::AbstractMatrix{S},
    compact_T::AbstractMatrix{S},
    work::AbstractVector{S} = Vector{S}(undef, min(size(A)...)),
) where {S<:AbstractFloat}
    Base.require_one_based_indexing(A, compact_T, work)

    m, n = size(A)
    r = min(m, n)

    size(compact_T, 1) >= r ||
        throw(DimensionMismatch("compact_T has too few rows"))
    size(compact_T, 2) >= r ||
        throw(DimensionMismatch("compact_T has too few columns"))
    length(work) >= r ||
        throw(DimensionMismatch("work has length $(length(work)); at least $r is required"))

    # The caller may be reusing storage from an earlier factorization.
    for j in 1:r
        for i in 1:r
            compact_T[i, j] = zero(S)
        end
    end

    for k in 1:r
        tau = _householder_reflector!(A, k)

        # Accumulate H_1 ... H_k = I - V_k T_k V_k'.  For k > 1,
        #
        #     T[1:k-1,k] = -tau * T[1:k-1,1:k-1] * (V_prev' * v_k).
        #
        # The full vectors are not formed.  For j < k,
        #
        #     v_j' * v_k = A[k,j] + A[k+1:m,j]' * A[k+1:m,k],
        #
        # because v_k has an implicit one in row k.
        if k > 1 && !iszero(tau)
            for j in 1:k-1
                s = A[k, j]
                for i in k+1:m
                    s += A[i, j] * A[i, k]
                end
                work[j] = s
            end

            # In-place multiplication work <- -tau * T * work.
            # T is upper triangular.  Processing rows from top to bottom is
            # safe because row i depends only on original work[i:k-1].
            for i in 1:k-1
                s = zero(S)
                for j in i:k-1
                    s += compact_T[i, j] * work[j]
                end
                work[i] = -tau * s
            end

            for i in 1:k-1
                compact_T[i, k] = work[i]
            end
        end
        compact_T[k, k] = tau

        # Apply H_k = I - tau*v_k*v_k' to the columns to the right.
        # The first entry of v_k is the implicit value one.
        if !iszero(tau)
            for j in k+1:n
                s = A[k, j]
                for i in k+1:m
                    s += A[i, k] * A[i, j]
                end
                s *= tau

                A[k, j] -= s
                for i in k+1:m
                    A[i, j] -= A[i, k] * s
                end
            end
        end
    end

    return (factors = A, T = compact_T)
end

# Construct one real Householder reflector for column k.
#
# On entry the active vector is A[k:m,k].  On return:
#
#   * A[k,k] is the new diagonal entry beta of R;
#   * A[k+1:m,k] is the tail of v, with v[1] = 1 implicit;
#   * the return value is tau in H = I - tau*v*v'.
function _householder_reflector!(A::AbstractMatrix{S}, k::Int) where {S<:AbstractFloat}
    m = size(A, 1)
    alpha = A[k, k]
    tail_norm = _householder_tail_norm(A, k + 1, m, k)

    # This includes the one-element case.  The identity reflector is enough
    # when the tail is already zero.
    if iszero(tail_norm)
        return zero(S)
    end

    beta = -copysign(hypot(alpha, tail_norm), alpha)
    tau = (beta - alpha) / beta
    denominator = alpha - beta

    for i in k+1:m
        A[i, k] /= denominator
    end
    A[k, k] = beta

    return tau
end

# Scaled sum-of-squares norm of A[first:last,column].  This avoids overflow
# and underflow without performing a square root for every vector entry.
function _householder_tail_norm(
    A::AbstractMatrix{S},
    first::Int,
    last::Int,
    column::Int,
) where {S<:AbstractFloat}
    scale = zero(S)
    sumsq = one(S)

    for i in first:last
        x = abs(A[i, column])
        if !iszero(x)
            if scale < x
                ratio = scale / x
                sumsq = one(S) + sumsq * ratio * ratio
                scale = x
            else
                ratio = x / scale
                sumsq += ratio * ratio
            end
        end
    end

    return iszero(scale) ? zero(S) : scale * sqrt(sumsq)
end

# Reusable storage used to reduce a rectangular product to a sequence of
# square, zero-padded products.  Each padded product has order at most
# `size(left, 1)`.
struct _HouseholderBlockMultiplyWorkspace{S<:AbstractFloat}
    left::Matrix{S}
    right::Matrix{S}
    product::Matrix{S}
    strassen::Vector{S}
end

function _householder_block_multiply_workspace(
    ::Type{S},
    order::Int,
    strassen_cutoff::Int,
) where {S<:AbstractFloat}
    order >= 0 || throw(ArgumentError("workspace order must be nonnegative"))

    workspace_length = iszero(order) ? 0 :
        strassen_workspace_length(order; cutoff = strassen_cutoff)

    return _HouseholderBlockMultiplyWorkspace(
        Matrix{S}(undef, order, order),
        Matrix{S}(undef, order, order),
        Matrix{S}(undef, order, order),
        Vector{S}(undef, workspace_length),
    )
end

"""
    householder_block(A;
        cutoff = DEFAULT_QR_BLOCK_CUTOFF,
        strassen_cutoff = DEFAULT_STRASSEN_CUTOFF)

Compute a recursive Householder QR factorization of the real floating-point
matrix `A`, returning

    (factors = F, T = T)

in the same packed compact-WY format as [`householder`](@ref):

  * the upper trapezoidal part of `F` is `R`;
  * below the diagonal, column `j` stores the tail of the Householder vector
    `v_j`, whose first nonzero entry is the implicit value one;
  * if `V` denotes the resulting matrix of Householder vectors, then

        Q = I - V * T * transpose(V)

    and

        transpose(Q) * A = R.

The QR recursion stops when the active subproblem has at most `cutoff`
Householder reflectors, at which point [`householder!`](@ref) is used.

The large matrix products in the recursive updates are generally rectangular.
They are partitioned into square blocks, zero-padded at their edges, and
computed by the package's square [`strassen!`](@ref) routine.  Square block
products of order at most `strassen_cutoff` are therefore evaluated by the
base multiplication selected by `strassen!`.

`A` is not modified.  In particular, the routine works with
`Matrix{BigFloat}` at the precision of its entries.
"""
function householder_block(
    A::AbstractMatrix{S};
    cutoff::Integer = DEFAULT_QR_BLOCK_CUTOFF,
    strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
) where {S<:AbstractFloat}
    cutoff_value = _check_qr_block_cutoff(cutoff)
    strassen_cutoff_value = _check_qr_strassen_cutoff(strassen_cutoff)

    factors = Matrix{S}(A)
    r = min(size(factors)...)
    compact_T = Matrix{S}(undef, r, r)

    householder_block!(
        factors,
        compact_T;
        cutoff = cutoff_value,
        strassen_cutoff = strassen_cutoff_value,
    )

    return (factors = factors, T = compact_T)
end

"""
    householder_block!(A, compact_T;
        cutoff = DEFAULT_QR_BLOCK_CUTOFF,
        strassen_cutoff = DEFAULT_STRASSEN_CUTOFF)

Overwrite `A` with a recursive packed Householder QR factorization and fill
`compact_T` with its upper triangular compact-WY factor.  All recursive QR and
Strassen workspaces are allocated once at the top level and then reused.

For a call in which the QR panel and vector workspaces are supplied by the
caller, use

    householder_block!(A, compact_T, block_work, vector_work;
                       cutoff, strassen_cutoff)

instead.  That method still constructs the three padded multiplication blocks
and the structural Strassen workspace once per top-level call.  The fully
workspace-controlled internal method additionally accepts an
`_HouseholderBlockMultiplyWorkspace`.
"""
function householder_block!(
    A::AbstractMatrix{S},
    compact_T::AbstractMatrix{S};
    cutoff::Integer = DEFAULT_QR_BLOCK_CUTOFF,
    strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
) where {S<:AbstractFloat}
    cutoff_value = _check_qr_block_cutoff(cutoff)
    strassen_cutoff_value = _check_qr_strassen_cutoff(strassen_cutoff)

    m, n = size(A)
    r = min(m, n)
    split = r > cutoff_value ? div(r, 2) : 0
    work_columns = iszero(split) ? 0 : n - split

    block_work = Matrix{S}(undef, split, work_columns)
    vector_work = Vector{S}(undef, r)

    return householder_block!(
        A,
        compact_T,
        block_work,
        vector_work;
        cutoff = cutoff_value,
        strassen_cutoff = strassen_cutoff_value,
    )
end

"""
    householder_block!(A, compact_T, block_work, vector_work;
        cutoff = DEFAULT_QR_BLOCK_CUTOFF,
        strassen_cutoff = DEFAULT_STRASSEN_CUTOFF)

Recursive Householder QR kernel with caller-provided QR workspaces.

If `r = min(size(A)...)` and `r > cutoff`, let `s = fld(r, 2)`.  Then
`block_work` must have at least `s` rows and `size(A, 2) - s` columns.  If
`r <= cutoff`, `block_work` may be empty.  In all cases, `vector_work` must
have length at least `r`.

A multiplication workspace of maximum square order `fld(r, 2)` is allocated
once by this method and reused throughout the recursion.
"""
function householder_block!(
    A::AbstractMatrix{S},
    compact_T::AbstractMatrix{S},
    block_work::AbstractMatrix{S},
    vector_work::AbstractVector{S};
    cutoff::Integer = DEFAULT_QR_BLOCK_CUTOFF,
    strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
) where {S<:AbstractFloat}
    cutoff_value = _check_qr_block_cutoff(cutoff)
    strassen_cutoff_value = _check_qr_strassen_cutoff(strassen_cutoff)

    r = min(size(A)...)
    maximum_order = r > cutoff_value ? div(r, 2) : 0
    multiply_work = _householder_block_multiply_workspace(
        S,
        maximum_order,
        strassen_cutoff_value,
    )

    return householder_block!(
        A,
        compact_T,
        block_work,
        vector_work,
        multiply_work;
        cutoff = cutoff_value,
        strassen_cutoff = strassen_cutoff_value,
    )
end

# Fully workspace-controlled recursive kernel.  This method is deliberately
# internal, but it is useful to the enclosing package if repeated QR
# factorizations must avoid all structural workspace allocation.
function householder_block!(
    A::AbstractMatrix{S},
    compact_T::AbstractMatrix{S},
    block_work::AbstractMatrix{S},
    vector_work::AbstractVector{S},
    multiply_work::_HouseholderBlockMultiplyWorkspace{S};
    cutoff::Integer = DEFAULT_QR_BLOCK_CUTOFF,
    strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
) where {S<:AbstractFloat}
    Base.require_one_based_indexing(
        A,
        compact_T,
        block_work,
        vector_work,
        multiply_work.left,
        multiply_work.right,
        multiply_work.product,
        multiply_work.strassen,
    )

    cutoff_value = _check_qr_block_cutoff(cutoff)
    strassen_cutoff_value = _check_qr_strassen_cutoff(strassen_cutoff)

    m, n = size(A)
    r = min(m, n)

    size(compact_T, 1) >= r ||
        throw(DimensionMismatch("compact_T has too few rows"))
    size(compact_T, 2) >= r ||
        throw(DimensionMismatch("compact_T has too few columns"))
    length(vector_work) >= r ||
        throw(DimensionMismatch(
            "vector_work has length $(length(vector_work)); at least $r is required",
        ))

    split = r > cutoff_value ? div(r, 2) : 0
    required_columns = iszero(split) ? 0 : n - split

    size(block_work, 1) >= split ||
        throw(DimensionMismatch(
            "block_work has $(size(block_work, 1)) rows; at least $split are required",
        ))
    size(block_work, 2) >= required_columns ||
        throw(DimensionMismatch(
            "block_work has $(size(block_work, 2)) columns; " *
            "at least $required_columns are required",
        ))

    maximum_order = split
    for padded in (multiply_work.left,
                   multiply_work.right,
                   multiply_work.product)
        size(padded, 1) >= maximum_order ||
            throw(DimensionMismatch(
                "padded multiplication workspace has too few rows",
            ))
        size(padded, 2) >= maximum_order ||
            throw(DimensionMismatch(
                "padded multiplication workspace has too few columns",
            ))
    end

    required_strassen_length = iszero(maximum_order) ? 0 :
        strassen_workspace_length(
            maximum_order;
            cutoff = strassen_cutoff_value,
        )
    length(multiply_work.strassen) >= required_strassen_length ||
        throw(DimensionMismatch(
            "Strassen workspace has length $(length(multiply_work.strassen)); " *
            "at least $required_strassen_length is required",
        ))

    # The lower triangle is not part of the compact-WY representation, but
    # clearing the complete active block makes reused storage deterministic.
    for j in 1:r
        for i in 1:r
            compact_T[i, j] = zero(S)
        end
    end

    _householder_block_recursive!(
        A,
        compact_T,
        block_work,
        vector_work,
        multiply_work,
        cutoff_value,
        strassen_cutoff_value,
    )

    return (factors = A, T = compact_T)
end

# Recursive kernel.  If r = min(m,n), the first floor(r/2) reflectors are
# obtained from the left block.  They are applied to every remaining column,
# after which the lower-right subproblem supplies the remaining reflectors.
function _householder_block_recursive!(
    A::AbstractMatrix{S},
    compact_T::AbstractMatrix{S},
    block_work::AbstractMatrix{S},
    vector_work::AbstractVector{S},
    multiply_work::_HouseholderBlockMultiplyWorkspace{S},
    cutoff::Int,
    strassen_cutoff::Int,
) where {S<:AbstractFloat}
    m, n = size(A)
    r = min(m, n)

    if r <= cutoff
        householder!(A, compact_T, view(vector_work, 1:r))
        return nothing
    end

    left_count = div(r, 2)
    right_count = r - left_count
    trailing_columns = n - left_count

    left_A = view(A, :, 1:left_count)
    left_T = view(compact_T, 1:left_count, 1:left_count)
    _householder_block_recursive!(
        left_A,
        left_T,
        block_work,
        vector_work,
        multiply_work,
        cutoff,
        strassen_cutoff,
    )

    update_work = view(block_work, 1:left_count, 1:trailing_columns)
    _householder_block_apply_left!(
        A,
        compact_T,
        left_count,
        update_work,
        multiply_work,
        strassen_cutoff,
    )

    right_A = view(A, left_count + 1:m, left_count + 1:n)
    right_T = view(
        compact_T,
        left_count + 1:r,
        left_count + 1:r,
    )
    _householder_block_recursive!(
        right_A,
        right_T,
        block_work,
        vector_work,
        multiply_work,
        cutoff,
        strassen_cutoff,
    )

    combine_work = view(block_work, 1:left_count, 1:right_count)
    _householder_block_combine!(
        A,
        compact_T,
        left_count,
        right_count,
        combine_work,
        multiply_work,
        strassen_cutoff,
    )

    return nothing
end

# Apply Q_left' to all columns to the right of the first `left_count`
# columns.  With
#
#     Q_left = I - V_left*T_left*V_left',
#
# this performs
#
#     A_right <- A_right - V_left*T_left'*(V_left'*A_right).
function _householder_block_apply_left!(
    A::AbstractMatrix{S},
    compact_T::AbstractMatrix{S},
    left_count::Int,
    work::AbstractMatrix{S},
    multiply_work::_HouseholderBlockMultiplyWorkspace{S},
    strassen_cutoff::Int,
) where {S<:AbstractFloat}
    m, n = size(A)
    trailing_columns = n - left_count

    # work <- L_left' * A_top, where L_left is unit lower triangular.
    for j in 1:trailing_columns
        global_column = left_count + j
        for i in 1:left_count
            value = A[i, global_column]
            for row in i + 1:left_count
                value += A[row, i] * A[row, global_column]
            end
            work[i, j] = value
        end
    end

    # work <- work + V_bottom' * A_bottom.
    V_bottom = view(A, left_count + 1:m, 1:left_count)
    A_bottom = view(A, left_count + 1:m, left_count + 1:n)
    _householder_block_mul!(
        work,
        transpose(V_bottom),
        A_bottom,
        one(S),
        one(S),
        multiply_work,
        strassen_cutoff,
    )

    # work <- T_left' * work.  T_left' is lower triangular; processing rows
    # from bottom to top makes the in-place multiplication safe.
    for j in 1:trailing_columns
        for i in left_count:-1:1
            value = zero(S)
            for row in 1:i
                value += compact_T[row, i] * work[row, j]
            end
            work[i, j] = value
        end
    end

    # A_bottom <- A_bottom - V_bottom * work.
    _householder_block_mul!(
        A_bottom,
        V_bottom,
        work,
        -one(S),
        one(S),
        multiply_work,
        strassen_cutoff,
    )

    # A_top <- A_top - L_left * work.
    for j in 1:trailing_columns
        global_column = left_count + j
        for i in 1:left_count
            value = work[i, j]
            for column in 1:i - 1
                value += A[i, column] * work[column, j]
            end
            A[i, global_column] -= value
        end
    end

    return nothing
end

# This is the same five-step compact-WY application as
# _householder_block_apply_left!, but targeting a matrix separate from the one
# that stores the reflectors, so that Q' can be applied to an unrelated block.

"""
    apply_Qt!(C, factors, compact_T;
        reflectors = min(size(factors)...),
        strassen_cutoff = DEFAULT_STRASSEN_CUTOFF)

Overwrite `C` with `transpose(Q) * C`, where

    Q = I - V*T*transpose(V)

is the orthogonal factor of the packed compact-WY factorization returned by
[`householder`](@ref) or [`householder_block`](@ref).  `V` is read from
`factors`: column `j` is zero above row `j`, one at row `j`, and `factors[i, j]`
below.  Only the leading `reflectors` columns of `V` and the leading
`reflectors`-by-`reflectors` block of `compact_T` are used, so a partial
application is obtained by passing a smaller `reflectors`.

`C` must have as many rows as `factors`.  Neither `factors` nor `compact_T` is
modified.

Since `transpose(Q) = I - V*transpose(T)*transpose(V)`, the application is

    W <- transpose(V) * C
    W <- transpose(T) * W
    C <- C - V * W

with the two `V` products split into their triangular top and dense bottom
parts, the latter going through [`strassen!`](@ref) via the same blocked
multiplication the QR recursion uses.

    apply_Qt!(C, factors, compact_T, reflectors, work, multiply_work, strassen_cutoff)

Workspace-controlled form.  `work` must be at least `reflectors`-by-`size(C, 2)`,
and `multiply_work` must have order at least
`min(reflectors, size(C, 1) - reflectors, size(C, 2))`.
"""
function apply_Qt!(
    C::AbstractMatrix{S},
    factors::AbstractMatrix{S},
    compact_T::AbstractMatrix{S};
    reflectors::Integer = min(size(factors)...),
    strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
) where {S<:AbstractFloat}
    k = Int(reflectors)
    m, n = size(C)

    work = Matrix{S}(undef, k, n)
    order = max(0, min(k, m - k, n))
    multiply_work = _householder_block_multiply_workspace(
        S, order, Int(strassen_cutoff),
    )

    return apply_Qt!(
        C, factors, compact_T, k, work, multiply_work, Int(strassen_cutoff),
    )
end

function apply_Qt!(
    C::AbstractMatrix{S},
    factors::AbstractMatrix{S},
    compact_T::AbstractMatrix{S},
    reflectors::Int,
    work::AbstractMatrix{S},
    multiply_work::_HouseholderBlockMultiplyWorkspace{S},
    strassen_cutoff::Int,
) where {S<:AbstractFloat}
    Base.require_one_based_indexing(C, factors, compact_T, work)

    m, n = size(C)
    k = reflectors

    size(factors, 1) == m || throw(DimensionMismatch(
        "C has $m rows, but factors has $(size(factors, 1))",
    ))
    0 <= k <= min(size(factors)...) || throw(DimensionMismatch(
        "cannot apply $k reflectors from a $(size(factors)) factorization",
    ))
    size(compact_T, 1) >= k || throw(DimensionMismatch("compact_T has too few rows"))
    size(compact_T, 2) >= k || throw(DimensionMismatch("compact_T has too few columns"))

    (iszero(k) || iszero(n)) && return C

    size(work, 1) >= k || throw(DimensionMismatch("work has too few rows"))
    size(work, 2) >= n || throw(DimensionMismatch("work has too few columns"))

    W = view(work, 1:k, 1:n)

    # W <- transpose(L) * C_top, where L is the unit lower triangular top block
    # of V.  Its implicit unit diagonal supplies the leading term.
    for j in 1:n
        for i in 1:k
            value = C[i, j]
            for row in i + 1:k
                value += factors[row, i] * C[row, j]
            end
            W[i, j] = value
        end
    end

    V_bottom = view(factors, k + 1:m, 1:k)
    C_bottom = view(C, k + 1:m, 1:n)

    # W <- W + transpose(V_bottom) * C_bottom.
    _householder_block_mul!(
        W,
        transpose(V_bottom),
        C_bottom,
        one(S),
        one(S),
        multiply_work,
        strassen_cutoff,
    )

    # W <- transpose(T) * W.  transpose(T) is lower triangular, so descending
    # rows makes the in-place multiplication safe: row i is written only after
    # every row that still needs to read it has been consumed.
    for j in 1:n
        for i in k:-1:1
            value = zero(S)
            for row in 1:i
                value += compact_T[row, i] * W[row, j]
            end
            W[i, j] = value
        end
    end

    # C_bottom <- C_bottom - V_bottom * W.
    _householder_block_mul!(
        C_bottom,
        V_bottom,
        W,
        -one(S),
        one(S),
        multiply_work,
        strassen_cutoff,
    )

    # C_top <- C_top - L * W, again with L's unit diagonal implicit.
    for j in 1:n
        for i in k:-1:1
            value = W[i, j]
            for column in 1:i - 1
                value += factors[i, column] * W[column, j]
            end
            C[i, j] -= value
        end
    end

    return C
end

# Combine
#
#     Q_left = I - V_left*T_left*V_left'
#
# and the embedded right transformation
#
#     Q_right = I - V_right*T_right*V_right'
#
# into
#
#     Q_left*Q_right = I - V*T*V',
#
# where V = [V_left V_right] and
#
#     T12 = -T_left * (V_left' * V_right) * T_right.
function _householder_block_combine!(
    A::AbstractMatrix{S},
    compact_T::AbstractMatrix{S},
    left_count::Int,
    right_count::Int,
    work::AbstractMatrix{S},
    multiply_work::_HouseholderBlockMultiplyWorkspace{S},
    strassen_cutoff::Int,
) where {S<:AbstractFloat}
    m = size(A, 1)

    # work <- B_top' * L_right, where B_top comprises the first right_count
    # rows of the lower part of V_left and L_right is unit lower triangular.
    for j in 1:right_count
        for i in 1:left_count
            value = A[left_count + j, i]
            for row in j + 1:right_count
                value += A[left_count + row, i] *
                         A[left_count + row, left_count + j]
            end
            work[i, j] = value
        end
    end

    # Add the contribution from rows below the unit lower-triangular part of
    # V_right.
    first_bottom_row = left_count + right_count + 1
    if first_bottom_row <= m
        left_bottom = view(A, first_bottom_row:m, 1:left_count)
        right_bottom = view(
            A,
            first_bottom_row:m,
            left_count + 1:left_count + right_count,
        )
        _householder_block_mul!(
            work,
            transpose(left_bottom),
            right_bottom,
            one(S),
            one(S),
            multiply_work,
            strassen_cutoff,
        )
    end

    # work <- T_left * work.  T_left is upper triangular; top-to-bottom row
    # order is safe for this in-place multiplication.
    for j in 1:right_count
        for i in 1:left_count
            value = zero(S)
            for column in i:left_count
                value += compact_T[i, column] * work[column, j]
            end
            work[i, j] = value
        end
    end

    # work <- -work * T_right.  T_right is upper triangular; processing
    # columns from right to left preserves all source columns still needed.
    for j in right_count:-1:1
        for i in 1:left_count
            value = zero(S)
            for column in 1:j
                value += work[i, column] *
                         compact_T[left_count + column, left_count + j]
            end
            work[i, j] = -value
        end
    end

    for j in 1:right_count
        for i in 1:left_count
            compact_T[i, left_count + j] = work[i, j]
        end
    end

    return nothing
end

# Compute
#
#     C <- alpha*A*B + beta*C
#
# using the package's square Strassen implementation.  Rectangular products
# are reduced to square products by blocking all three dimensions with
#
#     block_order = min(m, k, n),
#
# where A is m-by-k and B is k-by-n.  Edge blocks are zero-padded.  Thus the
# square Strassen routine is used without requiring it to implement a native
# rectangular recursion.
function _householder_block_mul!(
    C::AbstractMatrix{S},
    A::AbstractMatrix{S},
    B::AbstractMatrix{S},
    alpha::S,
    beta::S,
    work::_HouseholderBlockMultiplyWorkspace{S},
    strassen_cutoff::Int,
) where {S<:AbstractFloat}
    Base.require_one_based_indexing(C, A, B)

    m, k = size(A)
    kb, n = size(B)
    k == kb || throw(DimensionMismatch(
        "A has $k columns, but B has $kb rows",
    ))
    size(C) == (m, n) || throw(DimensionMismatch(
        "C has size $(size(C)); expected ($m, $n)",
    ))

    if iszero(m) || iszero(k) || iszero(n)
        _householder_scale_matrix!(C, beta)
        return C
    end

    block_order = min(m, k, n)

    # There is no benefit in packing square blocks for products which the
    # Strassen routine would immediately send to its classical leaf.
    if block_order <= strassen_cutoff
        LinearAlgebra.mul!(C, A, B, alpha, beta)
        return C
    end

    size(work.left, 1) >= block_order ||
        throw(DimensionMismatch("multiplication workspace is too small"))
    size(work.left, 2) >= block_order ||
        throw(DimensionMismatch("multiplication workspace is too small"))
    size(work.right, 1) >= block_order ||
        throw(DimensionMismatch("multiplication workspace is too small"))
    size(work.right, 2) >= block_order ||
        throw(DimensionMismatch("multiplication workspace is too small"))
    size(work.product, 1) >= block_order ||
        throw(DimensionMismatch("multiplication workspace is too small"))
    size(work.product, 2) >= block_order ||
        throw(DimensionMismatch("multiplication workspace is too small"))

    _householder_scale_matrix!(C, beta)
    iszero(alpha) && return C

    zero_value = zero(S)
    add_product = isone(alpha)
    subtract_product = alpha == -one(S)

    for column_start in 1:block_order:n
        output_columns = min(block_order, n - column_start + 1)

        for inner_start in 1:block_order:k
            inner_columns = min(block_order, k - inner_start + 1)

            for row_start in 1:block_order:m
                output_rows = min(block_order, m - row_start + 1)
                order = max(output_rows, inner_columns, output_columns)

                left_pad = view(work.left, 1:order, 1:order)
                right_pad = view(work.right, 1:order, 1:order)
                product_pad = view(work.product, 1:order, 1:order)

                fill!(left_pad, zero_value)
                fill!(right_pad, zero_value)

                for j in 1:inner_columns
                    for i in 1:output_rows
                        left_pad[i, j] =
                            A[row_start + i - 1, inner_start + j - 1]
                    end
                end

                for j in 1:output_columns
                    for i in 1:inner_columns
                        right_pad[i, j] =
                            B[inner_start + i - 1, column_start + j - 1]
                    end
                end

                strassen!(
                    product_pad,
                    left_pad,
                    right_pad;
                    cutoff = strassen_cutoff,
                    workspace = work.strassen,
                )

                for j in 1:output_columns
                    for i in 1:output_rows
                        row = row_start + i - 1
                        column = column_start + j - 1
                        if add_product
                            C[row, column] += product_pad[i, j]
                        elseif subtract_product
                            C[row, column] -= product_pad[i, j]
                        else
                            C[row, column] += alpha * product_pad[i, j]
                        end
                    end
                end
            end
        end
    end

    return C
end

function _householder_scale_matrix!(
    A::AbstractMatrix{S},
    factor::S,
) where {S<:AbstractFloat}
    if iszero(factor)
        fill!(A, zero(S))
    elseif !isone(factor)
        for index in eachindex(A)
            A[index] *= factor
        end
    end

    return A
end

function _check_qr_block_cutoff(cutoff::Integer)
    cutoff >= 1 || throw(ArgumentError("cutoff must be at least one"))
    return Int(cutoff)
end

function _check_qr_strassen_cutoff(cutoff::Integer)
    cutoff >= 1 ||
        throw(ArgumentError("strassen_cutoff must be at least one"))
    return Int(cutoff)
end

