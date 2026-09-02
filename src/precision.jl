# precision.jl
#
# Precision discipline and low-level MPFR helpers.
#
# Every BigFloat carries its own MPFR precision, while ordinary Julia BigFloat
# arithmetic creates results at the current default precision.  A
# `Matrix{BigFloat}` therefore has a uniform precision only when the code that
# constructs and updates it maintains that invariant explicitly.
#
# Package kernels that allocate or compute BigFloat values establish the working
# precision with `with_precision`.  Internal in-place MPFR helpers write into an
# existing destination at that destination's precision.  `uniform_precision` and
# `assert_precision` are available at boundaries where uniformity is required.
#
# `with_precision` changes Julia's global BigFloat default and is intentionally
# used only by the serial implementation.  Threaded code will need a task-safe
# precision policy.

# ---------------------------------------------------------------------------
# In-place MPFR arithmetic
# ---------------------------------------------------------------------------
#
# Every `BigFloat` operation in Julia allocates a fresh result, and the trailing
# update of this factorisation touches O(m*n^2) entries with several operations
# each, so value semantics would spend most of the run in the allocator.
#
# `Base.MPFR` exposes no in-place arithmetic (only `nextfloat!`, `prevfloat!`
# and the exponent-range setters), so these go straight to the C library. Each
# writes its result into an existing `BigFloat`, at that value's own precision.
#
# The rounding mode is fixed to round-to-nearest rather than read from
# `Base.MPFR.ROUNDING_MODE` on every call. Nothing in this package changes the
# rounding mode, and that global may be resolved through a scoped lookup, which
# would reintroduce per-operation overhead.
#
# !!! warning "Aliasing"
#     These mutate the `BigFloat` OBJECT, not a matrix slot. Two entries holding
#     the same object would both change. Only apply them to values this file
#     owns: the entries of `R`, which are freshly constructed per entry by
#     `_to_float`, and the temporaries in `_fused_temporaries`. In particular do
#     NOT use them on `B`, `U` or `tau`: `B` and `U` belong to the caller and may
#     share objects with a copy the caller still needs, and `tau` is initialised
#     with `fill!`, which puts one shared object in every slot.

const _MPFR_ROUND = Base.MPFR.MPFRRoundNearest

@inline function _mpfr_set!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_set, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          z, x, _MPFR_ROUND)
    return z
end

@inline function _mpfr_zero!(z::BigFloat)
    ccall((:mpfr_set_zero, Base.MPFR.libmpfr), Cvoid, (Ref{BigFloat}, Cint), z, 1)
    return z
end

for (name, symbol) in ((:_mpfr_mul!, :mpfr_mul), (:_mpfr_add!, :mpfr_add),
                       (:_mpfr_sub!, :mpfr_sub), (:_mpfr_div!, :mpfr_div))
    @eval @inline function $name(z::BigFloat, x::BigFloat, y::BigFloat)
        ccall(($(QuoteNode(symbol)), Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              z, x, y, _MPFR_ROUND)
        return z
    end
end

for (name, symbol) in ((:_mpfr_sqrt!, :mpfr_sqrt), (:_mpfr_neg!, :mpfr_neg))
    @eval @inline function $name(z::BigFloat, x::BigFloat)
        ccall(($(QuoteNode(symbol)), Base.MPFR.libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
              z, x, _MPFR_ROUND)
        return z
    end
end

"`z = x*y + w`, in one correctly rounded step."
@inline function _mpfr_fma!(z::BigFloat, x::BigFloat, y::BigFloat, w::BigFloat)
    ccall((:mpfr_fma, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat},
           Base.MPFR.MPFRRoundingMode),
          z, x, y, w, _MPFR_ROUND)
    return z
end

"`z = 1 / x`."
@inline function _mpfr_inv!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_ui_div, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Culong, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          z, Culong(1), x, _MPFR_ROUND)
    return z
end

"`z = round(x)`, to nearest with ties away from zero, as `mpfr_round` defines."
@inline function _mpfr_round!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_round, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}), z, x)
    return z
end


"""
    with_precision(f, bits)

Run `f()` with the `BigFloat` default precision set to `bits`, restoring the
previous value afterwards, including on exception.

Use this rather than `setprecision(BigFloat, bits) do ... end` anywhere
performance matters. The block form installs a `ScopedValue`, and every
subsequent `BigFloat` allocation then resolves the current precision through a
`PersistentDict` lookup — ruinous in code that allocates temporaries in inner
loops, which all of the arbitrary-precision kernels here do.

Setting the default directly and restoring it in a `finally` gives identical
semantics for single-threaded code at a fraction of the cost.

!!! warning "Not task-safe"
    Unlike the block form, this sets a global default rather than a
    task-local one. Two tasks running this concurrently at different precisions
    would interfere. This package is single-threaded by design; if that ever
    changes, revisit every call site.
"""
function with_precision(f, bits::Integer)
    previous = precision(BigFloat)
    setprecision(BigFloat, Int(bits))
    try
        return f()
    finally
        setprecision(BigFloat, previous)
    end
end

"""
    uniform_precision(A) -> Union{Int, Nothing}

The common precision of every entry of `A`, or `nothing` if the entries
disagree. An empty matrix reports the current default.

There is no cheaper way to ask this: precision is a property of each entry, not
of the array, so the check is linear in the number of entries. It is intended
for assertions and tests, not for inner loops.
"""
function uniform_precision(A::AbstractArray{BigFloat})
    isempty(A) && return precision(BigFloat)
    first_precision = precision(first(A))
    for value in A
        precision(value) == first_precision || return nothing
    end
    return first_precision
end

"""
    assert_precision(p, matrices...; names=nothing)

Throw unless every entry of every given matrix has precision `p`.

This catches the characteristic failure of the rebinding semantics: a matrix
that was built at one precision, then written to outside a `setprecision` block
and silently widened.
"""
function assert_precision(p::Integer, matrices::AbstractArray{BigFloat}...;
                          names = nothing)
    for (index, matrix) in enumerate(matrices)
        label = names === nothing ? "argument $index" : names[index]
        found = uniform_precision(matrix)
        found === nothing && throw(ArgumentError(
            "$label has entries of differing precision; it was probably " *
            "written to outside a setprecision block"))
        found == p || throw(ArgumentError(
            "$label has precision $found, expected $p"))
    end
    return nothing
end

"""
    at_precision(A, p) -> Matrix{BigFloat}

An independent copy of `A` with every entry rounded to exactly `p` bits.

Use this at a boundary between two computations running at different
precisions, such as passing an R-factor down a recursion. Rounding is explicit
and per entry, so the result is uniform regardless of what `A` was.

The entries are freshly allocated and then written through `_mpfr_set!`, rather
than built with `BigFloat(x; precision = p)`. That constructor returns `x`
ITSELF when the requested precision already matches, so it cannot be used to
obtain an independent copy — the result would share every value with the input,
and an in-place kernel writing through one would be visible through the other.
"""
function at_precision(A::AbstractMatrix{BigFloat}, p::Integer)
    bits = Int(p)
    result = Matrix{BigFloat}(undef, size(A)...)
    for index in eachindex(A)
        result[index] = _mpfr_set!(BigFloat(; precision = bits), A[index])
    end
    return result
end

"""
    bigfloat_matrix(M, p) -> Matrix{BigFloat}

The integer matrix `M` rounded to `BigFloat` at `p` bits.

The analogue of `float_matrix` for a plain matrix with no column exponents.
Each entry is rounded once, on conversion.
"""
function bigfloat_matrix(M::AbstractMatrix{<:Integer}, p::Integer)
    result = Matrix{BigFloat}(undef, size(M)...)
    for index in eachindex(M)
        result[index] = BigFloat(M[index]; precision = Int(p))
    end
    return result
end

"""
    float64_matrix(M) -> (F, shift)

The integer matrix `M` rounded to `Float64` with a common power-of-two scale,
so that `M ≈ F * 2^shift` entrywise.

A direct `Float64(::BigInt)` overflows to `Inf` on lattice-sized entries, so the
scale is pulled out first: `shift` is chosen to bring the largest entry into the
range where `Float64` has full relative accuracy. A zero matrix gives a zero
shift.

The shift is common to the whole matrix rather than per column, so that ratios
between entries -- which is all the size-reduction multipliers depend on -- are
unaffected by it.
"""
function float64_matrix(M::AbstractMatrix{<:Integer})
    largest = 0
    for value in M
        iszero(value) && continue
        largest = max(largest, ndigits(value; base = 2))
    end

    # Leave a little headroom below the 53 bit significand so that the
    # subsequent arithmetic has room before it starts losing bits.
    shift = max(0, largest - 48)

    result = Matrix{Float64}(undef, size(M)...)
    for index in eachindex(M)
        result[index] = Float64(M[index] >> shift)
    end
    return result, shift
end

"""
    safe_exponent(x) -> Int

`exponent(x)` with zero mapped to `typemin(Int)` rather than a `DomainError`.

The refinement loops compare multiplier magnitudes by exponent, and a multiplier
of exactly zero is an ordinary outcome there, not an error.
"""
safe_exponent(x::AbstractFloat) = iszero(x) ? typemin(Int) : exponent(x)
