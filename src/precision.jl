# precision.jl
#
# Conventions and helpers for working at a fixed BigFloat precision.
#
# Julia's BigFloat is MPFR, and every BigFloat carries its own precision, but
# Julia does NOT use MPFR's destination-driven rounding rule.  Measured on
# 1.12.6:
#
#   * arithmetic allocates its result at the current GLOBAL DEFAULT precision,
#     ignoring the operands entirely;
#   * `setindex!` on a `Matrix{BigFloat}` rebinds the reference without
#     re-rounding to the destination entry's precision;
#   * `mul!` therefore widens its output matrix rather than preserving it.
#
# Together these mean a `Matrix{BigFloat}` has no single precision as a matter
# of type or structure -- only as a matter of discipline.  Two entries of the
# same matrix can differ.
#
# THE CONVENTION
# --------------
# 1. Public entry points that do BigFloat work take (or derive) a precision and
#    wrap their ENTIRE body in `setprecision(BigFloat, p) do ... end`.  Wrapping
#    only the allocations is not enough: it is the arithmetic that reads the
#    default.
#
# 2. Internal routines assume they are already inside such a block and never
#    wrap.  Nesting is harmless but obscures where precision is decided.
#
# 3. A `Matrix{BigFloat}` returned from a block keeps its entries' precision,
#    because precision travels with the object.  It is safe to store and pass
#    around.  What is NOT safe is doing further arithmetic on it outside a
#    matching block -- that silently produces results at the ambient default.
#    Callers receiving such a matrix across a recursion boundary should convert
#    it explicitly with `at_precision` rather than assume.
#
# 4. Use `assert_precision` at the entry of routines that require uniformity,
#    mirroring flatter's
#        assert(R.prec() == r_col.prec() && R.prec() == tau.prec());
#    which is the same check, made cheap by MPFR's per-object precision.
#
# 5. None of this applies to the Float64 kernels.

"""
    with_precision(f, bits)

Run `f()` with the `BigFloat` default precision set to `bits`, restoring the
previous value afterwards, including on exception.

Use this rather than `setprecision(BigFloat, bits) do ... end` anywhere
performance matters. The block form installs a `ScopedValue`, and every
subsequent `BigFloat` allocation then resolves the current precision through a
`PersistentDict` lookup. Profiling the reduction driver found that lookup
accounting for **60-64% of total runtime** — more than every matrix product,
factorisation and compression combined — because the inner loops of the fused
QR allocate temporaries constantly.

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

A copy of `A` with every entry rounded to exactly `p` bits.

This is the conversion to use at a boundary between two computations running at
different precisions, such as passing an R-factor down a recursion. Rounding is
explicit and per entry, so the result is uniform regardless of what `A` was.
"""
function at_precision(A::AbstractMatrix{BigFloat}, p::Integer)
    result = Matrix{BigFloat}(undef, size(A)...)
    for index in eachindex(A)
        result[index] = BigFloat(A[index]; precision = Int(p))
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
