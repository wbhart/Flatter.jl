# recursive_reduction.jl
#
# The recursive lattice reduction driver.
#
# A port of flatter's `LatticeReductionImpl::RecursiveGeneric`
# (problems/lattice_reduction/recursive_generic.cpp), which is the skeleton all
# of flatter's reduction implementations are built on:
#
#     initialise, compress
#     loop until the goal is met:
#         pick a sublattice window
#         reduce that window recursively
#         apply the result, re-factor, re-compress
#     collect the accumulated transforms, size reduce once more
#
# The window schedule here is the hardcoded one from `Proved3`
# (proved_3.cpp) -- middle, left, right, cycling -- rather than the
# configurable `SublatticeSplit` tree that `Heuristic1/2/3` use. `Proved2` and
# `Proved3` reference no split object at all, which makes this the shortest
# route to a working end-to-end reducer. Swapping in the split tree later
# changes `_reduction_window` and nothing else in this file.
#
# What the recursion is for: reducing a lattice basis directly is expensive
# because the entries are enormous. Each level of this recursion compresses the
# basis -- discards low-order bits that provably cannot affect the result --
# reduces the smaller problem, then lifts the transform back. `collect_U`
# performs the lift, conjugating each level's transform by the scaling that was
# in force when it was computed.
#
# The two-column base case is exact: Lagrange for small entries, Schoenhage for
# large ones. Neither involves a precision policy.
#
# Contract: the input basis must be square, upper triangular, and nonsingular.
# That is what the recursion produces internally, and what flatter's `Proved2`
# and `Proved3` require. Bringing a general basis to that form is the job of
# `Irregular`/`CondUnknown`, which are not ported.

const DEFAULT_REDUCTION_RHF = 1.02
const DEFAULT_REDUCTION_MAX_ITERATIONS = 120

# A profile entry must move by more than this many bits for a cycle to count as
# having changed anything. The entries are base-two logarithms of magnitudes, so
# a movement far below one bit is noise from the floating point factorisation
# rather than progress.
const REDUCTION_STAGNATION_TOLERANCE = 1e-6

# At and above this dimension the reduction loop asks for an incremental
# collection once per iteration.
#
# Collection is worth roughly threefold in memory and costs a little time, so
# it is only worth paying where the memory matters. Measured excess with no
# collection at all: 44 MB at dimension 64, 580 MB at 96, 1141 MB at 128,
# 3092 MB at 160. Below about 128 there is nothing to manage, and collecting
# anyway is pure cost -- and a source of run-to-run variance.
#
# `BigInt` and `BigFloat` limbs are malloc'd through Julia's counted allocators
# and released by finalizers, and the collector's heuristics for malloc'd bytes
# are far less eager than for pooled objects. A reduction at high dimension
# discards an n-by-n arbitrary-precision matrix every iteration, so resident
# memory can climb well past the live set purely through lag. `GC.gc(false)` is
# a young-generation pass, which is where that garbage is, and is much cheaper
# than a full collection.
#
# Set `gc_dimension = typemax(Int)` to switch this off.
const REDUCTION_GC_DIMENSION = 128

# Iterations between collections, once the dimension threshold is met.
const REDUCTION_GC_INTERVAL = 2

# Bounded accumulation holds one running product rather than O(log k) partial
# products. It multiplies less efficiently -- an ever-growing accumulator
# against a fixed-width factor is the wrong shape for GMP -- so it is not worth
# paying for until the footprint is the binding constraint. Observed: the
# default settings reach about dimension 260, and bounded accumulation extends
# that to about 300.
const REDUCTION_LOW_MEMORY_DIMENSION = 256

# ---------------------------------------------------------------------------
# Logarithms of magnitudes
# ---------------------------------------------------------------------------

"""
    _log2_abs(x) -> Float64

`log2|x|`, for an integer of any size or a float of any precision. Returns
`-Inf` at zero rather than throwing, so a rank-deficient profile is detectable
rather than fatal at the point of measurement.
"""
function _log2_abs(x::Integer)
    iszero(x) && return -Inf
    magnitude = abs(x)
    bits = ndigits(magnitude; base = 2)
    bits <= 53 && return log2(Float64(magnitude))
    # Keep the top 53 bits and put the rest in the exponent.
    return log2(Float64(magnitude >> (bits - 53))) + (bits - 53)
end

_log2_abs(x::AbstractFloat) = iszero(x) ? -Inf : Float64(log2(abs(x)))

# ---------------------------------------------------------------------------
# The two-dimensional base case
# ---------------------------------------------------------------------------

"""
    lagrange_reduce!(B, U) -> (B, U)

Lagrange (Gauss) reduction of a basis of at most two columns, in place.

A port of flatter's `LatticeReductionImpl::Lagrange` (lagrange.cpp), but carried
out entirely in exact integer arithmetic: the squared lengths and inner products
are integers, and the multiplier is an exact rounded quotient, so no floating
point and no precision policy is involved. flatter needs MPFR there only because
it shares the code path with a float basis.

This is the base case of the recursion. On return the columns are ordered by
increasing length and the second is reduced against the first.
"""
function lagrange_reduce!(B::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T<:Integer}
    m, n = size(B)
    n <= 2 || throw(DimensionMismatch("Lagrange reduction needs at most two columns, got $n"))
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))

    _set_identity!(U)
    n <= 1 && return B, U

    squared_length(column) = sum(B[i, column] * B[i, column] for i in 1:m)
    inner_product() = sum(B[i, 1] * B[i, 2] for i in 1:m)

    swap_columns!() = begin
        for i in 1:m
            B[i, 1], B[i, 2] = B[i, 2], B[i, 1]
        end
        for i in 1:n
            U[i, 1], U[i, 2] = U[i, 2], U[i, 1]
        end
    end

    first_length = squared_length(1)
    second_length = squared_length(2)
    if first_length < second_length
        swap_columns!()
        first_length, second_length = second_length, first_length
    end

    while true
        iszero(second_length) && break

        multiplier = _div_round(inner_product(), second_length)

        # Swap, then reduce the (now second) longer column by the shorter one.
        swap_columns!()
        first_length = second_length
        for i in 1:m
            B[i, 2] -= multiplier * B[i, 1]
        end
        for i in 1:n
            U[i, 2] -= multiplier * U[i, 1]
        end
        second_length = squared_length(2)

        first_length <= second_length && break
    end

    return B, U
end

# Below this many bits in the largest entry, the two-column base case uses
# Lagrange; at or above it, Schoenhage. Lagrange is quadratic in the bit size
# and Schoenhage quasi-linear, but Schoenhage carries the overhead of forming
# the Gram matrix and entering its recursion, so small inputs are faster the
# direct way. flatter makes the same split at `prec < 1400`.
const DEFAULT_SCHOENHAGE_THRESHOLD = 512

# Dimensions at or below this are handed straight to `fplll_reduce` instead of
# being recursed on. flatter does the same (n <= 32), and for good reason: the
# recursion's cost is multiplicative in the depth, since every window reduction
# at one level spawns a whole reduction loop at the next. Descending all the way
# to two columns turns a dimension-16 problem into tens of thousands of calls.
const DEFAULT_BASE_CUTOFF = 32

"""
    _reduce_two_columns!(B, U, threshold) -> (B, U)

The base case of the recursion: reduce a basis of at most two columns.

Routes to [`lagrange_reduce!`](@ref) for small entries and to
[`schoenhage`](@ref) for large ones. Both produce a Gauss-reduced basis — the
columns ordered by length with `2|<b1,b2>| <= <b1,b1>` — so the choice is
purely one of cost.

`schoenhage` works on the Gram matrix rather than the basis, so the Gram is
formed here (exactly, in integer arithmetic) and the resulting unimodular
transform applied back to the columns.
"""
function _reduce_two_columns!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                              threshold::Int) where {T<:Integer}
    m, n = size(B)
    n <= 1 && return lagrange_reduce!(B, U)

    largest = 0
    for value in B
        iszero(value) || (largest = max(largest, ndigits(value; base = 2)))
    end
    largest < threshold && return lagrange_reduce!(B, U)

    gram = Matrix{BigInt}(undef, 2, 2)
    for a in 1:2, b in 1:2
        gram[a, b] = sum(BigInt(B[i, a]) * BigInt(B[i, b]) for i in 1:m)
    end

    _, transform = schoenhage(gram)

    applied = T.(transform)
    updated = B * applied
    for j in 1:2, i in 1:m
        B[i, j] = updated[i, j]
    end
    for j in 1:2, i in 1:2
        U[i, j] = applied[i, j]
    end
    return B, U
end

"""
    ReductionTelemetry()

Optional counters and timings for [`lattice_reduce!`](@ref), for working out
where the time actually goes rather than guessing.

Pass one via the `telemetry` keyword; it is mutated in place, accumulating over
the whole recursion. When no telemetry object is given the driver does nothing
beyond a null check, so this costs nothing in normal use.

Times are in seconds and count only the phase named, so they do not sum to the
total: `time_recursion` covers the nested reduction calls and therefore overlaps
everything measured at deeper levels.
"""
# Defaults live on the fields, so adding one cannot desynchronise a positional
# constructor -- which it did, twice, while this struct was growing.
Base.@kwdef mutable struct ReductionTelemetry
    levels::Int = 0
    max_depth::Int = 0
    iterations::Int = 0
    capped::Int = 0
    stagnated::Int = 0
    dense_rounds::Int = 0
    profile_calls::Int = 0
    measure_memory::Bool = false
    peak_live::Int = 0
    peak_data::Int = 0
    base_cases::Int = 0
    lagrange_calls::Int = 0
    schoenhage_calls::Int = 0
    fplll_calls::Int = 0
    fused_calls::Int = 0
    time_fused::Float64 = 0.0
    time_matmul::Float64 = 0.0
    time_compress::Float64 = 0.0
    time_base::Float64 = 0.0
    time_recursion::Float64 = 0.0
    time_finalise::Float64 = 0.0
    time_collect::Float64 = 0.0
    time_apply::Float64 = 0.0
    time_final_sr::Float64 = 0.0
    time_dense_qr::Float64 = 0.0
    time_profile::Float64 = 0.0
    fused_split::Vector{Float64} = zeros(Float64, FUSED_TIME_SLOTS)
end


function Base.show(io::IO, t::ReductionTelemetry)
    println(io, "ReductionTelemetry:")
    println(io, "  recursion levels entered : ", t.levels)
    println(io, "  maximum depth            : ", t.max_depth)
    println(io, "  window reductions        : ", t.iterations)
    println(io, "  levels stopped by the cap: ", t.capped, " of ", t.levels)
    println(io, "  levels stopped stagnating: ", t.stagnated, " of ", t.levels)
    if t.dense_rounds > 0
        println(io, "  dense-path rounds        : ", t.dense_rounds)
        println(io, "  time in dense-path QR    : ",
                    round(t.time_dense_qr; digits = 3), " s")
    end
    if t.measure_memory
        println(io, "  peak live bytes seen     : ",
                    round(t.peak_live / 1024^2; digits = 1), " MB")
        println(io, "  peak data held (measured): ",
                    round(t.peak_data / 1024^2; digits = 1), " MB")
    end
    println(io, "  base cases               : ", t.base_cases,
                " (", t.lagrange_calls, " Lagrange, ", t.schoenhage_calls,
                " Schoenhage, ", t.fplll_calls, " fplll)")
    println(io, "  fused QR calls           : ", t.fused_calls)
    println(io, "  time in fused QR         : ", round(t.time_fused; digits = 3), " s")
    if t.time_fused > 0
        names = ("size reduction", "re-orthogonalise", "reflectors", "trailing update")
        for slot in 1:FUSED_TIME_SLOTS
            println(io, "    ", rpad(names[slot], 21), ": ",
                        round(t.fused_split[slot]; digits = 3), " s (",
                        round(100 * t.fused_split[slot] / t.time_fused; digits = 1), "%)")
        end
    end
    println(io, "  time in matrix products  : ", round(t.time_matmul; digits = 3), " s")
    println(io, "  time in compression      : ", round(t.time_compress; digits = 3), " s")
    println(io, "  time in base cases       : ", round(t.time_base; digits = 3), " s")
    println(io, "  time finalising          : ", round(t.time_finalise; digits = 3), " s")
    println(io, "    lifting transforms     : ", round(t.time_collect; digits = 3), " s")
    println(io, "    applying to the basis  : ", round(t.time_apply; digits = 3), " s")
    print(io,   "    final size reduction   : ", round(t.time_final_sr; digits = 3), " s")
end

# `nothing` telemetry makes each of these a no-op the compiler can remove.
@inline _tick() = time_ns()
@inline _tock(start) = (time_ns() - start) / 1e9


"""
    _reported_profile(B, aggressive, want_profile, depth, telemetry) -> Vector{Float64}

The profile to hand back in `info`, or an empty vector when nobody will read it.

Computing this needs a full arbitrary-precision factorisation of a basis that is
no longer triangular, which is not cheap: on a single-level run it can cost more
than the reduction did. Two things make it skippable. Recursive calls discard
the info they are given, so only the outermost one can read it at all; and a
caller who does not want a profile can say so, which also makes a comparison
against a bare reduction fair, since that computes no profile either.
"""
function _reported_profile(B::AbstractMatrix{<:Integer}, aggressive::Bool,
                           want_profile::Bool, depth::Int,
                           telemetry::Union{Nothing, ReductionTelemetry})
    (want_profile && depth == 0) || return Float64[]
    started = _tick()
    result = _basis_profile(B, aggressive)
    if telemetry !== nothing
        telemetry.time_profile += _tock(started)
        telemetry.profile_calls += 1
    end
    return result
end

"""
    _basis_profile(B, aggressive) -> Vector{Float64}

The Gram-Schmidt log-profile of a general integer basis, via its R factor.

Needed for bases that are not triangular -- the output of `fplll_reduce`, for
instance -- where the diagonal carries no information. Does not modify `B`.
"""
function _basis_profile(B::AbstractMatrix{<:Integer}, aggressive::Bool)
    n = size(B, 2)
    largest = 0
    for value in B
        iszero(value) || (largest = max(largest, ndigits(value; base = 2)))
    end
    bits = lll_precision(largest, n; aggressive = aggressive)
    return with_precision(bits) do
        factors, _ = householder_block(bigfloat_matrix(B, bits))
        [_log2_abs(factors[i, i]) for i in 1:n]
    end
end

# ---------------------------------------------------------------------------
# The window schedule
# ---------------------------------------------------------------------------

"""
    _reduction_window(iteration, n) -> UnitRange{Int}

The sublattice to reduce on a given iteration, cycling middle, left, right.

flatter's `Proved3::setup_sublattice_reductions`, translated to one-based
inclusive ranges. The middle window straddles the boundary the left and right
windows share, which is what lets the recursion move mass across that boundary;
without it the two halves would be reduced independently forever.
"""
function _reduction_window(iteration::Int, n::Int)
    phase = mod(iteration, 3)
    phase == 0 && return (div(n, 4) + 1):div(3n, 4)
    phase == 1 && return 1:div(n, 2)
    return (div(n, 2) + 1):n
end

# A goal check only makes sense at the end of a full cycle, once every window
# has had a turn. flatter's `Proved3::is_reduced` says the same with
# `iterations % 3 == 0`, and refuses at iteration zero so at least one cycle
# always runs.
_should_check_goal(iteration::Int) = iteration > 0 && mod(iteration, 3) == 0

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

# `compression_shifts` takes the target precision as an argument, but only its
# `truncation` output depends on it, and linearly: truncation = C - precision.
# The precision we want comes from the spread, which is not known until after
# the call. Asking for precision zero recovers C, so one call suffices.
function _compression_plan(prof::AbstractVector{Float64}, n::Int, aggressive::Bool)
    all(isfinite, prof) || throw(ArgumentError(
        "the profile has a non-finite entry; the basis is rank deficient or " *
        "the working precision has collapsed a diagonal entry to zero"))

    shifts, offset, spread = compression_shifts(prof, 0)
    precision = lll_precision(spread, n; aggressive = aggressive)
    truncation = offset - precision
    return (shifts .+ truncation), precision
end

# Initial compression, straight from the integer triangular basis. flatter goes
# through a float copy of M here; doing it on the integers is exact and saves a
# conversion. A non-negative shift floors, matching `_scale_to_integer` and
# flatter's `mpz_div_2exp`.
function _compress_integer(B::AbstractMatrix{T}, prof::Vector{Float64},
                           offsets::Vector{Float64}, n::Int,
                           aggressive::Bool) where {T<:Integer}
    applied, precision = _compression_plan(prof, n, aggressive)

    compressed = zeros(T, n, n)
    for j in 1:n
        shift = applied[j]
        for i in 1:j
            compressed[i, j] = shift >= 0 ? B[i, j] >> shift : B[i, j] << (-shift)
        end
        offsets[j] += shift
        prof[j] -= shift
    end
    return compressed, applied, precision
end

# Compression of the floating point R factor, which is what flatter's
# `compress_R` does. `R` arrives packed from the fused factorisation, so the
# below-diagonal Householder tails are dropped here rather than zeroed in place.
function _compress_factor(R::AbstractMatrix{S}, prof::Vector{Float64},
                          offsets::Vector{Float64}, n::Int, ::Type{T},
                          aggressive::Bool) where {S<:AbstractFloat, T<:Integer}
    applied, precision = _compression_plan(prof, n, aggressive)

    compressed = zeros(T, n, n)
    for j in 1:n
        shift = applied[j]
        for i in 1:j
            compressed[i, j] = round(T, ldexp(R[i, j], -shift))
        end
        offsets[j] += shift
        prof[j] -= shift
    end
    return compressed, applied, precision
end

"""
    release_free_memory() -> Bool

Ask the C allocator to return free heap pages to the operating system.

Arbitrary-precision arithmetic allocates a limb block per value, a few dozen to
a few hundred bytes, and frees it almost immediately. glibc serves blocks that
size from the main heap and keeps them on free lists rather than returning
them, so resident memory tracks the high-water mark of the heap rather than
what is live -- measured at 300 times the actual data held. Neither garbage
collection nor a Julia heap-size hint touches this, because Julia has already
freed the memory; the allocator is holding it.

`malloc_trim` releases the free pages it can. Returns whether anything was
released, and is a no-op off glibc.
"""
function release_free_memory()
    @static if Sys.islinux()
        return ccall(:malloc_trim, Cint, (Csize_t,), 0) != 0
    else
        return false
    end
end

"""
    held_bytes(x) -> Int
    held_bytes(A) -> Int

Actual memory held by an arbitrary-precision value, or by every entry of an
array of them.

For `BigInt`, `x.alloc` is the number of 64-bit limbs GMP has reserved, which is
what is really held -- `x.size`, the number in use, can be much smaller after a
value shrinks. For `BigFloat` the significand is fixed at its precision. The
constant covers the Julia object header, the struct, and a typical malloc block
header.

Summing this over the live structures gives the working set of the DATA, which
compared against the process's resident size says how much of the footprint is
data and how much is garbage awaiting collection, or allocator fragmentation.
"""
held_bytes(x::BigInt) = 32 + 16 + 8 * abs(Int(x.alloc))

# A BigFloat's significand is allocated at its precision and never resized, so
# the limb count follows directly from `precision`.
held_bytes(x::BigFloat) = 32 + 16 + 8 * cld(precision(x), 64)

# Total over every integer type and container, so that instrumenting a
# reduction can never be what breaks it. Fixed-width integers are stored
# inline, so their cost is just the element width.
held_bytes(x::Integer) = sizeof(x)
held_bytes(x::AbstractFloat) = sizeof(x)
held_bytes(::Nothing) = 0

# Unassigned slots must be skipped, not indexed. `Matrix{BigInt}(undef, ...)`
# holds undefined REFERENCES, since BigInt is a mutable type, and a matrix can
# legitimately be in that state while a computation is in progress -- the
# transform `U` is only filled once the reduction finishes. Iterating such a
# matrix throws `UndefRefError`.
function held_bytes(A::AbstractArray)
    total = 0
    for index in eachindex(A)
        isassigned(A, index) || continue
        total += held_bytes(A[index])
    end
    return total
end
held_bytes(items::Vector{<:AbstractArray}) = sum(held_bytes, items; init = 0)

# ---------------------------------------------------------------------------
# Lifting the accumulated transforms
# ---------------------------------------------------------------------------

# flatter's `collect_U`. Each iteration's transform is computed against a
# COMPRESSED basis, so before it can be composed with the others it has to be
# conjugated by the scaling that was in force at the time -- the D^-1 U D of
# Saruchi-Morel-Stehle-Villard, which `conjugate_transform` performs.
#
# The scaling a transform needs is the one already in force when that transform
# is produced, so both steps happen inside the reduction loop and neither the
# raw transforms nor the compression history is ever retained. What is kept is a
# small stack of partial products, combined in the manner of a binary counter:
# pushing a factor merges it with the top of the stack for as long as the top
# represents the same number of original factors. That leaves O(log k) live
# matrices instead of k, which matters because these are the widest matrices in
# the run -- conjugation shifts their entries up by the compression amounts.
#
# The merge order is the natural one, so the product stays correctly ordered and
# balanced: it performs exactly the pairings a product tree over all k factors
# would, without needing all k at once.

# The stack has two modes, and the choice is a genuine trade.
#
# Balanced (the default) merges a pushed factor with the top for as long as the
# top represents the same number of original factors, performing exactly the
# pairings a product tree over all k factors would. GMP is fastest on operands
# of similar width, so this is the quicker option.
#
# Bounded folds every factor into a single running product. That is the slower
# multiplication pattern -- an ever-growing accumulator against a fixed-width
# factor -- but it holds ONE matrix instead of O(log k) partial products.
#
# The distinction matters more than the matrix count suggests. Entry widths add
# across a product, so a partial product of 2^j factors is about 2^j times as
# wide as one factor; summing over the balanced stack gives Theta(k) bits, the
# same order as keeping every factor. The bounded accumulator instead holds a
# single matrix whose entries are bounded by the FINAL transform, which for a
# lattice reduction is governed by the basis geometry rather than by k.

mutable struct _TransformStack{T}
    factors::Vector{Matrix{T}}   # partial products, oldest first
    widths::Vector{Int}          # how many original factors each represents
    bounded::Bool
end

_TransformStack{T}(bounded::Bool = false) where {T} =
    _TransformStack{T}(Matrix{T}[], Int[], bounded)

function _push_transform!(stack::_TransformStack{T}, factor::Matrix{T}) where {T}
    if stack.bounded
        if isempty(stack.factors)
            push!(stack.factors, factor)
            push!(stack.widths, 1)
        else
            stack.factors[1] = _reduction_mul(stack.factors[1], factor)
            stack.widths[1] += 1
        end
        return stack
    end

    width = 1
    while !isempty(stack.widths) && stack.widths[end] == width
        factor = _reduction_mul(pop!(stack.factors), factor)
        pop!(stack.widths)
        width *= 2
    end
    push!(stack.factors, factor)
    push!(stack.widths, width)
    return stack
end

"Conjugate a window transform by the scaling it was computed under."
function _lift_transform(U::AbstractMatrix{T}, shifts::Vector{Int}) where {T<:Integer}
    conjugated = conjugate_transform(U, shifts)
    # `conjugate_transform` already returns integer entries; converting again
    # would copy every entry for nothing.
    return conjugated isa Matrix{T} ? conjugated : T.(conjugated)
end

"""
    _drain_transform(stack, n) -> Matrix

Combine the partial products left on the stack into the final transform.

Folds by popping rather than by building each level of a product tree, because
this is where the matrices are at their widest and the peak here sets the
footprint of the whole run. A tree holds an entire level while constructing the
next, roughly doubling the live set at the worst possible moment; popping holds
two operands and a result, and releases each factor as it is consumed.

The tree is still the right structure while the stack is being built, where the
factors are narrow and the balanced pairings are what keep the multiplications
cheap. Only the final combination trades that for the smaller peak, and by then
there are only O(log k) factors left, so the difference in multiplication cost
is negligible.
"""
function _drain_transform(stack::_TransformStack{T}, n::Int) where {T}
    factors = stack.factors
    if isempty(factors)
        result = Matrix{T}(undef, n, n)
        _set_identity!(result)
        return result
    end

    # Popping from the back keeps the order: the stack holds the factors oldest
    # first, so this builds factors[1] * (factors[2] * (... * factors[end])).
    result = pop!(factors)
    empty!(stack.widths)
    while !isempty(factors)
        result = _reduction_mul(pop!(factors), result)
    end
    return result
end

# One place to choose the matrix product used throughout the driver. The
# BigInt-specific work lives in `strassen.jl`'s leaf multiplication, so this
# stays a plain call and every caller in the package benefits.
_reduction_mul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T} = strassen(A, B)

# The window transform embedded in an n x n identity is mostly identity: only
# the `window` block differs. Forming it and calling a general matrix product
# costs O(n^3) to compute something that only touches O(n * w^2) entries, and
# for a half-width window that is four times the necessary work. Both sides get
# a block-aware version, and the embedded matrix is never materialised.

"""
    _apply_window_right(A, W, window) -> Matrix

`A * E`, where `E` is the identity with `E[window, window] = W`.

Columns outside the window pass through unchanged; the rest is one
`n x w` by `w x w` product.
"""
function _apply_window_right(A::AbstractMatrix{T}, W::AbstractMatrix{T},
                             window::UnitRange{Int}) where {T<:Integer}
    result = Matrix{T}(A)
    result[:, window] = _reduction_mul(Matrix{T}(view(A, :, window)), W)
    return result
end

"""
    _apply_window_left(W, A, window) -> Matrix

`E * A`, where `E` is the identity with `E[window, window] = W`.

Rows outside the window pass through unchanged; the rest is one
`w x w` by `w x n` product.
"""
function _apply_window_left(W::AbstractMatrix{T}, A::AbstractMatrix{T},
                            window::UnitRange{Int}) where {T<:Integer}
    result = Matrix{T}(A)
    result[window, :] = _reduction_mul(W, Matrix{T}(view(A, window, :)))
    return result
end

# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

function _check_reduction_input(B::AbstractMatrix)
    m, n = size(B)
    m == n || throw(DimensionMismatch("expected a square basis, got $m x $n"))
    n >= 1 || throw(ArgumentError("the basis must be non-empty"))
    for j in 1:n
        iszero(B[j, j]) && throw(ArgumentError(
            "B has a zero diagonal entry at index $j; the basis is singular"))
        for i in (j + 1):n
            iszero(B[i, j]) || throw(ArgumentError(
                "B must be upper triangular; B[$i,$j] is nonzero"))
        end
    end
    return n
end

"""
    lattice_reduce!(B, U; kwargs...) -> (B, U, info)

Reduce the square upper triangular integer basis `B` in place, overwriting `U`
with the unimodular transform applied, so that on return the new `B` equals the
old `B` times `U`.

Returns `(B, U, info)` where `info` is a named tuple with

  * `iterations` -- how many window reductions ran at this level;
  * `goal_met`   -- whether the goal was satisfied;
  * `stopped`    -- why the loop ended: `:goal` if the goal was met,
                    `:stagnated` if a whole cycle of windows left the profile
                    unmoved, `:cap` if `max_iterations` was reached, or
                    `:base_case` if the dimension was small enough to hand
                    straight to a base-case reducer;
  * `profile`    -- `log2 |r_ii|` of the reduced basis, at true scale. Empty
                    for a base case reached recursively: computing it needs a
                    factorisation, and only the outermost caller reads it.

!!! note "The returned basis is not triangular"
    Input must be upper triangular, but output is a general reduced basis: the
    transform is unimodular and not triangular, so the diagonal of the returned
    `B` carries no meaning and may well contain zeros. Read the profile from
    `info.profile`, which comes from the R factor. To feed a result back into
    this function, re-triangularise it first — for instance with
    [`fused_qr_size_reduction!`](@ref) followed by rounding the R factor to
    integers.

    flatter behaves the same way: `RecursiveGeneric::fini_solver` leaves `M` as
    a general basis and reports the profile separately.

Keyword arguments:

  * `goal`           -- a [`ReductionGoal`](@ref). Defaults to
                        `goal_from_rhf(n, rhf)`.
  * `rhf`            -- target root Hermite factor when `goal` is not given.
                        Default `$(DEFAULT_REDUCTION_RHF)`.
  * `max_iterations` -- cap per recursion level. flatter has no cap; this exists
                        so a basis the goal cannot reach terminates rather than
                        spinning. Reaching it is not an error -- the returned
                        basis is still valid, just less reduced.
  * `aggressive`     -- passed to [`lll_precision`](@ref); halves the working
                        precision, faster but likelier to stall.
  * `low_memory`     -- fold accumulated transforms into a single running
                        product instead of a balanced tree of partial products.
                        Multiplies less efficiently, but holds one matrix rather
                        than several partial products at their widest. `nothing`
                        chooses it automatically at dimension
                        $(REDUCTION_LOW_MEMORY_DIMENSION) and above, which is
                        roughly where the footprint becomes the binding
                        constraint rather than the runtime.
  * `trim_memory`    -- after each periodic collection, ask the C allocator to
                        return free pages to the operating system. Measured
                        saving on this workload: none, so it is off by default.
                        Linux only; harmless elsewhere.
  * `gc_full`        -- make the periodic collection two full passes rather
                        than one incremental one. Arbitrary-precision limbs are
                        released by finalizers, and a finalizable object
                        survives one cycle, so only this form can reclaim them.
                        Expensive; measure before enabling.
  * `want_profile`   -- compute `info.profile`, which needs a factorisation of
                        the reduced basis. Only the outermost call can report
                        one, and on a single-level run it can cost more than the
                        reduction. Set `false` when the profile is not read.
  * `gc_dimension`   -- at or above this dimension, collect periodically.
                        `typemax(Int)` disables it.
  * `gc_interval`    -- iterations between collections. Measured at dimension
                        128 the runtime is flat across every setting while the
                        footprint varies threefold, so a small value is close to
                        free; `gc_tradeoff()` in the scaling probe re-measures
                        this.
  * `base_cutoff`    -- dimensions at or below this go straight to
                        [`fplll_reduce`](@ref) instead of being recursed on.
                        Default `$(DEFAULT_BASE_CUTOFF)`, matching flatter. The
                        recursion's cost is multiplicative in the depth, so
                        cutting it off early matters far more than it looks.
  * `telemetry`      -- an optional [`ReductionTelemetry`](@ref), mutated in
                        place with counts and per-phase timings.
  * `schoenhage_threshold` -- entry bit-length at which the two-column base
                        case switches from Lagrange to [`schoenhage`](@ref).
                        Both give a Gauss-reduced basis; Schoenhage is
                        quasi-linear in the bit size where Lagrange is
                        quadratic, so the crossover is a cost trade only.

# What is guaranteed

`B_out == B_in * U` exactly, and `U` is unimodular. Those hold regardless of
whether the goal was met, of the working precision, and of the iteration cap.
Reduction *quality* is what the goal and precision govern.

# Contract

The basis must be square, upper triangular and nonsingular. Reducing a general
basis means first bringing it to that form, which flatter does in `Irregular`
and `CondUnknown`; neither is ported here.
"""
function lattice_reduce!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
                         goal::Union{Nothing, ReductionGoal} = nothing,
                         rhf::Real = DEFAULT_REDUCTION_RHF,
                         max_iterations::Integer = DEFAULT_REDUCTION_MAX_ITERATIONS,
                         aggressive::Bool = false,
                         schoenhage_threshold::Integer = DEFAULT_SCHOENHAGE_THRESHOLD,
                         base_cutoff::Integer = DEFAULT_BASE_CUTOFF,
                         want_profile::Bool = true,
                         gc_dimension::Integer = REDUCTION_GC_DIMENSION,
                         gc_interval::Integer = REDUCTION_GC_INTERVAL,
                         low_memory::Union{Nothing, Bool} = nothing,
                         gc_full::Bool = false,
                         trim_memory::Bool = false,
                         _held_above::Int = 0,
                         telemetry::Union{Nothing, ReductionTelemetry} = nothing,
                         _depth::Integer = 0) where {T<:Integer}
    n = _check_reduction_input(B)
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be at least one"))

    resolved_goal = goal === nothing ? goal_from_rhf(n, rhf) : goal
    resolved_goal.n == n || throw(ArgumentError(
        "the goal has dimension $(resolved_goal.n) but the basis has $n columns"))

    if telemetry !== nothing
        telemetry.levels += 1
        telemetry.max_depth = max(telemetry.max_depth, Int(_depth))
    end

    # Base case. Small dimensions go to fplll rather than being recursed on;
    # two columns are handled exactly, with no compression and no precision
    # policy at all.
    if 2 < n <= base_cutoff
        if telemetry !== nothing
            telemetry.base_cases += 1
            telemetry.fplll_calls += 1
        end
        started = _tick()
        reduced, applied = fplll_reduce(B)
        for j in 1:n, i in 1:n
            B[i, j] = reduced[i, j]
            U[i, j] = applied[i, j]
        end
        telemetry === nothing || (telemetry.time_base += _tock(started))
        # The profile costs a full arbitrary-precision factorisation, and only
        # the outermost caller ever reads it -- every recursive call discards
        # the info it is returned in. At dimension 96 the q-ary family makes 507
        # base-case calls, so computing it unconditionally means 507 wasted
        # factorisations.
        return B, U, (iterations = 0, goal_met = true, stopped = :base_case,
                      profile = _reported_profile(B, aggressive, want_profile,
                                                  _depth, telemetry))
    end

    if n <= 2
        if telemetry === nothing
            _reduce_two_columns!(B, U, Int(schoenhage_threshold))
        else
            largest = 0
            for value in B
                iszero(value) || (largest = max(largest, ndigits(value; base = 2)))
            end
            telemetry.base_cases += 1
            if n <= 1 || largest < schoenhage_threshold
                telemetry.lagrange_calls += 1
            else
                telemetry.schoenhage_calls += 1
            end
            started = _tick()
            _reduce_two_columns!(B, U, Int(schoenhage_threshold))
            telemetry.time_base += _tock(started)
        end
        return B, U, (iterations = 0, goal_met = true, stopped = :base_case,
                      profile = [_log2_abs(B[i, i]) for i in 1:n])
    end

    # --- initialise and compress -------------------------------------------
    original = Matrix{T}(B)
    profile = [_log2_abs(B[i, i]) for i in 1:n]
    offsets = zeros(Float64, n)

    bounded = low_memory === nothing ? n >= REDUCTION_LOW_MEMORY_DIMENSION :
                                       low_memory
    stack = _TransformStack{T}(bounded)

    compression_started = _tick()
    working, applied, precision = _compress_integer(B, profile, offsets, n, aggressive)
    telemetry === nothing || (telemetry.time_compress += _tock(compression_started))
    compression = applied

    # --- the reduction loop -------------------------------------------------
    iteration = 0
    goal_met = false
    stopped = :cap

    # The profile at the last cycle boundary, at TRUE scale. Compressed
    # profiles are shifted every iteration, so they cannot be compared across
    # iterations directly -- only profile .+ offsets is meaningful.
    cycle_profile = profile .+ offsets

    while true
        if _should_check_goal(iteration)
            if goal_check(resolved_goal, profile)
                goal_met = true
                stopped = :goal
                break
            end

            # Nothing moved over a whole cycle of windows, so no further cycle
            # will move anything either: the reduction has converged on
            # something short of the goal. flatter has this test in
            # `Proved3::is_reduced` but leaves it commented out.
            current = profile .+ offsets
            if all(i -> abs(current[i] - cycle_profile[i]) <= REDUCTION_STAGNATION_TOLERANCE,
                   1:n)
                stopped = :stagnated
                telemetry === nothing || (telemetry.stagnated += 1)
                break
            end
            cycle_profile = current
        end
        if iteration >= max_iterations
            telemetry === nothing || (telemetry.capped += 1)
            break
        end

        window = _reduction_window(iteration, n)
        width = length(window)

        # Reduce the window recursively. A window of one column has nothing to
        # do, but the iteration still counts so the schedule advances.
        # Measuring what is held walks every entry of four matrices, inspecting
        # each value's allocation -- O(n^2) big-integer inspections per
        # iteration, at every level. That is a real cost, not a free
        # observation, so it is off unless asked for. Set
        # `telemetry.measure_memory = true` when the question is memory.
        held_above = (telemetry !== nothing && telemetry.measure_memory) ?
                     _held_above + held_bytes(working) + held_bytes(B) +
                     held_bytes(original) + held_bytes(stack.factors) : 0

        sub_transform = nothing
        if width >= 2
            sub_basis = Matrix{T}(working[window, window])
            sub_transform = Matrix{T}(undef, width, width)
            recursion_started = _tick()
            lattice_reduce!(sub_basis, sub_transform;
                            goal = subgoal(resolved_goal, first(window) - 1, last(window)),
                            max_iterations = max_iterations,
                            aggressive = aggressive,
                            schoenhage_threshold = schoenhage_threshold,
                            base_cutoff = base_cutoff,
                            want_profile = want_profile,
                            gc_dimension = gc_dimension,
                            gc_interval = gc_interval,
                            low_memory = low_memory,
                            gc_full = gc_full,
                            trim_memory = trim_memory,
                            _held_above = held_above,
                            telemetry = telemetry,
                            _depth = Int(_depth) + 1)
            telemetry === nothing || (telemetry.time_recursion += _tock(recursion_started))
        end

        # Apply it, then re-triangularise and size reduce. The product is not
        # triangular -- the window transform mixes columns whose support
        # extends below their own row -- so a fresh factorisation is needed,
        # which is exactly what the fused routine provides.
        multiply_started = _tick()
        candidate = sub_transform === nothing ? Matrix{T}(working) :
                    _apply_window_right(working, sub_transform, window)
        telemetry === nothing || (telemetry.time_matmul += _tock(multiply_started))

        size_reduction = Matrix{T}(undef, n, n)
        fused_started = _tick()
        _, _, factor, _ = fused_qr_size_reduction!(
            candidate, size_reduction; precision = precision,
            timings = telemetry === nothing ? nothing : telemetry.fused_split)
        if telemetry !== nothing
            telemetry.time_fused += _tock(fused_started)
            telemetry.fused_calls += 1
        end

        multiply_started = _tick()
        window_transform = sub_transform === nothing ? size_reduction :
                           _apply_window_left(sub_transform, size_reduction, window)
        telemetry === nothing || (telemetry.time_matmul += _tock(multiply_started))

        # Lift and fold in immediately: the scaling this transform needs is the
        # one currently in force, and holding it unlifted would only cost memory.
        collect_started = _tick()
        _push_transform!(stack, _lift_transform(window_transform, compression))
        telemetry === nothing || (telemetry.time_collect += _tock(collect_started))

        for i in 1:n
            profile[i] = _log2_abs(factor[i, i])
        end
        # Sample here rather than at the end of the iteration: this is the
        # point where the most is live, with the previous working basis, the
        # candidate, its floating point factor and the size reduction all still
        # in scope alongside the transform stack. `_held_above` carries what the
        # levels above this one hold, so the figure is the whole live path
        # rather than one level's share. `U` is excluded because the caller
        # supplies it with `undef` slots that are not filled until the end.
        if telemetry !== nothing && telemetry.measure_memory
            here = held_bytes(working) + held_bytes(B) + held_bytes(original) +
                   held_bytes(candidate) + held_bytes(size_reduction) +
                   held_bytes(factor) + held_bytes(stack.factors)
            telemetry.peak_data = max(telemetry.peak_data, _held_above + here)
        end

        compression_started = _tick()
        working, applied, precision =
            _compress_factor(factor, profile, offsets, n, T, aggressive)
        telemetry === nothing || (telemetry.time_compress += _tock(compression_started))
        compression = applied

        iteration += 1
        telemetry === nothing || (telemetry.iterations += 1)

        # Everything from this iteration -- the candidate basis, its R factor,
        # the fused factorisation's temporaries -- is now unreachable.
        if n >= gc_dimension && iszero(mod(iteration, max(gc_interval, 1)))
            # Arbitrary-precision values carry finalizers, and a finalizable
            # object survives one collection: the first queues its finalizer,
            # the second releases the limbs. An incremental pass therefore
            # cannot reclaim them at all, so `gc_full` runs two full passes.
            if gc_full
                GC.gc(); GC.gc()
            else
                GC.gc(false)
            end
            # Julia freeing a limb block does not return it to the operating
            # system; the C allocator keeps it. This is what actually reduces
            # resident memory for this workload.
            trim_memory && release_free_memory()
        end

        # `gc_live_bytes` reports the figure from the most recent collection,
        # so sampling it each iteration tracks a high-water mark rather than the
        # instantaneous set. It is ABSOLUTE, not relative to the start of this
        # reduction, so a caller comparing runs must subtract its own baseline --
        # or better, measure each run in a fresh process.
        if telemetry !== nothing && telemetry.measure_memory
            telemetry.peak_live = max(telemetry.peak_live, Base.gc_live_bytes())
        end
    end

    # --- lift, apply, and size reduce once at true scale ---------------------
    # This phase works at FULL scale, on the original uncompressed entries, so
    # it can easily cost more than the whole compressed loop above it.
    finalise_started = _tick()

    # (a) Combine the partial products left on the stack. The lifting itself
    #     already happened, iteration by iteration, inside the loop.
    collect_started = _tick()
    transform = _drain_transform(stack, n)
    telemetry === nothing || (telemetry.time_collect += _tock(collect_started))

    # (b) Apply the composed transform to the ORIGINAL basis, whose entries
    #     were never compressed. Unavoidable, and the widest arithmetic here.
    apply_started = _tick()
    reduced = _reduction_mul(original, transform)
    telemetry === nothing || (telemetry.time_apply += _tock(apply_started))

    # Release the originals before the next product: this phase holds the widest
    # matrices of the run, and `original` and `reduced` are both finished with
    # once the result has been copied out.
    original = Matrix{T}(undef, 0, 0)
    for j in 1:n, i in 1:n
        B[i, j] = reduced[i, j]
    end
    reduced = Matrix{T}(undef, 0, 0)
    n >= gc_dimension && GC.gc(false)

    # The profile was tracked in compressed coordinates throughout; restore it.
    for i in 1:n
        profile[i] += offsets[i]
    end

    final_transform = Matrix{T}(undef, n, n)
    # (c) One size reduction at true scale. The precision comes from the
    #     UNcompressed spread, so this is the widest fused QR in the whole run.
    final_precision = lll_precision(profile_spread(profile), n; aggressive = aggressive)
    final_sr_started = _tick()
    _, _, final_factor, _ = fused_qr_size_reduction!(B, final_transform;
                                                     precision = final_precision)
    if telemetry !== nothing
        telemetry.fused_calls += 1
        telemetry.time_final_sr += _tock(final_sr_started)
    end

    result = _reduction_mul(transform, final_transform)
    for j in 1:n, i in 1:n
        U[i, j] = result[i, j]
    end
    for i in 1:n
        profile[i] = _log2_abs(final_factor[i, i])
    end

    telemetry === nothing || (telemetry.time_finalise += _tock(finalise_started))

    return B, U, (iterations = iteration, goal_met = goal_met,
                  stopped = stopped, profile = profile)
end

"""
    lattice_reduce(B; kwargs...) -> (B_reduced, U, info)

Non-mutating form of [`lattice_reduce!`](@ref). `B` is left untouched.
"""
function lattice_reduce(B::AbstractMatrix{T}; kwargs...) where {T<:Integer}
    reduced = Matrix{T}(B)
    transform = Matrix{T}(undef, size(B, 2), size(B, 2))
    return lattice_reduce!(reduced, transform; kwargs...)
end
