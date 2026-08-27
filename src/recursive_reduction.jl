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

    # Heuristic3-only profiling.  `time_fused` still measures the entire tiled
    # update; these counters divide it into optimisation targets without
    # changing the algorithm.
    h3_calls::Int = 0
    h3_sr_pairs::Int = 0
    h3_sr_reuse_qr::Int = 0
    h3_sr_refactor::Int = 0
    h3_sr_triangular::Int = 0
    h3_time_precheck::Float64 = 0.0
    h3_time_setup::Float64 = 0.0
    h3_time_basis::Float64 = 0.0
    h3_time_basis_materialise::Float64 = 0.0
    h3_time_basis_mul::Float64 = 0.0
    h3_time_basis_write::Float64 = 0.0
    h3_time_qr::Float64 = 0.0
    h3_time_sr_setup::Float64 = 0.0
    h3_time_sr::Float64 = 0.0
    h3_time_sr_materialise::Float64 = 0.0
    h3_time_sr_precision::Float64 = 0.0
    h3_time_sr_factorprep::Float64 = 0.0
    h3_time_sr_reduce::Float64 = 0.0
    h3_time_sr_orthogonal::Float64 = 0.0
    h3_time_sr_triangular::Float64 = 0.0
    h3_time_sr_writeback::Float64 = 0.0
    h3_time_sr_propagate::Float64 = 0.0

    # Heuristic2 phase/schedule counters.  H2 is a fixed left/right/all cycle;
    # the all step hands the full window to Heuristic3.
    h2_calls::Int = 0
    h2_left_steps::Int = 0
    h2_right_steps::Int = 0
    h2_all_steps::Int = 0
    h2_phase3_calls::Int = 0
    h2_relative_calls::Int = 0
    h2_time_qr::Float64 = 0.0
    h2_time_wrapper_qr::Float64 = 0.0
    h2_time_relative::Float64 = 0.0
    h2_time_u2_compose::Float64 = 0.0
    h2_time_top_right::Float64 = 0.0
    h2_time_collect::Float64 = 0.0
    h2_time_compress::Float64 = 0.0
    h2_time_apply::Float64 = 0.0

    time_fused::Float64 = 0.0
    time_matmul::Float64 = 0.0
    time_compress::Float64 = 0.0
    time_base::Float64 = 0.0
    time_recursion::Float64 = 0.0
    time_finalise::Float64 = 0.0
    time_collect::Float64 = 0.0
    time_push::Float64 = 0.0
    time_apply::Float64 = 0.0
    time_final_sr::Float64 = 0.0
    time_dense_qr::Float64 = 0.0
    time_profile::Float64 = 0.0
    time_gc::Float64 = 0.0
    time_setup::Float64 = 0.0
    final_precision::Int = 0
    final_spread::Float64 = 0.0
    widest_transform::Int = 0
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
    if t.h2_calls > 0
        println(io, "  Heuristic2 calls         : ", t.h2_calls)
        println(io, "    phase-2 steps          : left ", t.h2_left_steps,
                    ", right ", t.h2_right_steps, ", all ", t.h2_all_steps)
        println(io, "    phase-3 handoffs       : ", t.h2_phase3_calls)
        println(io, "    relative-B2 wrappers   : ", t.h2_relative_calls)
        println(io, "    representation QR      : ", round(t.h2_time_qr; digits = 3), " s")
        println(io, "    LatRedRelSR QR         : ", round(t.h2_time_wrapper_qr; digits = 3), " s")
        println(io, "    relative size reduction: ", round(t.h2_time_relative; digits = 3), " s")
        println(io, "    U2 composition         : ", round(t.h2_time_u2_compose; digits = 3), " s")
        println(io, "    top-right product      : ", round(t.h2_time_top_right; digits = 3), " s")
        println(io, "    structured collect_U   : ", round(t.h2_time_collect; digits = 3), " s")
        println(io, "    scalar compression     : ", round(t.h2_time_compress; digits = 3), " s")
        println(io, "    final exact apply      : ", round(t.h2_time_apply; digits = 3), " s")
    end
    if t.h3_calls > 0
        println(io, "  Heuristic3 updates       : ", t.h3_calls)
        println(io, "    precondition scan      : ", round(t.h3_time_precheck; digits = 3), " s")
        println(io, "    tile/embed setup       : ", round(t.h3_time_setup; digits = 3), " s")
        println(io, "    tiled basis update     : ", round(t.h3_time_basis; digits = 3), " s")
        println(io, "      materialise views    : ", round(t.h3_time_basis_materialise; digits = 3), " s")
        println(io, "      integer products     : ", round(t.h3_time_basis_mul; digits = 3), " s")
        println(io, "      copy/write blocks    : ", round(t.h3_time_basis_write; digits = 3), " s")
        println(io, "    diagonal tile QR       : ", round(t.h3_time_qr; digits = 3), " s")
        println(io, "    size-red setup         : ", round(t.h3_time_sr_setup; digits = 3), " s")
        println(io, "    tiled size reduction   : ", round(t.h3_time_sr; digits = 3), " s")
        println(io, "      materialise views    : ", round(t.h3_time_sr_materialise; digits = 3), " s")
        println(io, "      precision scans      : ", round(t.h3_time_sr_precision; digits = 3), " s")
        println(io, "      compact-WY prep      : ", round(t.h3_time_sr_factorprep; digits = 3), " s")
        println(io, "      relative reduction   : ", round(t.h3_time_sr_reduce; digits = 3), " s")
        println(io, "        orthogonal/reuse   : ", round(t.h3_time_sr_orthogonal; digits = 3), " s")
        println(io, "        triangular         : ", round(t.h3_time_sr_triangular; digits = 3), " s")
        println(io, "      writeback            : ", round(t.h3_time_sr_writeback; digits = 3), " s")
        println(io, "      propagation          : ", round(t.h3_time_sr_propagate; digits = 3), " s")
        println(io, "    SR tile pairs          : ", t.h3_sr_pairs,
                    " (reuse QR ", t.h3_sr_reuse_qr,
                    ", refactor ", t.h3_sr_refactor,
                    ", triangular ", t.h3_sr_triangular, ")")
    end
    println(io, "  time in matrix products  : ", round(t.time_matmul; digits = 3), " s")
    println(io, "  time in compression      : ", round(t.time_compress; digits = 3), " s")
    println(io, "  time in base cases       : ", round(t.time_base; digits = 3), " s")
    println(io, "  time in recursive calls  : ", round(t.time_recursion; digits = 3),
                " s  (a parent's view of its children, so NOT part of any total)")
    println(io, "  time collecting garbage  : ", round(t.time_gc; digits = 3), " s")
    println(io, "  time in per-level setup  : ", round(t.time_setup; digits = 3), " s")
    if t.final_precision > 0
        println(io, "  widest final size red.   : ", t.final_precision, " bits, from a ",
                    round(t.final_spread; digits = 1), " bit spread")
    end
    if t.widest_transform > 0
        println(io, "  widest accumulated U     : ", t.widest_transform, " bits")
    end
    if t.profile_calls > 0
        println(io, "  time reporting profiles  : ", round(t.time_profile; digits = 3),
                    " s (", t.profile_calls, " factorisations)")
    end
    println(io, "  time finalising          : ", round(t.time_finalise; digits = 3), " s")
    println(io, "    folding in transforms  : ", round(t.time_push; digits = 3), " s")
    println(io, "    combining transforms   : ", round(t.time_collect; digits = 3), " s")
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

# Goal checks happen only at schedule stopping points.  The legacy cycle is a
# three-step middle/left/right schedule.  flatter's Heuristic3 instead asks its
# split tree whether the CURRENT state is a stopping point; phase 3 says yes on
# even iterations, including the initial state.
function _should_check_goal(iteration::Int, schedule::Symbol,
                            split_node::Union{Nothing, SplitPhase3})
    schedule === :split && return split_stopping_point(split_node::SplitPhase3)
    return iteration > 0 && mod(iteration, 3) == 0
end

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

"""
    _ReductionSettings

Everything the recursion passes down unchanged, gathered so that adding a
setting does not mean editing every call site.

There were fifteen of these threaded through by hand; a list that long is one
where a forgotten entry silently changes a sub-problem's behaviour rather than
failing to compile.
"""
struct _ReductionSettings
    max_iterations::Int
    aggressive::Bool
    schoenhage_threshold::Int
    base_cutoff::Int
    blocksize::Int
    panelsize::Int
    tiled::Bool
    want_profile::Bool
    gc_dimension::Int
    gc_interval::Int
    low_memory::Union{Nothing, Bool}
    gc_full::Bool
    trim_memory::Bool
    schedule::Symbol
    validate::Bool
end

"""
Extra bits the validator's reference factorisation may use beyond the driver's
working precision.

Enough to resolve a genuine disagreement -- the ones seen were 4 to 112 bits --
without letting an uncompressed candidate drive the reference to thousands.
"""
const VALIDATION_PRECISION_HEADROOM = 512

"""
How much a single update may widen the profile spread before it counts as having
made the basis worse rather than better.

An iteration can legitimately move the profile around while working towards a
flatter one, so this is deliberately loose. It exists to catch an update that is
diverging, not one that is merely not converging yet.
"""
const PROFILE_GROWTH_ALLOWANCE = 64.0

"""
    _validate_update(working, candidate, transform, factor, iteration, depth)

Check the invariants a representation update must maintain, at the iteration
that breaks them.

Every failure seen from the tiled update so far has surfaced downstream -- a
zero diagonal in a later factorisation, a non-finite profile entry in the
compression -- several steps after the step that caused it. These are the
properties that must hold on EVERY iteration, so a break is reported where it
happens:

  * `candidate == working * transform`, exactly. Integer arithmetic, so there is
    no tolerance to argue about.
  * `transform` unimodular, checked by an exact determinant.
  * the reported profile finite and nonzero, since the working precision for the
    next iteration is derived from it.

Off by default, and expensive when on: an exact determinant and a full
high-precision factorisation on every iteration at every level. The last run
with it enabled was killed, so keep it to the small dimensions used for
debugging rather than turning it on for a benchmark.
"""
function _validate_update(working::AbstractMatrix{T}, candidate::AbstractMatrix{T},
                          transform::AbstractMatrix{T}, factor::AbstractMatrix{S},
                          iteration::Int, depth::Int,
                          precision::Integer,
                          windows::Vector{UnitRange{Int}}) where {T<:Integer, S<:AbstractFloat}
    n = size(working, 2)
    where_ = "iteration $iteration at depth $depth"

    expected = _reduction_mul(Matrix{T}(working), Matrix{T}(transform))
    expected == candidate || throw(ErrorException(
        "$where_: the update does not satisfy candidate == working * transform"))

    _exact_determinant(transform) in (one(T), -one(T)) || throw(ErrorException(
        "$where_: the update's transform is not unimodular"))

    for i in 1:n
        value = factor[i, i]
        iszero(value) && throw(ErrorException(
            "$where_: R has a zero diagonal entry at index $i"))
        isfinite(_log2_abs(value)) || throw(ErrorException(
            "$where_: R has a non-finite diagonal entry at index $i"))
    end

    # And R must actually DESCRIBE the candidate. The diagonal being finite is
    # not enough: the next iteration's working precision is derived from it, so
    # an R that is well formed but belongs to a different matrix sends a wrong
    # number downstream and the failure appears somewhere else entirely.
    #
    # The reference needs the precision the CANDIDATE demands, not the driver's.
    # Factorising at too few bits cannot resolve the cancellation that makes a
    # small diagonal small, so it OVERSTATES it -- and the tiled update already
    # factorises each tile at whatever that tile requires, which can be far more
    # than the driver's figure. Comparing the two then reports a disagreement
    # caused entirely by the reference being the less accurate of the pair.
    widest = 0
    for value in candidate
        iszero(value) || (widest = max(widest, ndigits(value; base = 2)))
    end
    # Bounded: the candidate is uncompressed, so `widest` can be many thousands
    # of bits and a factorisation at that precision, run every iteration at every
    # level, took the test suite from under two minutes to over five. Well past
    # the driver's own precision is enough to tell a real disagreement from one
    # caused by an under-precise reference, which is all this has to do.
    bits = clamp(householder_precision(n, Float64(widest)),
                 Int(precision), Int(precision) + VALIDATION_PRECISION_HEADROOM)
    truth = with_precision(bits) do
        factors, _ = householder_block(bigfloat_matrix(candidate, bits))
        [_log2_abs(factors[i, i]) for i in 1:n]
    end
    for i in 1:n
        reported = _log2_abs(factor[i, i])
        abs(reported - truth[i]) < 1.0 || throw(ErrorException(
            "$where_: R's diagonal disagrees with a true factorisation at " *
            "index $i: reported $(round(reported; digits = 2)) bits against " *
            "$(round(truth[i]; digits = 2))"))
    end

    # And the update has to REDUCE. Every check above is satisfied by a
    # transform that makes the basis catastrophically worse: the tiled update
    # took a knapsack basis from an 8.6 bit profile spread to 61,326 while
    # keeping `candidate == working * transform` exactly, unimodular, with an R
    # matching a true factorisation. The precision demanded downstream then rose
    # to 122,750 bits, and every symptom looked like a precision problem.
    entering = [_log2_abs(working[i, i]) for i in 1:n]

    # A block diagonal transform mixes only a window's own columns, so every
    # leading sublattice OUTSIDE the windows keeps its determinant, and size
    # reduction cannot move any of them either. The profile outside the windows
    # is therefore an invariant of the whole update -- if it moves, the transform
    # is reaching columns it has no business touching.
    inside = falses(n)
    for w in windows, i in w
        inside[i] = true
    end
    for i in 1:n
        inside[i] && continue
        abs(truth[i] - entering[i]) < 1.0 || throw(ErrorException(
            "$where_: the profile moved OUTSIDE the reduced windows, at index " *
            "$i: $(round(entering[i]; digits = 2)) bits became " *
            "$(round(truth[i]; digits = 2)). A block diagonal transform cannot " *
            "do that, so the transform is not block diagonal over these tiles"))
    end

    # And within each window the sub-reduction is supposed to FLATTEN the
    # profile. Reporting per window separates "the sub-transform is bad" from
    # "the update mishandles a good one", which the overall spread cannot.
    for w in windows
        length(w) >= 2 || continue
        window_before = profile_spread(entering[w])
        window_after = profile_spread(truth[w])
        window_after <= window_before + PROFILE_GROWTH_ALLOWANCE || throw(ErrorException(
            "$where_: window $(first(w)):$(last(w)) came back WORSE, spread " *
            "$(round(window_before; digits = 1)) to " *
            "$(round(window_after; digits = 1)); the sub-reduction returned a " *
            "transform that flattens nothing"))
    end

    before = profile_spread(entering)
    after = profile_spread(truth)
    after <= before + PROFILE_GROWTH_ALLOWANCE || throw(ErrorException(
        "$where_: the update GREW the profile spread from " *
        "$(round(before; digits = 1)) bits to $(round(after; digits = 1)); " *
        "the transform is valid but the basis is worse. Windows: " *
        "$(join([string(first(w), ":", last(w)) for w in windows], ", "))"))
    return nothing
end

"Exact determinant by fraction-free elimination."
function _exact_determinant(A::AbstractMatrix{T}) where {T<:Integer}
    n = size(A, 1)
    M = Matrix{T}(A)
    sign = one(T)
    previous = one(T)
    for k in 1:(n - 1)
        if iszero(M[k, k])
            pivot = findfirst(r -> !iszero(M[r, k]), (k + 1):n)
            pivot === nothing && return zero(T)
            row = k + pivot
            for c in 1:n
                M[k, c], M[row, c] = M[row, c], M[k, c]
            end
            sign = -sign
        end
        for i in (k + 1):n, j in (k + 1):n
            M[i, j] = div(M[i, j] * M[k, k] - M[i, k] * M[k, j], previous)
        end
        previous = M[k, k]
    end
    return sign * M[n, n]
end

"""
    _window_transforms(T, windows, transforms, n) -> (embedded, tiles, reducible)

The block-diagonal transform that reduces every window at once.

`transforms` may hold `nothing` for a window too narrow to have been reduced;
those blocks become the identity. Returns the tile decomposition alongside, since
every caller that wants the transform also wants the tiles it was built from.
"""
function _window_transforms(::Type{T}, windows::Vector{UnitRange{Int}},
                            transforms::Vector, n::Int) where {T<:Integer}
    tiles, reducible = tile_partition(windows, n)
    blocks = Matrix{T}[]
    for (slot, w) in enumerate(windows)
        block = transforms[slot]
        if block === nothing
            block = Matrix{T}(undef, length(w), length(w))
            _set_identity!(block)
        end
        push!(blocks, Matrix{T}(block))
    end
    embedded = Matrix{T}(undef, n, n)
    embed_transforms!(embedded, tiles, reducible, blocks)
    return embedded, tiles, reducible
end

"""
    _apply_windows(working, windows, transforms, n)

`working` times the block-diagonal transform of all the windows.

With a single window this is the existing block-aware product, which never
materialises the embedded matrix. With several it goes through the tiled update,
which exploits the same structure a tile at a time.
"""
function _apply_windows(working::AbstractMatrix{T}, windows::Vector{UnitRange{Int}},
                        transforms::Vector, n::Int) where {T<:Integer}
    if length(windows) == 1
        transforms[1] === nothing && return Matrix{T}(working)
        return _apply_window_right(working, transforms[1], windows[1])
    end
    embedded, tiles, _ = _window_transforms(T, windows, transforms, n)
    product = zeros(T, n, n)
    tiled_basis_update!(product, working, embedded, tiles)
    return product
end

"""
    _compose_windows(windows, transforms, size_reduction, n)

The window transforms followed by the size reduction, as one matrix.
"""
function _compose_windows(windows::Vector{UnitRange{Int}}, transforms::Vector,
                          size_reduction::AbstractMatrix{T}, n::Int) where {T<:Integer}
    if length(windows) == 1
        transforms[1] === nothing && return size_reduction
        return _apply_window_left(transforms[1], size_reduction, windows[1])
    end
    embedded, _, _ = _window_transforms(T, windows, transforms, n)
    return _reduction_mul(embedded, Matrix{T}(size_reduction))
end

"""
    _reduce_window(working, window, goal, current_drop, held_above, settings, telemetry, depth)

Reduce one window recursively, returning its transform, or `nothing` when the
window is too narrow to have one.

Extracted from the reduction loop unchanged. The loop currently calls it once
per iteration; flatter's phase 3 schedule wants it called once per window, of
which there are two on odd iterations.
"""
function _reduce_window(working::AbstractMatrix{T}, window::UnitRange{Int},
                        goal, current_drop::Real, held_above::Int,
                        settings::_ReductionSettings,
                        telemetry::Union{Nothing, ReductionTelemetry},
                        depth::Int, child = nothing) where {T<:Integer}
    width = length(window)
    if width < 2
        child_profile = [_log2_abs(working[i, i]) for i in window]
        return (transform = nothing, profile = child_profile)
    end

    # Extracting the sub-basis copies a window of the working matrix and
    # allocating the sub-transform fills n^2 slots. Individually small, but paid
    # at every level of a recursion that reaches thousands of levels, so worth
    # seeing rather than inferring.
    setup_started = _tick()
    sub_basis = Matrix{T}(working[window, window])
    sub_transform = Matrix{T}(undef, width, width)
    telemetry === nothing || (telemetry.time_setup += _tock(setup_started))

    # flatter's Heuristic3 deliberately throttles child reductions.  The
    # parent's inherited subgoal can be much stricter than useful early in a
    # reduction, so compare it with a second goal derived from one sixth of the
    # current whole-profile drop and use the looser one.  `get_slope()` in
    # flatter returns the raw quality scalar for heuristic goals; `goal_slope`
    # preserves that convention.
    inherited_goal = subgoal(goal, first(window) - 1, last(window))
    drop_goal = goal_from_drop(width, Float64(current_drop) / 6)
    child_goal = goal_slope(inherited_goal) < goal_slope(drop_goal) ?
                 drop_goal : inherited_goal

    recursion_started = _tick()
    _, _, child_info = lattice_reduce!(sub_basis, sub_transform;
                    goal = child_goal,
                    max_iterations = settings.max_iterations,
                    aggressive = settings.aggressive,
                    schoenhage_threshold = settings.schoenhage_threshold,
                    base_cutoff = settings.base_cutoff,
                    blocksize = settings.blocksize,
                    panelsize = settings.panelsize,
                    tiled = settings.tiled,
                    want_profile = settings.want_profile,
                    gc_dimension = settings.gc_dimension,
                    gc_interval = settings.gc_interval,
                    low_memory = settings.low_memory,
                    gc_full = settings.gc_full,
                    trim_memory = settings.trim_memory,
                    schedule = settings.schedule,
                    validate = settings.validate,
                    _held_above = held_above,
                    # The sub-problem gets ITS node of the schedule tree, not a
                    # fresh one: the tree carries iteration state, and rebuilding
                    # it per call would restart every sub-schedule from zero.
                    _split = child,
                    telemetry = telemetry,
                    _depth = depth + 1)
    telemetry === nothing || (telemetry.time_recursion += _tock(recursion_started))

    # Heuristic3 uses the profiles returned by the child reductions to choose
    # the precision for THIS iteration's tiled QR/size-reduction stage.  The
    # recursive driver normally suppresses profiles from fplll base cases, so
    # recover that small profile here when necessary.
    child_profile = child_info.profile
    if settings.tiled && isempty(child_profile)
        child_profile = _basis_profile(sub_basis, settings.aggressive)
    end
    return (transform = sub_transform, profile = child_profile)
end

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
  * `blocksize`      -- predecessors size-reduced before the deferred `B` and
                        `U` updates are applied as one batch. Exact: the result
                        is identical whatever the value. 1 is the elementwise
                        sweep.
  * `panelsize`      -- columns processed before a panel's reflectors are pushed
                        through the rest of the matrix as one compact-WY block.
                        0 applies each reflector as it is generated.
  * `validate`       -- check the update's invariants every iteration, so a
                        break is reported where it happens rather than as a
                        downstream symptom. Cubic per iteration; for debugging.
  * `schedule`       -- `:legacy` for the hand-rolled middle/left/right cycle,
                        `:split` for flatter's phase 3 tree, which reduces TWO
                        windows on odd iterations rather than one.
  * `tiled`          -- use flatter's `Heuristic3` representation update, which
                        factorises each tile separately instead of the whole
                        matrix once. Produces a DIFFERENT basis for the same
                        lattice, since the R factor is assembled locally.
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
                         blocksize::Integer = DEFAULT_FUSED_BLOCKSIZE,
                         panelsize::Integer = DEFAULT_FUSED_PANELSIZE,
                         tiled::Bool = false,
                         schedule::Symbol = :legacy,
                         validate::Bool = false,
                         want_profile::Bool = true,
                         gc_dimension::Integer = REDUCTION_GC_DIMENSION,
                         gc_interval::Integer = REDUCTION_GC_INTERVAL,
                         low_memory::Union{Nothing, Bool} = nothing,
                         gc_full::Bool = false,
                         trim_memory::Bool = false,
                         _held_above::Int = 0,
                         _split = nothing,
                         telemetry::Union{Nothing, ReductionTelemetry} = nothing,
                         _depth::Integer = 0) where {T<:Integer}
    n = _check_reduction_input(B)
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    # Validated here rather than beside the code that reads it: a dimension at or
    # below `base_cutoff` returns from the base case long before the schedule is
    # consulted, and an argument that is wrong should be rejected whichever path
    # the call happens to take.
    schedule in (:legacy, :split) || throw(ArgumentError(
        "schedule must be :legacy or :split, got $schedule"))
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
    # A genuine profile on the way in: `_check_reduction_input` has already
    # required `B` upper triangular with a nonzero diagonal, so its diagonal IS
    # the R factor's. That is not true of the basis on the way out. Copied
    # because `profile` is mutated throughout.
    entry_profile = validate ? copy(profile) : Float64[]
    offsets = zeros(Float64, n)

    # Gathered once and handed down unchanged, rather than listed at each
    # recursive call site.
    split_node = schedule === :split ?
                 (_split === nothing ? SplitPhase3(n) : _split) : nothing

    settings = _ReductionSettings(Int(max_iterations), aggressive,
                                  Int(schoenhage_threshold), Int(base_cutoff),
                                  Int(blocksize), Int(panelsize), tiled,
                                  want_profile, Int(gc_dimension),
                                  Int(gc_interval), low_memory, gc_full,
                                  trim_memory, schedule, validate)

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
        if _should_check_goal(iteration, schedule, split_node)
            if goal_check(resolved_goal, profile)
                goal_met = true
                stopped = :goal
                break
            end

            # Nothing moved over a whole cycle of windows, so no further cycle
            # will move anything either.  A split schedule is a stopping point
            # at iteration zero, before anything has had a chance to move, so
            # the stagnation test starts only after at least one iteration.
            if iteration > 0
                current = profile .+ offsets
                if all(i -> abs(current[i] - cycle_profile[i]) <= REDUCTION_STAGNATION_TOLERANCE,
                       1:n)
                    stopped = :stagnated
                    telemetry === nothing || (telemetry.stagnated += 1)
                    break
                end
                cycle_profile = current
            end
        end
        if iteration >= max_iterations
            telemetry === nothing || (telemetry.capped += 1)
            break
        end

        # flatter's phase 3 schedule yields TWO windows on odd iterations, the
        # left and right halves, and one -- the middle -- on even ones. The
        # hand-rolled cycle it replaces only ever produced a single window, so
        # the tiled update never saw more than one reduced tile.
        windows = schedule === :split ? split_windows(split_node) :
                                        [_reduction_window(iteration, n)]
        window = first(windows)
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

        children = schedule === :split ? split_children(split_node) :
                                        [nothing for _ in windows]
        # Heuristic3 uses the drop of the CURRENT WHOLE profile to throttle
        # every child created in this iteration.  In particular, both disjoint
        # windows of a split iteration see the same value.
        current_drop = profile_drop(profile)
        sub_results = [_reduce_window(working, w, resolved_goal, current_drop,
                                      held_above, settings, telemetry,
                                      Int(_depth), children[slot])
                       for (slot, w) in enumerate(windows)]
        sub_transforms = Any[result.transform for result in sub_results]
        sub_transform = sub_transforms[1]

        # flatter's Heuristic3 does not choose this iteration's precision from
        # the old profile.  It starts with the old profile, substitutes the
        # profiles returned by all child reductions, and uses that profile_next
        # for the tiled QR/size-reduction precision.
        tiled_precision = precision
        if tiled
            profile_next = copy(profile)
            for (slot, w) in enumerate(windows)
                child_profile = sub_results[slot].profile
                length(child_profile) == length(w) || throw(ErrorException(
                    "child profile has length $(length(child_profile)) for window $w"))
                for (k, i) in enumerate(w)
                    profile_next[i] = child_profile[k]
                end
            end
            tiled_precision = lll_precision(profile_spread(profile_next), n;
                                            aggressive = aggressive)
        end

        schedule === :split && split_advance!(split_node)

        # Apply it, then re-triangularise and size reduce. The product is not
        # triangular -- the window transform mixes columns whose support
        # extends below their own row -- so a fresh factorisation is needed,
        # which is exactly what the fused routine provides.
        multiply_started = _tick()
        if tiled
            # Heuristic3: no factorisation of the whole matrix. The columns are
            # cut at the window boundaries, each reduced tile is factorised on
            # its own, and the off-diagonal blocks of R come from relative size
            # reduction between tiles.
            telemetry === nothing || (telemetry.time_matmul += _tock(multiply_started))
            fused_started = _tick()
            # A window too small to recurse on contributes no transform, so the
            # tile update folds in the identity and does size reduction alone.
            blocks = Matrix{T}[]
            for (slot, w) in enumerate(windows)
                block = sub_transforms[slot]
                if block === nothing
                    block = Matrix{T}(undef, length(w), length(w))
                    _set_identity!(block)
                end
                push!(blocks, Matrix{T}(block))
            end
            candidate, window_transform, factor, _ = heuristic3_update!(
                working, windows, blocks, n, tiled_precision; telemetry = telemetry)
            if telemetry !== nothing
                telemetry.time_fused += _tock(fused_started)
                telemetry.fused_calls += 1
            end
        else
            candidate = _apply_windows(working, windows, sub_transforms, n)
            telemetry === nothing || (telemetry.time_matmul += _tock(multiply_started))

            size_reduction = Matrix{T}(undef, n, n)
            fused_started = _tick()
            _, _, factor, _ = fused_qr_size_reduction!(
                candidate, size_reduction; precision = precision,
                blocksize = blocksize, panelsize = panelsize,
                timings = telemetry === nothing ? nothing : telemetry.fused_split)
            if telemetry !== nothing
                telemetry.time_fused += _tock(fused_started)
                telemetry.fused_calls += 1
            end

            multiply_started = _tick()
            window_transform = _compose_windows(windows, sub_transforms,
                                                size_reduction, n)
            telemetry === nothing || (telemetry.time_matmul += _tock(multiply_started))
        end

        # Check the update where it happened, not where its consequences show
        # up. Every tiled-path failure so far has surfaced downstream: a zero
        # diagonal in a later factorisation, a non-finite profile in the
        # compression, several steps after whatever caused them.
        validation_precision = tiled ? tiled_precision : precision
        validate && _validate_update(working, candidate, window_transform, factor,
                                     iteration, Int(_depth), validation_precision, windows)

        # Lift and fold in immediately: the scaling this transform needs is the
        # one currently in force, and holding it unlifted would only cost memory.
        # Lifting and folding happen in the LOOP, while draining happens in the
        # finalise block. They had shared a counter, which meant the in-loop
        # half -- real integer matrix products -- was timed into a figure that
        # nothing summed, and surfaced only as unaccounted time.
        push_started = _tick()
        _push_transform!(stack, _lift_transform(window_transform, compression))
        telemetry === nothing || (telemetry.time_push += _tock(push_started))

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
        # Collection is not free and fires often at high dimension -- every
        # `gc_interval` iterations at every level -- so it is timed rather than
        # left to surface as unaccounted.
        gc_started = _tick()
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
        telemetry === nothing || (telemetry.time_gc += _tock(gc_started))

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
    # Not added to `time_gc`: this collection sits inside the finalise block and
    # is already counted there, and counting it twice pushed the accounted total
    # past 100%.
    n >= gc_dimension && GC.gc(false)

    # The profile was tracked in compressed coordinates throughout; restore it.
    for i in 1:n
        profile[i] += offsets[i]
    end

    final_transform = Matrix{T}(undef, n, n)
    # (c) One size reduction at true scale, so this is the widest fused QR in the
    #     whole run. The precision has to come from the MATRIX, not from the
    #     profile: the profile records diagonal magnitudes, and a basis whose
    #     off-diagonal entries are large relative to its diagonal needs more bits
    #     than any profile-derived figure will give. The monolithic path never
    #     shows this because its own iterations leave `B` well size reduced; the
    #     tiled update reduces across tiles only, and the difference surfaces
    #     here as "column N did not reduce in 64 passes".
    #     The figure must stay a RATIO. `householder_precision` of the widest
    #     entry is proportional to its absolute width, which on an uncompressed
    #     basis runs to tens of thousands of bits and turns the failure into a
    #     run that never finishes. What the size reduction actually needs is the
    #     span between the largest entry it must remove and the smallest
    #     diagonal it removes against -- bounded, and larger than the profile's
    #     own spread exactly when the off-diagonal entries are the problem.
    widest_entry = 0
    for value in B
        iszero(value) || (widest_entry = max(widest_entry, ndigits(value; base = 2)))
    end
    entry_spread = Float64(widest_entry) - minimum(profile)
    final_precision = lll_precision(max(profile_spread(profile), entry_spread), n;
                                    aggressive = aggressive)
    if telemetry !== nothing
        telemetry.final_precision = max(telemetry.final_precision, final_precision)
        telemetry.final_spread = max(telemetry.final_spread,
                                     max(profile_spread(profile), entry_spread))
    end
    final_sr_started = _tick()
    _, _, final_factor, _ = fused_qr_size_reduction!(B, final_transform;
                                                     precision = final_precision)
    if telemetry !== nothing
        telemetry.fused_calls += 1
        telemetry.time_final_sr += _tock(final_sr_started)
    end

    # Composing the accumulated transform with the final size reduction: another
    # full-width integer product, and the last part of `finalise` that had no
    # counter of its own.
    compose_started = _tick()
    result = _reduction_mul(transform, final_transform)
    for j in 1:n, i in 1:n
        U[i, j] = result[i, j]
    end

    # Measured HERE, after `U` is written. `U` arrives as
    # `Matrix{BigInt}(undef, n, n)`, whose entries are undefined references
    # until assigned, so iterating it any earlier throws `UndefRefError` --
    # which is exactly what an earlier version of this did.
    if telemetry !== nothing
        for value in result
            iszero(value) || (telemetry.widest_transform = max(
                telemetry.widest_transform, ndigits(value; base = 2)))
        end
    end
    telemetry === nothing || (telemetry.time_apply += _tock(compose_started))
    for i in 1:n
        profile[i] = _log2_abs(final_factor[i, i])
    end

    telemetry === nothing || (telemetry.time_finalise += _tock(finalise_started))

    # Validate what is RETURNED, not just each iteration. A reduction whose every
    # update checks out can still hand back a worse basis than it was given: the
    # loop is not the whole function, and `finalise` runs after it. This is the
    # one stage never isolated, and a sub-problem whose iterations all passed was
    # still returning a transform that made its window worse.
    if validate
        # `profile` was just set from `final_factor`'s diagonal, which is the R
        # factor of the returned basis. NOT `B`'s own diagonal: `lattice_reduce!`
        # returns a GENERAL basis, so its diagonal is not a profile at all and a
        # zero there reads as a spread of Inf. A helper in the test file carries
        # this warning; I wrote it and then walked into it here.
        opening = profile_spread(entry_profile)
        closing = profile_spread(profile)
        if closing > opening + PROFILE_GROWTH_ALLOWANCE
            # The accumulated transform is the thing finalise builds its result
            # from, so report how wide it got. Each iteration's transform is
            # valid in COMPRESSED coordinates, but the stack stores it lifted:
            # `conjugate_transform` multiplies entry (i,j) by 2^(shift[j]-shift[i]),
            # which is exact but amplifying. That is safe only while the size
            # reduction bounds each coefficient against the matching diagonal
            # ratio. If the tiled transform is orders of magnitude wider than the
            # monolithic one here, the block-wise reduction is not bounding them
            # tightly enough for the lift.
            # `telemetry.widest_transform` already holds this, measured from
            # `result` after `U` was written. Reading `U` here would work, but
            # only by accident of position: it is `Matrix{BigInt}(undef, ...)`
            # and iterating it before it is filled throws.
            widest_u = telemetry === nothing ? 0 : telemetry.widest_transform
            widest_b = 0
            for value in B
                iszero(value) ||
                    (widest_b = max(widest_b, ndigits(value; base = 2)))
            end
            throw(ErrorException(
                "depth $(Int(_depth)): the reduction RETURNED a worse basis than " *
                "it was given, spread $(round(opening; digits = 1)) to " *
                "$(round(closing; digits = 1)), after $iteration iterations that " *
                "each validated. Accumulated transform is $widest_u bits wide, " *
                "returned basis $widest_b bits. The loss is in finalise"))
        end
    end

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
