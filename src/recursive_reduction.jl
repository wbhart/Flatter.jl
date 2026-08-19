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

# ---------------------------------------------------------------------------
# Lifting the accumulated transforms
# ---------------------------------------------------------------------------

# flatter's `collect_U`. Each iteration's transform was computed against a
# compressed basis, so before it can be composed with the others it has to be
# conjugated by the scaling that was in force at the time -- the D^-1 U D of
# Saruchi-Morel-Stehle-Villard, which `conjugate_transform` performs.
#
# The pairing matters and is easy to get wrong: `compressions` holds one entry
# more than `transforms`, because the initial compression happens before the
# first iteration. The last compression is discarded (nothing was computed
# against it yet), and thereafter transform k pairs with compression k-1, the
# scaling that produced the basis it was computed against.
function _collect_transform(transforms::Vector{Matrix{T}},
                            compressions::Vector{Vector{Int}},
                            n::Int) where {T<:Integer}
    length(compressions) == length(transforms) + 1 || throw(ErrorException(
        "compression and transform stacks are out of step: " *
        "$(length(compressions)) vs $(length(transforms))"))

    result = Matrix{T}(undef, n, n)
    _set_identity!(result)
    pop!(compressions)

    while !isempty(transforms)
        lifted = conjugate_transform(pop!(transforms), pop!(compressions))
        result = _reduction_mul(T.(lifted), result)
    end
    return result
end

# One place to choose the matrix product used throughout the driver.
_reduction_mul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T} = strassen(A, B)

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
  * `goal_met`   -- whether the goal was satisfied, as opposed to the iteration
                    cap being reached;
  * `profile`    -- `log2 |r_ii|` of the reduced basis, at true scale.

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
                         schoenhage_threshold::Integer = DEFAULT_SCHOENHAGE_THRESHOLD) where {T<:Integer}
    n = _check_reduction_input(B)
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be at least one"))

    resolved_goal = goal === nothing ? goal_from_rhf(n, rhf) : goal
    resolved_goal.n == n || throw(ArgumentError(
        "the goal has dimension $(resolved_goal.n) but the basis has $n columns"))

    # Base case. Two columns are handled exactly, with no compression, no
    # precision policy and no recursion.
    if n <= 2
        _reduce_two_columns!(B, U, Int(schoenhage_threshold))
        return B, U, (iterations = 0, goal_met = true,
                      profile = [_log2_abs(B[i, i]) for i in 1:n])
    end

    # --- initialise and compress -------------------------------------------
    original = Matrix{T}(B)
    profile = [_log2_abs(B[i, i]) for i in 1:n]
    offsets = zeros(Float64, n)

    transforms = Matrix{T}[]
    compressions = Vector{Int}[]

    working, applied, precision = _compress_integer(B, profile, offsets, n, aggressive)
    push!(compressions, applied)

    # --- the reduction loop -------------------------------------------------
    iteration = 0
    goal_met = false

    while true
        if _should_check_goal(iteration) && goal_check(resolved_goal, profile)
            goal_met = true
            break
        end
        iteration >= max_iterations && break

        window = _reduction_window(iteration, n)
        width = length(window)

        # Reduce the window recursively. A window of one column has nothing to
        # do, but the iteration still counts so the schedule advances.
        embedded = Matrix{T}(undef, n, n)
        _set_identity!(embedded)
        if width >= 2
            sub_basis = Matrix{T}(working[window, window])
            sub_transform = Matrix{T}(undef, width, width)
            lattice_reduce!(sub_basis, sub_transform;
                            goal = subgoal(resolved_goal, first(window) - 1, last(window)),
                            max_iterations = max_iterations,
                            aggressive = aggressive,
                            schoenhage_threshold = schoenhage_threshold)
            embedded[window, window] = sub_transform
        end

        # Apply it, then re-triangularise and size reduce. The product is not
        # triangular -- the window transform mixes columns whose support
        # extends below their own row -- so a fresh factorisation is needed,
        # which is exactly what the fused routine provides.
        candidate = _reduction_mul(working, embedded)
        size_reduction = Matrix{T}(undef, n, n)
        _, _, factor, _ = fused_qr_size_reduction!(candidate, size_reduction;
                                                   precision = precision)

        push!(transforms, _reduction_mul(embedded, size_reduction))

        for i in 1:n
            profile[i] = _log2_abs(factor[i, i])
        end
        working, applied, precision =
            _compress_factor(factor, profile, offsets, n, T, aggressive)
        push!(compressions, applied)

        iteration += 1
    end

    # --- lift, apply, and size reduce once at true scale ---------------------
    transform = _collect_transform(transforms, compressions, n)

    reduced = _reduction_mul(original, transform)
    for j in 1:n, i in 1:n
        B[i, j] = reduced[i, j]
    end

    # The profile was tracked in compressed coordinates throughout; restore it.
    for i in 1:n
        profile[i] += offsets[i]
    end

    final_transform = Matrix{T}(undef, n, n)
    final_precision = lll_precision(profile_spread(profile), n; aggressive = aggressive)
    _, _, final_factor, _ = fused_qr_size_reduction!(B, final_transform;
                                                     precision = final_precision)

    result = _reduction_mul(transform, final_transform)
    for j in 1:n, i in 1:n
        U[i, j] = result[i, j]
    end
    for i in 1:n
        profile[i] = _log2_abs(final_factor[i, i])
    end

    return B, U, (iterations = iteration, goal_met = goal_met, profile = profile)
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
