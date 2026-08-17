# goal.jl
#
# The reduction quality target, and the test for whether a profile has met it.
#
# A port of flatter's `LatticeReductionGoal` (problems/lattice_reduction/goal.h
# and goal.cpp). This is the termination criterion for the recursive driver:
# each level of the recursion carries a goal, hands scaled-down subgoals to its
# children, and stops when `goal_check` accepts the current profile.
#
# Two families of goal exist, selected by the `proved` flag:
#
#   * heuristic (the default, and the one flatter actually uses): the target is
#     expressed as a `quality` scalar, and the acceptance test looks at three
#     properties of the profile rather than just its total drop.
#   * proved: the target is a slope that interpolates between a base slope and a
#     top slope as a power of the dimension ratio, and acceptance is the single
#     condition `drop < alpha(n) * n`.
#
# See Notes.md for the one flatter quirk carried over verbatim.

# Empirical slopes from flatter's goal.h.
const BKZ_BEST_SLOPE = 0.031281
const HERMITE_BEST_SLOPE = 0.41503749927884365
const DEFAULT_G = 3.0

# ---------------------------------------------------------------------------
# Profile operations
# ---------------------------------------------------------------------------

"""
    profile_drop(prof) -> Float64

The reducible extent of a profile: its spread, less the total size of any gaps
where the profile jumps cleanly upward.

A gap sits after index `i` when every entry from `i+1` onward is strictly above
every entry up to `i`; its size is that separation. Such a gap is exactly what
the block compression in [`compression_shifts`](@ref) removes, so it does not
represent un-reducedness, and the quality test discounts it. What remains is the
extent a reducer could still work on, which is why this rather than the raw
spread is what [`goal_check`](@ref) compares against the drop budget.

Distinct from [`profile_spread`](@ref), which is `maximum - minimum`. The two
coincide on a monotonically decreasing profile, which a reduced basis gives —
there is no upward jump to discount — and diverge otherwise. A strictly
increasing profile has drop zero and spread equal to its full range. In general
`0 <= profile_drop(prof) <= profile_spread(prof)`.
"""
function profile_drop(prof::AbstractVector{<:Real})
    n = length(prof)
    n <= 1 && return 0.0

    max_from_left = Vector{Float64}(undef, n)
    min_from_right = Vector{Float64}(undef, n)

    max_from_left[1] = Float64(prof[1])
    for i in 2:n
        max_from_left[i] = max(Float64(prof[i]), max_from_left[i - 1])
    end

    min_from_right[n] = Float64(prof[n])
    for i in (n - 1):-1:1
        min_from_right[i] = min(Float64(prof[i]), min_from_right[i + 1])
    end

    # max_from_left[n] and min_from_right[1] are the profile's maximum and
    # minimum, so this starts from the spread.
    result = max_from_left[n] - min_from_right[1]

    for i in 1:(n - 1)
        gap = min_from_right[i + 1] - max_from_left[i]
        gap > 0 && (result -= gap)
    end

    return result
end

# ---------------------------------------------------------------------------
# The goal
# ---------------------------------------------------------------------------

"""
    ReductionGoal

A reduction quality target. Immutable; the mutating operation in flatter,
`set_best_slope`, is [`with_best_slope`](@ref) here and returns a new goal.

Construct with [`goal_from_slope`](@ref), [`goal_from_rhf`](@ref) or
[`goal_from_drop`](@ref) rather than directly.
"""
struct ReductionGoal
    n::Int
    top_N::Int
    quality::Float64
    best_slope::Float64
    top_slope::Float64
    g::Float64
    log_g::Float64
    proved::Bool
end

ReductionGoal() = ReductionGoal(0, 0, 0.0, BKZ_BEST_SLOPE, 0.0, DEFAULT_G,
                                log2(DEFAULT_G), false)

"""
    heuristic_goal(n, quality) -> ReductionGoal

A heuristic goal of the given quality scalar. `quality` is not itself a slope;
it scales the dimension-dependent budget [`_goal_shape`](@ref). Use
[`goal_from_slope`](@ref) to specify a slope directly.
"""
heuristic_goal(n::Integer, quality::Real) =
    ReductionGoal(Int(n), Int(n), Float64(quality), BKZ_BEST_SLOPE, 0.0,
                  DEFAULT_G, log2(DEFAULT_G), false)

"""
    proved_goal(n, top_slope; base_slope=BKZ_BEST_SLOPE, g=DEFAULT_G, top_N=0) -> ReductionGoal

A proved goal, whose slope at dimension `k` interpolates from `base_slope` to
`top_slope` as `(k / top_N)^log2(g)`. `top_N = 0` means `top_N = n`.
"""
function proved_goal(n::Integer, top_slope::Real;
                     base_slope::Real = BKZ_BEST_SLOPE,
                     g::Real = DEFAULT_G,
                     top_N::Integer = 0)
    top_slope > base_slope || throw(ArgumentError(
        "top_slope ($top_slope) must exceed base_slope ($base_slope)"))
    resolved_N = iszero(top_N) ? Int(n) : Int(top_N)
    return ReductionGoal(Int(n), resolved_N, 0.0, Float64(base_slope),
                         Float64(top_slope), Float64(g), log2(Float64(g)), true)
end

"""
    _goal_shape(k) -> Float64

The dimension-dependent drop budget shape, `3 * (1 + 3^(log2 k + 1) - 2^(log2 k + 2)) / 2`.

Equivalently `3 * (1 + 3 * k^log2(3) - 4k) / 2`, which is how it is evaluated
here. It appears in flatter both as the scale factor on `quality` and, under the
name `s_guess`, in the slope conversions. Note `_goal_shape(1) == 0`, which is
why the conversions guard against a one-dimensional goal.
"""
function _goal_shape(k::Real)
    k <= 0 && return 0.0
    return 3 * (1 + 3 * Float64(k)^log2(3.0) - 4 * Float64(k)) / 2
end

"""
    goal_alpha(goal, k) -> Float64

The proved slope target at dimension `k`: `best_slope + (k / top_N)^log2(g) * (top_slope - best_slope)`.

Equals `best_slope` at `k = 0` and `top_slope` at `k = top_N`.
"""
goal_alpha(goal::ReductionGoal, k::Integer) =
    goal.best_slope +
    (Float64(k) / Float64(goal.top_N))^goal.log_g * (goal.top_slope - goal.best_slope)

"""
    goal_max_drop(goal) -> Float64

The largest total profile drop this goal will accept.
"""
function goal_max_drop(goal::ReductionGoal)
    goal.proved && return goal_alpha(goal, goal.n) * goal.n
    return goal.quality * _goal_shape(goal.n) + goal.best_slope * goal.n
end

"""
    goal_slope(goal) -> Float64

For a proved goal, the slope target at its own dimension. For a heuristic goal,
flatter returns the raw `quality` scalar, which is not a slope; that behaviour is
preserved here. For the slope a heuristic goal actually targets, use
`goal_max_drop(goal) / goal.n`.
"""
goal_slope(goal::ReductionGoal) =
    goal.proved ? goal_alpha(goal, goal.n) : goal.quality

goal_quality(goal::ReductionGoal) = goal.quality

"""
    goal_rhf(goal) -> Float64

The root Hermite factor implied by this goal, `2^(max_drop / n / 2)`.
"""
goal_rhf(goal::ReductionGoal) = 2.0^((goal_max_drop(goal) / goal.n) / 2)

"""
    subgoal(goal, start, stop) -> ReductionGoal

The goal to hand to a child reducing the profile range `start:stop`.

A proved subgoal keeps the parent's interpolation parameters, including `top_N`,
so its slope target is the parent's curve evaluated at the smaller dimension. A
heuristic subgoal keeps the parent's `quality` and `best_slope` and only changes
the dimension.
"""
function subgoal(goal::ReductionGoal, start::Integer, stop::Integer)
    start < stop || throw(ArgumentError("empty subgoal range $start:$stop"))
    stop <= goal.n || throw(ArgumentError(
        "subgoal range $start:$stop exceeds the goal dimension $(goal.n)"))
    width = Int(stop) - Int(start)

    goal.proved && return ReductionGoal(width, goal.top_N, 0.0, goal.best_slope,
                                        goal.top_slope, goal.g, goal.log_g, true)
    return ReductionGoal(width, width, goal.quality, goal.best_slope, 0.0,
                         goal.g, goal.log_g, false)
end

"""
    with_best_slope(goal, slope) -> ReductionGoal

A heuristic goal that targets the same total drop as `goal`, but attributes it to
a different assumed base slope.

The budget is **invariant** under this operation. A heuristic goal's target slope
decomposes as `max_drop / n = best_slope + gap`, where `gap = quality * shape / n`
is the excess the reduction is expected to achieve over the base slope. Changing
the base slope moves the same total between the two terms: `quality` absorbs the
difference, and `goal_max_drop` is unchanged. Use this when a better estimate of
the achievable base slope arrives and the target should not move.

Errors if `slope` is at or above the goal's current target slope, which would
leave a non-positive excess.
"""
function with_best_slope(goal::ReductionGoal, slope::Real)
    goal.proved && throw(ArgumentError(
        "with_best_slope applies to heuristic goals only"))
    shape = _goal_shape(goal.n)
    shape > 0 || throw(ArgumentError(
        "cannot adjust the base slope of a goal of dimension $(goal.n)"))

    gap = goal.quality * shape / goal.n
    new_gap = (gap + (goal.best_slope - Float64(slope))) * goal.n
    new_gap > 0 || throw(ArgumentError(
        "base slope $slope is at or above the goal's target slope " *
        "$(goal_max_drop(goal) / goal.n), leaving no excess to attribute"))

    return ReductionGoal(goal.n, goal.top_N, new_gap / shape, Float64(slope),
                         goal.top_slope, goal.g, goal.log_g, false)
end

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

"""
    goal_from_slope(n, slope; proved=false) -> ReductionGoal

A goal targeting the given profile slope.

For a heuristic goal the slope is clamped up to `BKZ_BEST_SLOPE`, since no
reduction is expected to beat it, and `quality` is chosen so that
`goal_max_drop(goal) / n` recovers the clamped slope exactly.
"""
function goal_from_slope(n::Integer, slope::Real; proved::Bool = false)
    n >= 1 || throw(ArgumentError("dimension must be at least one"))
    proved && return proved_goal(n, slope)

    shape = _goal_shape(n)
    # _goal_shape(1) is zero, so a one-dimensional goal has no budget to scale.
    # It is accepted unconditionally by goal_check anyway.
    iszero(shape) && return heuristic_goal(n, 0.0)

    top_slope = max(Float64(slope), BKZ_BEST_SLOPE)
    return heuristic_goal(n, (top_slope - BKZ_BEST_SLOPE) * n / shape)
end

"""
    goal_from_rhf(n, rhf; proved=false) -> ReductionGoal

A goal targeting the given root Hermite factor, via `slope = 2 * log2(rhf)`.

Note that a heuristic goal clamps the slope at `BKZ_BEST_SLOPE`, so any `rhf`
below `2^(BKZ_BEST_SLOPE/2)`, about 1.0109, produces the same goal as that
value. Asking for a better factor than the clamp does not make the goal
stricter.
"""
goal_from_rhf(n::Integer, rhf::Real; proved::Bool = false) =
    goal_from_slope(n, log2(Float64(rhf)) * 2; proved = proved)

"""
    goal_from_drop(n, drop; proved=false) -> ReductionGoal

A goal targeting the given total profile drop.

The heuristic path converts with `drop / (n - 1)`, one slope per gap between
profile entries.

Carried over from flatter verbatim: the proved path converts with `drop / n`,
clamps the result just above `BKZ_BEST_SLOPE`, and then calls the slope
conversion **without forwarding the proved flag**, so it returns a heuristic
goal. That looks unintended (`from_slope(n, slope)` rather than
`from_slope(n, slope, proved)`), but the branch is unreachable in flatter: all
three call sites pass two arguments, so `proved` is false and the other branch
runs. Reproduced as written rather than silently corrected. See Notes.md.
"""
function goal_from_drop(n::Integer, drop::Real; proved::Bool = false)
    n >= 1 || throw(ArgumentError("dimension must be at least one"))
    if proved
        slope = max(Float64(drop) / n, BKZ_BEST_SLOPE + 0.000001)
        return goal_from_slope(n, slope)          # flatter drops `proved` here
    end
    n >= 2 || throw(ArgumentError(
        "a drop-based heuristic goal needs dimension at least two"))
    return goal_from_slope(n, Float64(drop) / (n - 1); proved = false)
end

# ---------------------------------------------------------------------------
# The acceptance test
# ---------------------------------------------------------------------------

"""
    goal_check(goal, prof) -> Bool

Whether the profile `prof` meets this goal.

A one-dimensional goal is met unconditionally. A proved goal asks only that the
total drop stay under `alpha(n) * n`.

A heuristic goal asks three things at once, which is what stops the recursion
settling for a profile that is flat overall but badly shaped:

  1. the total drop is within budget;
  2. the mean of the left half exceeds the mean of the right half by less than
     `mu_sep`, so the drop is not concentrated as a step between the halves;
  3. the drop across the middle span — the region straddling the halves — is
     within what is left of the budget after the two halves claim their own.

Condition 3 exists because a profile whose entire drop sits in the gap between
the two halves would pass conditions 1 and 2 while still being improvable by
further reduction across that gap.
"""
function goal_check(goal::ReductionGoal, prof::AbstractVector{<:Real})
    goal.n != 0 || throw(ArgumentError("cannot check an empty goal"))
    length(prof) == goal.n || throw(DimensionMismatch(
        "goal has dimension $(goal.n) but the profile has $(length(prof)) entries"))

    goal.n == 1 && return true

    n = goal.n
    max_drop = goal_max_drop(goal)

    if goal.proved
        return profile_drop(prof) < max_drop
    end

    gamma = goal.quality * Float64(n)^log2(3.0)      # quality * 3^log2(n)
    mu_sep = (max_drop - gamma) / 2 + gamma

    # The halves. Three is split as 2 + 1 rather than 1 + 2.
    n_L = n == 3 ? 2 : div(n, 2)
    n_R = n - n_L

    l_drop = goal.best_slope * n_L + goal.quality * _goal_shape(n_L)
    r_drop = goal.best_slope * n_R + goal.quality * _goal_shape(n_R)

    # The middle span: from the midpoint of the left half to the midpoint of the
    # right half, so it straddles the boundary the halves were split at.
    n1 = n_L == 3 ? 2 : div(n_L, 2)
    n3 = n_R == 3 ? 2 : div(n_R, 2)
    mid_drop = profile_drop(view(prof, (n1 + 1):(n_L + n3)))

    mu_L = sum(Float64, view(prof, 1:n_L)) / n_L
    mu_R = sum(Float64, view(prof, (n_L + 1):n)) / n_R

    # The third condition is written as flatter writes it; it reduces to
    # mid_drop <= max_drop - r_drop, but the original form shows the intent.
    return profile_drop(prof) < max_drop &&
           mu_L - mu_R < mu_sep &&
           mid_drop <= l_drop + (max_drop - l_drop - r_drop)
end
