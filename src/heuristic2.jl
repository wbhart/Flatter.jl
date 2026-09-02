# heuristic2.jl
#
# The phase-2 heuristic from flatter.
#
# Heuristic2 is not a second tiled kernel.  It is the phase which prepares an
# upper-triangular compressed representation for Heuristic3.  A phase-2 split
# has one child at a time:
#
#     left half  -> phase 2, carrying the right-hand block as B2
#     right half -> phase 2
#     whole      -> phase 3 (Heuristic3)
#
# The B2 case in flatter is dispatched through LatRedRelSR: reduce the reference
# block first, then relative-size-reduce B2 against the reduced result and fold
# the reference transform into U2.  Keeping that wrapper explicit here avoids
# threading B2 through every recursive state while preserving the same algebra.
#
# H2's simulated basis is compressed with ONE common power-of-two shift for all
# columns.  This is important: unlike H3's per-column compression, a scalar
# shift commutes with every column transform, so the transforms produced by the
# compressed recursion can be accumulated directly and applied once to the
# original basis at the end.

"Return the maximum binary width of a nonzero integer matrix entry."
function _h2_widest_bits(A::AbstractMatrix{<:Integer})
    widest = 0
    for value in A
        iszero(value) || (widest = max(widest, ndigits(value; base = 2)))
    end
    return widest
end

"The current-drop approximation used by flatter's Heuristic2."
function _h2_current_drop(profile::AbstractVector{<:Real})
    n = length(profile)
    mid = div(n, 2)
    left = 0.0
    right = 0.0
    for i in 1:mid
        left += Float64(profile[i])
    end
    for i in (mid + 1):n
        right += Float64(profile[i])
    end
    left /= mid
    right /= n - mid
    drop = left - right
    return drop < 0 ? profile_drop(profile) : drop
end

"Choose Heuristic2's common compression shift and next working precision."
function _h2_uniform_plan(profile::Vector{Float64}, n::Int, aggressive::Bool)
    applied, precision = _compression_plan(profile, n, aggressive)
    # flatter computes the usual shifts, then replaces every one by shifts[0].
    return applied[1], precision
end

"Initial phase-2 compression: upper triangular input, one scalar shift."
function _h2_initial_compress(B::AbstractMatrix{T}, profile::Vector{Float64},
                              aggressive::Bool) where {T<:Integer}
    n = size(B, 2)
    shift, precision = _h2_uniform_plan(profile, n, aggressive)
    compressed = zeros(T, n, n)
    for j in 1:n
        for i in 1:j
            compressed[i, j] = shift >= 0 ? B[i, j] >> shift : B[i, j] << (-shift)
        end
        profile[j] -= shift
    end
    return compressed, shift, precision
end

"Direct MPFR Householder factorisation of an exact integer block."
function _h2_direct_factor(A::AbstractMatrix{T}, precision::Integer) where {T<:Integer}
    m, n = size(A)
    widest = _h2_widest_bits(A)
    # The profile-derived figure is flatter's policy.  The extra lower bound is
    # the same safety used by H3's Julia tile QR: unlike flatter's C++ Matrix,
    # this teaching port materialises the raw integer tile before factorising it.
    bits = max(Int(precision), householder_precision(min(m, n), Float64(widest)))
    return with_precision(bits) do
        F = Matrix{BigFloat}(undef, m, n)
        for j in 1:n, i in 1:m
            F[i, j] = BigFloat(A[i, j])
        end
        factors, tau = _heuristic3_direct_qr!(F)
        (factors, tau, bits)
    end
end

"Scale an integer upper-triangular entry by a common power of two."
@inline _h2_shift_integer(x::T, shift::Int) where {T<:Integer} =
    shift >= 0 ? x >> shift : x << (-shift)

"Compress the representation after the left-half phase-2 step."
function _h2_compress_left(factors::AbstractMatrix{BigFloat},
                           candidate::AbstractMatrix{T}, k::Int,
                           profile::Vector{Float64}, aggressive::Bool,
                           factor_bits::Int) where {T<:Integer}
    n = size(candidate, 2)
    shift, precision = _h2_uniform_plan(profile, n, aggressive)
    next = zeros(T, n, n)

    with_precision(factor_bits) do
        for j in 1:n
            for i in 1:min(k, j)
                next[i, j] = round(T, ldexp(factors[i, j], -shift))
            end
        end
    end
    for j in (k + 1):n
        for i in (k + 1):j
            next[i, j] = _h2_shift_integer(candidate[i, j], shift)
        end
    end
    for i in 1:n
        profile[i] -= shift
    end
    return next, shift, precision
end

"Compress the representation after the right-half phase-2 step."
function _h2_compress_right(factors::AbstractMatrix{BigFloat},
                            candidate::AbstractMatrix{T}, k::Int,
                            profile::Vector{Float64}, aggressive::Bool,
                            factor_bits::Int) where {T<:Integer}
    n = size(candidate, 2)
    shift, precision = _h2_uniform_plan(profile, n, aggressive)
    next = zeros(T, n, n)

    # The top rows are already triangular and relatively size reduced.
    for j in 1:n
        for i in 1:min(k, j)
            next[i, j] = _h2_shift_integer(candidate[i, j], shift)
        end
    end

    # The lower-right child is general; keep only the R factor.
    with_precision(factor_bits) do
        r = n - k
        for jj in 1:r
            j = k + jj
            for ii in 1:jj
                i = k + ii
                next[i, j] = round(T, ldexp(factors[ii, jj], -shift))
            end
        end
    end
    for i in 1:n
        profile[i] -= shift
    end
    return next, shift, precision
end

"""
    _h2_collect_steps(steps, bounds, n)

Collect phase-2 iteration transforms in the same structured reverse order used by
flatter's `Heuristic3::collect_U` (which Heuristic2 inherits).  Each earlier
iteration is identity outside the row interval recorded in `bounds`; exploiting
that structure avoids a full `n x n` BigInt product after every H2 step.

For an iteration with affected interval `start:finish`, upstream forms only

    U_iter[1:finish, start:n] * U[start:n, :]

and then adds/replaces the affected rows.  Uniform H2 compression needs no
`D^-1 U D` correction because every diagonal entry of `D` is the same.
"""
function _h2_collect_steps(steps::Vector{Matrix{T}},
                           bounds::Vector{Tuple{Int, Int}},
                           n::Int) where {T<:Integer}
    length(steps) == length(bounds) || throw(DimensionMismatch(
        "there are $(length(steps)) H2 transforms but $(length(bounds)) bounds"))

    if isempty(steps)
        U = Matrix{T}(undef, n, n)
        _set_identity!(U)
        return U
    end

    U = copy(steps[end])
    for idx in (length(steps) - 1):-1:1
        U_iter = steps[idx]
        start, finish = bounds[idx]
        1 <= start <= finish <= n || throw(ArgumentError(
            "invalid H2 transform bounds $start:$finish for dimension $n"))

        # This is the 1-based form of flatter's structured collect_U product.
        product = _reduction_mul(
            Matrix{T}(view(U_iter, 1:finish, start:n)),
            Matrix{T}(view(U, start:n, 1:n)))

        for j in 1:n
            for i in 1:(start - 1)
                U[i, j] += product[i, j]
            end
            for i in start:finish
                U[i, j] = product[i, j]
            end
        end
    end
    return U
end

"Embed the left child and its auxiliary-block transform into one phase-2 step."
function _h2_left_transform(::Type{T}, n::Int, k::Int,
                            U_sub::AbstractMatrix{T}, U2::AbstractMatrix{T}) where {T<:Integer}
    U = Matrix{T}(undef, n, n)
    _set_identity!(U)
    for j in 1:k, i in 1:k
        U[i, j] = U_sub[i, j]
    end
    for j in (k + 1):n, i in 1:k
        U[i, j] = U2[i, j - k]
    end
    return U
end

"Embed the right child and the subsequent relative size reduction."
function _h2_right_transform(::Type{T}, n::Int, k::Int,
                             U_sub::AbstractMatrix{T}, U_sr::AbstractMatrix{T}) where {T<:Integer}
    U = Matrix{T}(undef, n, n)
    _set_identity!(U)
    r = n - k
    for jj in 1:r, ii in 1:r
        U[k + ii, k + jj] = U_sub[ii, jj]
    end
    for jj in 1:r, i in 1:k
        U[i, k + jj] = U_sr[i, jj]
    end
    return U
end

"A LatRedRelSR-style child: reduce B1, then reduce B2 against the result."
function _heuristic2_relative_child!(B1::AbstractMatrix{T}, B2::AbstractMatrix{T},
                                     U1::AbstractMatrix{T}, U2::AbstractMatrix{T};
                                     goal::ReductionGoal,
                                     split::SplitPhase2,
                                     max_iterations::Int,
                                     aggressive::Bool,
                                     schoenhage_threshold::Int,
                                     base_cutoff::Int,
                                     blocksize::Int,
                                     panelsize::Int,
                                     validate::Bool,
                                     telemetry::Union{Nothing, ReductionTelemetry},
                                     depth::Int) where {T<:Integer}
    _, _, info = _heuristic2_reduce!(B1, U1;
        goal = goal, split = split,
        max_iterations = max_iterations, aggressive = aggressive,
        schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
        blocksize = blocksize, panelsize = panelsize,
        validate = validate, telemetry = telemetry, depth = depth)

    n = size(B1, 2)
    width2 = size(B2, 2)
    U_rel = zeros(T, n, width2)
    if width2 > 0
        # LatRedRelSR chooses precision from the reduced reference profile, with
        # extra headroom when B2 lives at a larger exponent.  Both blocks here
        # share H2's scalar compression, so an entry-width difference is the
        # corresponding exact-integer proxy.
        spread = isempty(info.profile) ? profile_spread(_basis_profile(B1, aggressive)) :
                                         profile_spread(info.profile)
        b1_bits = _h2_widest_bits(B1)
        b2_bits = _h2_widest_bits(B2)
        # LatRedRelSR uses `2 * (spread + 30)` and then widens further when
        # the auxiliary block is represented at a larger exponent than B1.
        # MPZ matrices do not carry a separate precision in this Julia port, so
        # the difference of exact entry widths is the corresponding proxy.
        bits = max(ceil(Int, 2 * (spread + 30)), max(0, b2_bits - b1_bits))
        qr_started = _tick()
        factors, tau, actual_bits = _h2_direct_factor(B1, bits)
        telemetry === nothing || (telemetry.h2_time_wrapper_qr += _tock(qr_started))
        started = _tick()
        with_precision(actual_bits) do
            _relative_orthogonal_reflectors!(B1, B2, U_rel, factors, tau, nothing;
                deadband = RELATIVE_SIZE_REDUCTION_DEADBAND,
                max_passes = RELATIVE_SIZE_REDUCTION_MAX_PASSES)
        end
        if telemetry !== nothing
            telemetry.h2_relative_calls += 1
            telemetry.h2_time_relative += _tock(started)
        end
    end

    # B2_new = B2_old + (B1_old*U1)*U_rel, so the off-diagonal block in the
    # transform of the ORIGINAL child basis is U1*U_rel.
    compose_started = _tick()
    U2_total = width2 == 0 ? zeros(T, n, 0) : strassen(U1, U_rel)
    telemetry === nothing || (telemetry.h2_time_u2_compose += _tock(compose_started))
    for j in 1:width2, i in 1:n
        U2[i, j] = U2_total[i, j]
    end
    return info
end

"Check the exact column-transform contract before H2 changes representation."
function _h2_validate_step(working::AbstractMatrix{T}, candidate::AbstractMatrix{T},
                           step_U::AbstractMatrix{T}, label::AbstractString) where {T<:Integer}
    expected = _reduction_mul(Matrix{T}(working), Matrix{T}(step_U))
    expected == candidate || throw(ErrorException(
        "Heuristic2 $label step does not satisfy candidate == working * U"))
    _exact_determinant(step_U) in (one(T), -one(T)) || throw(ErrorException(
        "Heuristic2 $label step produced a non-unimodular transform"))
    return nothing
end

"Internal phase-2 driver.  Inputs are square, upper triangular and nonsingular."
function _heuristic2_reduce!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
                             goal::ReductionGoal,
                             split::SplitPhase2,
                             max_iterations::Int,
                             aggressive::Bool,
                             schoenhage_threshold::Int,
                             base_cutoff::Int,
                             blocksize::Int,
                             panelsize::Int,
                             validate::Bool,
                             telemetry::Union{Nothing, ReductionTelemetry},
                             depth::Int) where {T<:Integer}
    n = size(B, 2)

    # The upstream dispatcher never instantiates H2 below the fplll threshold.
    if n <= base_cutoff || n <= 2
        return lattice_reduce!(B, U;
            goal = goal, max_iterations = max_iterations,
            aggressive = aggressive, schoenhage_threshold = schoenhage_threshold,
            base_cutoff = base_cutoff, blocksize = blocksize, panelsize = panelsize,
            tiled = true, schedule = :split, validate = validate,
            want_profile = true, telemetry = telemetry, _depth = depth)
    end

    if telemetry !== nothing
        telemetry.levels += 1
        telemetry.max_depth = max(telemetry.max_depth, depth)
        telemetry.h2_calls += 1
    end

    entry_profile = [_log2_abs(B[i, i]) for i in 1:n]
    profile = copy(entry_profile)
    if goal_check(goal, profile)
        # Do not materialise a second copy of the whole basis on the common
        # entry-goal fast path.  `original` is needed only by the final exact
        # validation after H2 has actually changed the basis.
        _set_identity!(U)
        return B, U, (iterations = 0, goal_met = true,
                      stopped = :goal, profile = profile)
    end

    original = Matrix{T}(B)
    compression_started = _tick()
    working, total_shift, precision = _h2_initial_compress(B, profile, aggressive)
    telemetry === nothing || (telemetry.h2_time_compress += _tock(compression_started))

    step_transforms = Matrix{T}[]
    step_bounds = Tuple{Int, Int}[]
    iterations = 0
    stopped = :phase2_complete
    child_profile = copy(profile)

    while !split_stopping_point(split)
        if iterations >= max_iterations
            stopped = :cap
            telemetry === nothing || (telemetry.capped += 1)
            break
        end

        window = only(split_windows(split))
        child = only(split_children(split))
        start = first(window)
        finish = last(window)
        width = length(window)
        inherited = subgoal(goal, start - 1, finish)

        current_drop = _h2_current_drop(profile)
        child_goal = inherited
        if !(start == 1 && finish == n)
            drop_goal = goal_from_drop(width, current_drop / 6)
            child_goal = goal_slope(inherited) < goal_slope(drop_goal) ? drop_goal : inherited
        end

        if start == 1 && finish < n
            # L: phase-2 child plus the right-hand block, via LatRedRelSR.
            telemetry === nothing || (telemetry.h2_left_steps += 1)
            B_sub = Matrix{T}(working[window, window])
            B2 = Matrix{T}(working[window, (finish + 1):n])
            U_sub = Matrix{T}(undef, width, width)
            U2 = Matrix{T}(undef, width, n - finish)
            info = _heuristic2_relative_child!(B_sub, B2, U_sub, U2;
                goal = child_goal, split = child::SplitPhase2,
                max_iterations = max_iterations, aggressive = aggressive,
                schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
                blocksize = blocksize, panelsize = panelsize,
                validate = validate, telemetry = telemetry, depth = depth + 1)

            candidate = Matrix{T}(working)
            for j in 1:width, i in 1:width
                candidate[i, j] = B_sub[i, j]
            end
            for jj in 1:(n - finish), i in 1:width
                candidate[i, finish + jj] = B2[i, jj]
            end
            step_U = _h2_left_transform(T, n, finish, U_sub, U2)

            validate && _h2_validate_step(working, candidate, step_U, "left")

            child_profile = isempty(info.profile) ? _basis_profile(B_sub, aggressive) : info.profile
            for i in 1:finish
                profile[i] = child_profile[i]
            end
            precision = lll_precision(profile_spread(profile), n; aggressive = aggressive)

            qr_started = _tick()
            factors, _, factor_bits = _h2_direct_factor(view(candidate, 1:finish, 1:n), precision)
            telemetry === nothing || (telemetry.h2_time_qr += _tock(qr_started))
            # Upstream reads the profile back from this R before choosing the
            # next common compression shift.  The first `finish` diagonals are
            # the child's GS norms; recomputing them also keeps the numerical
            # representation and compression policy in lock-step.
            for i in 1:finish
                profile[i] = _log2_abs(factors[i, i])
            end
            compress_started = _tick()
            working, shift, precision = _h2_compress_left(
                factors, candidate, finish, profile, aggressive, factor_bits)
            total_shift += shift
            telemetry === nothing || (telemetry.h2_time_compress += _tock(compress_started))

        elseif start > 1 && finish == n
            # R: phase-2 child, then exact relative SR against the untouched left.
            telemetry === nothing || (telemetry.h2_right_steps += 1)
            B_sub = Matrix{T}(working[window, window])
            U_sub = Matrix{T}(undef, width, width)
            _, _, info = _heuristic2_reduce!(B_sub, U_sub;
                goal = child_goal, split = child::SplitPhase2,
                max_iterations = max_iterations, aggressive = aggressive,
                schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
                blocksize = blocksize, panelsize = panelsize,
                validate = validate, telemetry = telemetry, depth = depth + 1)

            k = start - 1
            # First apply the right child's column transform to the upper-right
            # block, exactly as B_next(0:start,start:end) = B*U_sub upstream.
            top_started = _tick()
            top_right = strassen(Matrix{T}(view(working, 1:k, window)), U_sub)
            telemetry === nothing || (telemetry.h2_time_top_right += _tock(top_started))
            candidate = Matrix{T}(working)
            for jj in 1:width, i in 1:k
                candidate[i, k + jj] = top_right[i, jj]
            end
            for jj in 1:width, ii in 1:width
                candidate[k + ii, k + jj] = B_sub[ii, jj]
            end

            U_sr = zeros(T, k, width)
            sr_started = _tick()
            relative_size_reduction!(Matrix{T}(view(working, 1:k, 1:k)),
                                     view(candidate, 1:k, window), U_sr;
                                     triangular = true, coordinates = false)
            telemetry === nothing || (telemetry.h2_time_relative += _tock(sr_started))
            step_U = _h2_right_transform(T, n, k, U_sub, U_sr)
            validate && _h2_validate_step(working, candidate, step_U, "right")

            child_profile = isempty(info.profile) ? _basis_profile(B_sub, aggressive) : info.profile
            for (ii, i) in enumerate(window)
                profile[i] = child_profile[ii]
            end
            precision = lll_precision(profile_spread(profile), n; aggressive = aggressive)

            qr_started = _tick()
            factors, _, factor_bits = _h2_direct_factor(B_sub, precision)
            telemetry === nothing || (telemetry.h2_time_qr += _tock(qr_started))
            for ii in 1:width
                profile[k + ii] = _log2_abs(factors[ii, ii])
            end
            compress_started = _tick()
            working, shift, precision = _h2_compress_right(
                factors, candidate, k, profile, aggressive, factor_bits)
            total_shift += shift
            telemetry === nothing || (telemetry.h2_time_compress += _tock(compress_started))

        else
            # ALL: the phase transition.  The child split is phase 3 and the
            # inherited goal is used without the /6 throttle, exactly upstream.
            telemetry === nothing || begin
                telemetry.h2_all_steps += 1
                telemetry.h2_phase3_calls += 1
            end
            B_sub = Matrix{T}(working)
            U_sub = Matrix{T}(undef, n, n)
            _, _, info = lattice_reduce!(B_sub, U_sub;
                goal = inherited, max_iterations = max_iterations,
                aggressive = aggressive, schoenhage_threshold = schoenhage_threshold,
                base_cutoff = base_cutoff, blocksize = blocksize, panelsize = panelsize,
                tiled = true, schedule = :split, validate = validate,
                want_profile = true, _split = child::SplitPhase3,
                telemetry = telemetry, _depth = depth + 1)
            step_U = U_sub
            validate && _h2_validate_step(working, B_sub, step_U, "all")
            child_profile = isempty(info.profile) ? _basis_profile(B_sub, aggressive) : info.profile
            profile = child_profile
        end

        push!(step_transforms, step_U)
        push!(step_bounds, (start, finish))
        split_advance!(split)
        iterations += 1
        telemetry === nothing || (telemetry.iterations += 1)
    end

    # Uniform compression is a scalar row scaling, so every accumulated column
    # transform is valid unchanged at true scale.  Like upstream, collect the
    # structured iteration transforms only once, in reverse order.
    collect_started = _tick()
    total = _h2_collect_steps(step_transforms, step_bounds, n)
    telemetry === nothing || (telemetry.h2_time_collect += _tock(collect_started))

    apply_started = _tick()
    reduced = _reduction_mul(original, total)
    for j in 1:n, i in 1:n
        B[i, j] = reduced[i, j]
        U[i, j] = total[i, j]
    end
    telemetry === nothing || (telemetry.h2_time_apply += _tock(apply_started))

    # The full phase-3 child reports its profile at the current compressed scale.
    # Restore H2's scalar compression to report the profile of the true basis.
    true_profile = profile .+ total_shift
    goal_met = goal_check(goal, true_profile)

    if validate
        expected = _reduction_mul(original, total)
        expected == B || throw(ErrorException(
            "Heuristic2 returned a basis inconsistent with its accumulated transform"))
        _exact_determinant(total) in (one(T), -one(T)) || throw(ErrorException(
            "Heuristic2 returned a non-unimodular transform"))
    end

    return B, U, (iterations = iterations, goal_met = goal_met,
                  stopped = stopped, profile = true_profile)
end

"""
    heuristic2_reduce!(B, U; kwargs...) -> (B, U, info)

Reduce a square upper-triangular integer basis with flatter's phase-2 heuristic.

The phase-2 schedule is left, right, whole. Partial children remain phase 2; the
whole-window child is handed to the already-ported phase-3 Heuristic3. Small
children use the same fplll base case as [`lattice_reduce!`](@ref).

This remains an explicit phase-2 entry point as well as the implementation used
by the public dispatcher. With CondUnknown and Heuristic1 now ported, the public
heuristic route reaches phase 2 through the same triangular, known-condition and
unknown-condition entry routes as flatter.
"""
function heuristic2_reduce!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
                            goal::Union{Nothing, ReductionGoal} = nothing,
                            rhf::Real = DEFAULT_REDUCTION_RHF,
                            max_iterations::Integer = DEFAULT_REDUCTION_MAX_ITERATIONS,
                            aggressive::Bool = false,
                            schoenhage_threshold::Integer = DEFAULT_SCHOENHAGE_THRESHOLD,
                            base_cutoff::Integer = DEFAULT_BASE_CUTOFF,
                            blocksize::Integer = DEFAULT_FUSED_BLOCKSIZE,
                            panelsize::Integer = DEFAULT_FUSED_PANELSIZE,
                            validate::Bool = false,
                            telemetry::Union{Nothing, ReductionTelemetry} = nothing,
                            _split = nothing,
                            _depth::Integer = 0) where {T<:Integer}
    n = _check_reduction_input(B)
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be at least one"))
    base_cutoff >= 2 || throw(ArgumentError("base_cutoff must be at least two"))

    resolved_goal = goal === nothing ? goal_from_rhf(n, rhf) : goal
    resolved_goal.n == n || throw(ArgumentError(
        "the goal has dimension $(resolved_goal.n) but the basis has $n columns"))
    split = _split === nothing ? SplitPhase2(n) : _split
    split isa SplitPhase2 || throw(ArgumentError(
        "Heuristic2 needs a SplitPhase2 tree, got $(typeof(split))"))
    split.n == n || throw(ArgumentError(
        "the phase-2 split has dimension $(split.n) but the basis has $n columns"))

    return _heuristic2_reduce!(B, U;
        goal = resolved_goal, split = split,
        max_iterations = Int(max_iterations), aggressive = aggressive,
        schoenhage_threshold = Int(schoenhage_threshold), base_cutoff = Int(base_cutoff),
        blocksize = Int(blocksize), panelsize = Int(panelsize),
        validate = validate, telemetry = telemetry, depth = Int(_depth))
end

"Non-mutating form of [`heuristic2_reduce!`](@ref)."
function heuristic2_reduce(B::AbstractMatrix{T}; kwargs...) where {T<:Integer}
    reduced = Matrix{T}(B)
    transform = Matrix{T}(undef, size(B, 2), size(B, 2))
    return heuristic2_reduce!(reduced, transform; kwargs...)
end
