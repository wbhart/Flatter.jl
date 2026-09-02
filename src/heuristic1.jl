# heuristic1.jl
#
# flatter's phase-1 heuristic for a full-column-rank basis whose condition
# number is known.  Heuristic1 inherits most of Heuristic2 upstream, but its
# working representation is rectangular and initially uncompressed.  Partial
# children remain in phase 1; the whole-window child enters phase 3.
#
# The important difference from H2 is B2.  A left child is given both the
# right-hand basis columns and any external auxiliary block, so its U2 block is
# part of the phase-1 transform.  We keep that algebra explicit below rather
# than forcing the rectangular state through the square H2 wrapper.

"Precision used by upstream Heuristic1 from the caller's log2(cond) bound."
_h1_precision(log_cond::Real, n::Int, aggressive::Bool) =
    lll_precision(log_cond, n; aggressive = aggressive)

"Direct Householder QR at Heuristic1's fixed known-condition precision."
function _h1_direct_factor(A::AbstractMatrix{T}, log_cond::Real, n::Int,
                           aggressive::Bool) where {T<:Integer}
    bits = _h1_precision(log_cond, n, aggressive)
    return with_precision(bits) do
        F = Matrix{BigFloat}(undef, size(A, 1), size(A, 2))
        for j in axes(A, 2), i in axes(A, 1)
            F[i, j] = BigFloat(A[i, j])
        end
        factors, tau = _heuristic3_direct_qr!(F)
        (factors, tau, bits)
    end
end

"Initial phase-1 profile: upstream `Profile(n)` is valid but filled with NaNs."
_h1_initial_profile(n::Integer) = fill(NaN, Int(n))

"The deliberately asymmetric current-drop estimate in upstream Heuristic1."
function _h1_current_drop(profile::AbstractVector{<:Real},
                          global_offsets::AbstractVector{<:Real})
    n = length(profile)
    mid = div(n, 2)
    mid > 0 || return 0.0
    mu_left = 0.0
    for i in 1:mid
        mu_left += Float64(profile[i]) + Float64(global_offsets[i])
    end
    mu_left /= mid
    # Upstream initialises mu_R to zero and never fills it in Heuristic1.
    drop = mu_left
    return drop < 0 ? profile_drop(profile) : drop
end

"Choose the tall row count used for one phase-1 child."
function _h1_sub_rows(B::AbstractMatrix{<:Integer}, start::Int, finish::Int,
                      original_precision::Int)
    m, n = size(B)
    start0 = start - 1
    rows = m - start0

    if start == 1 && finish == n
        return n
    elseif start == 1
        # Left child: discard only genuinely zero trailing rows.  For ordinary
        # (<10000-bit) input upstream checks all n columns; at extreme precision
        # it checks just the child columns.
        rbound = original_precision < 10000 ? n : finish
        while rows > finish
            row = rows
            any(!iszero(B[row, j]) for j in 1:rbound) && break
            rows -= 1
        end
    else
        # This is upstream's exact condition: trim while rows > n, not while
        # rows > child width.
        while rows > n
            row = start0 + rows
            any(!iszero(B[row, j]) for j in 1:n) && break
            rows -= 1
        end
    end
    return rows
end

"Apply H1's scalar compression to a plain integer entry."
@inline _h1_shift_integer(x::T, shift::Int) where {T<:Integer} =
    _h2_shift_integer(x, shift)

"H1's common shift: current simulated entry width minus known-cond precision."
_h1_total_shift(B::AbstractMatrix{<:Integer}, log_cond::Real, n::Int,
                aggressive::Bool) =
    _h2_widest_bits(B) - _h1_precision(log_cond, n, aggressive)

"Build and factor [Bpart B2part] without changing the caller's matrices."
function _h1_factor_combined(Bpart::AbstractMatrix{T}, B2part::AbstractMatrix{T},
                             log_cond::Real, n::Int,
                             aggressive::Bool) where {T<:Integer}
    rows = size(Bpart, 1)
    q = size(B2part, 2)
    combined = Matrix{T}(undef, rows, size(Bpart, 2) + q)
    for j in axes(Bpart, 2), i in 1:rows
        combined[i, j] = Bpart[i, j]
    end
    off = size(Bpart, 2)
    for j in 1:q, i in 1:rows
        combined[i, off + j] = B2part[i, j]
    end
    return _h1_direct_factor(combined, log_cond, n, aggressive)
end

"Reduce an auxiliary B2 block against an already-reduced reference basis."
function _h1_reduce_auxiliary!(B1::AbstractMatrix{T}, B2::AbstractMatrix{T},
                               U1::AbstractMatrix{T}, U2::AbstractMatrix{T},
                               profile::AbstractVector{<:Real};
                               aggressive::Bool,
                               telemetry::Union{Nothing, ReductionTelemetry}) where {T<:Integer}
    n = size(B1, 2)
    q = size(B2, 2)
    size(U2) == (n, q) || throw(DimensionMismatch(
        "U2 must be $n x $q, got $(size(U2))"))
    fill!(U2, zero(T))
    q == 0 && return U2

    spread = isempty(profile) ? profile_spread(_basis_profile(B1, aggressive)) :
                                profile_spread(profile)
    b1_bits = _h2_widest_bits(B1)
    b2_bits = _h2_widest_bits(B2)
    bits = max(ceil(Int, 2 * (spread + 30)), max(0, b2_bits - b1_bits))

    qr_started = _tick()
    factors, tau, actual_bits = _h2_direct_factor(B1, bits)
    telemetry === nothing || (telemetry.h1_time_wrapper_qr += _tock(qr_started))

    U_rel = zeros(T, n, q)
    sr_started = _tick()
    with_precision(actual_bits) do
        _relative_orthogonal_reflectors!(B1, B2, U_rel, factors, tau, nothing;
            deadband = RELATIVE_SIZE_REDUCTION_DEADBAND,
            max_passes = RELATIVE_SIZE_REDUCTION_MAX_PASSES)
    end
    if telemetry !== nothing
        telemetry.h1_relative_calls += 1
        telemetry.h1_time_relative += _tock(sr_started)
    end

    compose_started = _tick()
    total = strassen(U1, U_rel)
    telemetry === nothing || (telemetry.h1_time_u2_compose += _tock(compose_started))
    for j in 1:q, i in 1:n
        U2[i, j] = total[i, j]
    end
    return U2
end

"Base/phase-3 child with optional LatRedRelSR-style auxiliary reduction."
function _h1_reference_child!(B1::AbstractMatrix{T}, B2::AbstractMatrix{T},
                              U1::AbstractMatrix{T}, U2::AbstractMatrix{T};
                              goal::ReductionGoal,
                              child_split,
                              phase3::Bool,
                              max_iterations::Int,
                              aggressive::Bool,
                              schoenhage_threshold::Int,
                              base_cutoff::Int,
                              blocksize::Int,
                              panelsize::Int,
                              validate::Bool,
                              telemetry::Union{Nothing, ReductionTelemetry},
                              depth::Int) where {T<:Integer}
    n = size(B1, 2)
    info = nothing
    if n <= 2
        started = _tick()
        # The C++ dispatcher only selects Lagrange for the 2 x 2 case.  A tall
        # two-column phase-1 child goes to Schoenhage even when its entries are
        # small, so force the existing Gram-matrix path for that case.
        base_threshold = (n == 2 && size(B1, 1) != 2) ? 0 : schoenhage_threshold
        _reduce_two_columns!(B1, U1, base_threshold)
        telemetry === nothing || begin
            telemetry.base_cases += 1
            telemetry.time_base += _tock(started)
        end
        profile = _basis_profile(B1, aggressive)
        info = (iterations = 0, goal_met = goal_check(goal, profile),
                stopped = :base, profile = profile)
    elseif phase3
        _, _, info = lattice_reduce!(B1, U1;
            goal = goal, max_iterations = max_iterations,
            aggressive = aggressive, schoenhage_threshold = schoenhage_threshold,
            base_cutoff = base_cutoff, blocksize = blocksize, panelsize = panelsize,
            tiled = true, schedule = :split, validate = validate,
            want_profile = true, _split = child_split::SplitPhase3,
            telemetry = telemetry, _depth = depth)
    else
        throw(ArgumentError("non-phase3 reference child must be handled by Heuristic1 recursion"))
    end

    child_profile = isempty(info.profile) ? _basis_profile(B1, aggressive) :
                                            Float64.(info.profile)
    if isempty(info.profile)
        info = (iterations = info.iterations, goal_met = info.goal_met,
                stopped = info.stopped, profile = child_profile)
    end
    _h1_reduce_auxiliary!(B1, B2, U1, U2, child_profile;
                          aggressive = aggressive, telemetry = telemetry)
    return info
end

"Internal Heuristic1 driver, including the B2/U2 contract used by left children."
function _heuristic1_reduce!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                             B2::AbstractMatrix{T}, U2::AbstractMatrix{T};
                             goal::ReductionGoal,
                             split::SplitPhase2,
                             log_cond::Real,
                             initial_profile::AbstractVector{<:Real},
                             profile_offset::AbstractVector{<:Real},
                             max_iterations::Int,
                             aggressive::Bool,
                             schoenhage_threshold::Int,
                             base_cutoff::Int,
                             blocksize::Int,
                             panelsize::Int,
                             validate::Bool,
                             telemetry::Union{Nothing, ReductionTelemetry},
                             depth::Int) where {T<:Integer}
    m, n = size(B)
    q = size(B2, 2)
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n"))
    size(B2, 1) == m || throw(DimensionMismatch("B2 must have $m rows"))
    size(U2) == (n, q) || throw(DimensionMismatch("U2 must be $n x $q"))
    length(initial_profile) == n || throw(DimensionMismatch("profile length must be $n"))
    length(profile_offset) == n || throw(DimensionMismatch("profile_offset length must be $n"))
    log_cond > 0 || throw(ArgumentError("Heuristic1 requires log_cond > 0"))

    original = Matrix{T}(B)
    original_B2 = Matrix{T}(B2)
    original_precision = _h2_widest_bits(original)
    profile = Float64.(initial_profile)
    global_offsets = Float64.(profile_offset)
    working = Matrix{T}(B)             # init_compressed_B: exact copy, no shift
    B2_sim = Matrix{T}(B2)
    total_shift = 0

    # Dispatcher base case precedes phase selection upstream, even with B2.
    if n <= 2
        _set_identity!(U)
        fill!(U2, zero(T))
        info = _h1_reference_child!(B, B2, U, U2;
            goal = goal, child_split = split.all, phase3 = false,
            max_iterations = max_iterations, aggressive = aggressive,
            schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
            blocksize = blocksize, panelsize = panelsize, validate = validate,
            telemetry = telemetry, depth = depth)
        return B, U, B2, U2, info
    end

    if telemetry !== nothing
        telemetry.levels += 1
        telemetry.max_depth = max(telemetry.max_depth, depth)
        telemetry.h1_calls += 1
    end

    step_transforms = Matrix{T}[]
    step_bounds = Tuple{Int, Int}[]
    external_U2 = zeros(T, n, q)
    iterations = 0
    stopped = :phase1_complete

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
        bsub_rows = _h1_sub_rows(working, start, finish, original_precision)

        inherited = subgoal(goal, start - 1, finish)
        child_goal = inherited
        if !(start == 1 && finish == n)
            current_drop = _h1_current_drop(profile, global_offsets)
            drop_goal = goal_from_drop(width, current_drop / 6)
            child_goal = goal_quality(inherited) < goal_quality(drop_goal) ?
                         drop_goal : inherited
        end

        step_U = Matrix{T}(undef, n, n)
        _set_identity!(step_U)

        if start == 1 && finish < n
            telemetry === nothing || (telemetry.h1_left_steps += 1)
            internal_q = n - finish
            B_sub = Matrix{T}(view(working, 1:bsub_rows, window))
            child_B2 = Matrix{T}(undef, bsub_rows, internal_q + q)
            for j in 1:internal_q, i in 1:bsub_rows
                child_B2[i, j] = working[i, finish + j]
            end
            for j in 1:q, i in 1:bsub_rows
                child_B2[i, internal_q + j] = B2_sim[i, j]
            end
            U_sub = Matrix{T}(undef, width, width)
            child_U2 = Matrix{T}(undef, width, internal_q + q)
            _, _, _, _, info = _heuristic1_reduce!(B_sub, U_sub, child_B2, child_U2;
                goal = child_goal, split = child::SplitPhase2,
                log_cond = log_cond, initial_profile = view(profile, window),
                profile_offset = view(global_offsets, window),
                max_iterations = max_iterations, aggressive = aggressive,
                schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
                blocksize = blocksize, panelsize = panelsize, validate = validate,
                telemetry = telemetry, depth = depth + 1)

            for i in 1:finish
                profile[i] = info.profile[i]
            end
            step_U = _h2_left_transform(T, n, finish, U_sub,
                                        view(child_U2, :, 1:internal_q))
            if q > 0
                for j in 1:q, i in 1:finish
                    external_U2[i, j] = child_U2[i, internal_q + j]
                end
            end

            candidate = Matrix{T}(working)
            for j in 1:finish, i in 1:bsub_rows
                candidate[i, j] = B_sub[i, j]
            end
            for jj in 1:internal_q, i in 1:bsub_rows
                candidate[i, finish + jj] = child_B2[i, jj]
            end
            B2_candidate = Matrix{T}(B2_sim)
            for j in 1:q, i in 1:bsub_rows
                B2_candidate[i, j] = child_B2[i, internal_q + j]
            end

            qr_started = _tick()
            factors, _, factor_bits = _h1_factor_combined(
                view(candidate, 1:bsub_rows, 1:n),
                view(B2_candidate, 1:bsub_rows, 1:q),
                log_cond, n, aggressive)
            telemetry === nothing || (telemetry.h1_time_qr += _tock(qr_started))
            for i in 1:min(n, bsub_rows)
                profile[i] = _log2_abs(factors[i, i])
            end

            shift = _h1_total_shift(working, log_cond, n, aggressive)
            total_shift += shift
            for i in 1:n
                profile[i] -= shift
                global_offsets[i] += shift
            end

            compress_started = _tick()
            next = Matrix{T}(candidate)
            next_B2 = Matrix{T}(B2_candidate)
            with_precision(factor_bits) do
                for i in 1:bsub_rows
                    for j in 1:n
                        next[i, j] = j < i ? zero(T) :
                            round(T, ldexp(factors[i, j], -shift))
                    end
                    for j in 1:q
                        cj = n + j
                        next_B2[i, j] = cj < i ? zero(T) :
                            round(T, ldexp(factors[i, cj], -shift))
                    end
                end
            end
            for i in (bsub_rows + 1):m
                for j in 1:n
                    next[i, j] = _h1_shift_integer(candidate[i, j], shift)
                end
                for j in 1:q
                    next_B2[i, j] = _h1_shift_integer(B2_candidate[i, j], shift)
                end
            end
            working = next
            B2_sim = next_B2
            telemetry === nothing || (telemetry.h1_time_compress += _tock(compress_started))

        elseif start > 1 && finish == n
            telemetry === nothing || (telemetry.h1_right_steps += 1)
            k = start - 1
            rows = start:(start + bsub_rows - 1)
            B_sub = Matrix{T}(view(working, rows, window))
            U_sub = Matrix{T}(undef, width, width)
            child_B2 = zeros(T, bsub_rows, 0)
            child_U2 = zeros(T, width, 0)
            _, _, _, _, info = _heuristic1_reduce!(B_sub, U_sub, child_B2, child_U2;
                goal = child_goal, split = child::SplitPhase2,
                log_cond = log_cond, initial_profile = view(profile, window),
                profile_offset = view(global_offsets, window),
                max_iterations = max_iterations, aggressive = aggressive,
                schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
                blocksize = blocksize, panelsize = panelsize, validate = validate,
                telemetry = telemetry, depth = depth + 1)
            for (ii, i) in enumerate(window)
                profile[i] = info.profile[ii]
            end

            candidate = Matrix{T}(working)
            top_started = _tick()
            top_right = strassen(Matrix{T}(view(working, 1:k, window)), U_sub)
            telemetry === nothing || (telemetry.h1_time_top_right += _tock(top_started))
            for jj in 1:width, i in 1:k
                candidate[i, k + jj] = top_right[i, jj]
            end
            for jj in 1:width, ii in 1:bsub_rows
                candidate[k + ii, k + jj] = B_sub[ii, jj]
            end

            U_sr = zeros(T, k, width)
            sr_started = _tick()
            relative_size_reduction!(Matrix{T}(view(working, 1:k, 1:k)),
                                     view(candidate, 1:k, window), U_sr;
                                     triangular = true, coordinates = false)
            telemetry === nothing || (telemetry.h1_time_relative += _tock(sr_started))
            step_U = _h2_right_transform(T, n, k, U_sub, U_sr)

            qr_started = _tick()
            factors, _, factor_bits = _h1_factor_combined(
                view(candidate, rows, window), view(B2_sim, rows, 1:q),
                log_cond, n, aggressive)
            telemetry === nothing || (telemetry.h1_time_qr += _tock(qr_started))
            for ii in 1:width
                profile[k + ii] = _log2_abs(factors[ii, ii])
            end

            shift = _h1_total_shift(working, log_cond, n, aggressive)
            total_shift += shift
            for i in 1:n
                profile[i] -= shift
                global_offsets[i] += shift
            end

            compress_started = _tick()
            next = Matrix{T}(working)
            # Upstream copies only the top rows from B_next and the right-hand R
            # block below them.  Trailing rows excluded by the trim are zeros.
            for i in 1:k, j in 1:n
                next[i, j] = i > j ? zero(T) : _h1_shift_integer(candidate[i, j], shift)
            end
            with_precision(factor_bits) do
                for ii in 1:bsub_rows
                    gi = k + ii
                    for jj in 1:width
                        gj = k + jj
                        next[gi, gj] = gi > gj ? zero(T) :
                            round(T, ldexp(factors[ii, jj], -shift))
                    end
                end
            end

            next_B2 = Matrix{T}(B2_sim)
            with_precision(factor_bits) do
                for j in 1:q, i in 1:m
                    if start <= i <= start + bsub_rows - 1
                        ii = i - k
                        next_B2[i, j] = round(T,
                            ldexp(factors[ii, width + j], -shift))
                    else
                        next_B2[i, j] = _h1_shift_integer(B2_sim[i, j], shift)
                    end
                end
            end
            working = next
            B2_sim = next_B2
            telemetry === nothing || (telemetry.h1_time_compress += _tock(compress_started))

        else
            telemetry === nothing || begin
                telemetry.h1_all_steps += 1
                telemetry.h1_phase3_calls += 1
            end
            B_sub = Matrix{T}(view(working, 1:n, 1:n))
            child_B2 = Matrix{T}(view(B2_sim, 1:n, 1:q))
            U_sub = Matrix{T}(undef, n, n)
            child_U2 = Matrix{T}(undef, n, q)
            info = _h1_reference_child!(B_sub, child_B2, U_sub, child_U2;
                goal = inherited, child_split = child, phase3 = true,
                max_iterations = max_iterations, aggressive = aggressive,
                schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
                blocksize = blocksize, panelsize = panelsize, validate = validate,
                telemetry = telemetry, depth = depth + 1)
            step_U = U_sub
            profile = Float64.(info.profile)

            if q > 0
                # update_all_representation: U2 += (U_L U_R) * U2_all.
                prior = _h2_collect_steps(step_transforms, step_bounds, n)
                composed = strassen(prior, child_U2)
                for j in 1:q, i in 1:n
                    external_U2[i, j] += composed[i, j]
                end
            end
        end

        push!(step_transforms, step_U)
        push!(step_bounds, (start, finish))
        split_advance!(split)
        iterations += 1
        telemetry === nothing || (telemetry.iterations += 1)
    end

    collect_started = _tick()
    total = _h2_collect_steps(step_transforms, step_bounds, n)
    telemetry === nothing || (telemetry.h1_time_collect += _tock(collect_started))

    apply_started = _tick()
    reduced = _reduction_mul(original, total)
    for j in 1:n, i in 1:m
        B[i, j] = reduced[i, j]
    end
    for j in 1:n, i in 1:n
        U[i, j] = total[i, j]
    end
    if q > 0
        added = _reduction_mul(original, external_U2)
        for j in 1:q, i in 1:m
            B2[i, j] = original_B2[i, j] + added[i, j]
        end
        for j in 1:q, i in 1:n
            U2[i, j] = external_U2[i, j]
        end
    end
    telemetry === nothing || (telemetry.h1_time_apply += _tock(apply_started))

    # Heuristic1 inherits Heuristic2::fini_solver, not RecursiveGeneric's
    # final-size-reduction path.  Restore only the scalar compression offsets.
    true_profile = profile .+ total_shift
    if !all(isfinite, true_profile)
        # Only reachable through this port's artificial iteration cap. Upstream
        # has no such cap; a complete left/right/all cycle has a finite profile.
        true_profile = _basis_profile(B, aggressive)
    end
    goal_met = goal_check(goal, true_profile)

    if validate
        _reduction_mul(original, total) == B || throw(ErrorException(
            "Heuristic1 returned a basis inconsistent with its accumulated transform"))
        _exact_determinant(total) in (one(T), -one(T)) || throw(ErrorException(
            "Heuristic1 returned a non-unimodular transform"))
        if q > 0
            expected_B2 = original_B2 + _reduction_mul(original, external_U2)
            expected_B2 == B2 || throw(ErrorException(
                "Heuristic1 returned B2 inconsistent with U2"))
        end
    end

    return B, U, B2, U2,
           (iterations = iterations, goal_met = goal_met,
            stopped = stopped, profile = true_profile)
end

"""
    heuristic1_reduce!(B, U; log_cond, kwargs...) -> (B, U, info)

Reduce a full-column-rank `m x n` integer basis (`m >= n`) with flatter's
phase-1 Heuristic1.  `log_cond` is a positive upper estimate for
`log2(cond(R))`; this is the known-condition route in flatter's dispatcher.

Heuristic1 deliberately has no iteration-zero goal exit.  Its phase-2 split is
walked left/right/all (left/all for dimension three), partial children remain in
phase 1, and the all child is handed to phase 3.
"""
function heuristic1_reduce!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
                            log_cond::Real,
                            profile::Union{Nothing, AbstractVector{<:Real}} = nothing,
                            profile_offset::Union{Nothing, AbstractVector{<:Real}} = nothing,
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
    m, n = size(B)
    m >= n || throw(DimensionMismatch("expected at least as many rows as columns, got $m x $n"))
    n >= 1 || throw(ArgumentError("B must be non-empty"))
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    isfinite(log_cond) && log_cond > 0 || throw(ArgumentError("log_cond must be positive and finite"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be at least one"))
    base_cutoff >= 2 || throw(ArgumentError("base_cutoff must be at least two"))

    resolved_goal = goal === nothing ? goal_from_rhf(n, rhf) : goal
    resolved_goal.n == n || throw(ArgumentError(
        "the goal has dimension $(resolved_goal.n) but the basis has $n columns"))
    split = _split === nothing ? SplitPhase2(n) : _split
    split isa SplitPhase2 || throw(ArgumentError(
        "Heuristic1 needs a SplitPhase2 tree, got $(typeof(split))"))
    split.n == n || throw(ArgumentError(
        "the phase-2 split has dimension $(split.n) but the basis has $n columns"))

    # Faithful to `Lattice(B)`: `Profile(n)` is valid but its entries are NaN.
    # Heuristic1 copies those entries verbatim; the first left QR update creates
    # the first finite profile.  Precomputing a QR profile here changes the first
    # child-goal comparison and is therefore observably different from flatter.
    initial = profile === nothing ? _h1_initial_profile(n) : Float64.(profile)
    length(initial) == n || throw(DimensionMismatch("profile must have $n entries"))
    offsets = profile_offset === nothing ? zeros(Float64, n) : Float64.(profile_offset)
    length(offsets) == n || throw(DimensionMismatch("profile_offset must have $n entries"))

    B2 = zeros(T, m, 0)
    U2 = zeros(T, n, 0)
    B, U, _, _, info = _heuristic1_reduce!(B, U, B2, U2;
        goal = resolved_goal, split = split, log_cond = Float64(log_cond),
        initial_profile = initial, profile_offset = offsets,
        max_iterations = Int(max_iterations), aggressive = aggressive,
        schoenhage_threshold = Int(schoenhage_threshold), base_cutoff = Int(base_cutoff),
        blocksize = Int(blocksize), panelsize = Int(panelsize), validate = validate,
        telemetry = telemetry, depth = Int(_depth))
    return B, U, info
end

"Non-mutating form of [`heuristic1_reduce!`](@ref)."
function heuristic1_reduce(B::AbstractMatrix{T}; kwargs...) where {T<:Integer}
    reduced = Matrix{T}(B)
    transform = Matrix{T}(undef, size(B, 2), size(B, 2))
    return heuristic1_reduce!(reduced, transform; kwargs...)
end
