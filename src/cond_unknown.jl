# cond_unknown.jl
#
# flatter's phase-1 condition-discovery algorithm.
#
# A dense basis arrives here when Irregular cannot orient it triangular and no
# useful condition-number bound is known.  CondUnknown does not try to estimate
# a pessimistic condition number from the entry sizes.  Instead it discovers as
# much of the rank as the current floating precision can resolve, reduces that
# well-conditioned prefix through phase 2, applies the resulting unimodular
# transforms to the ORIGINAL exact basis, and retries at higher precision until
# every remaining column has become exactly zero.
#
# This follows `LatticeReductionImpl::CondUnknown` in
# `src/problems/lattice_reduction/cond_unknown.cpp`:
#
#   extract_similar -> size/relative size reduction -> phase 2 -> exact apply
#       -> if unresolved: sort exact columns by length -> increase precision
#
# The C++ implementation mutates its private exact basis but does not retain a
# global transform through this outer loop.  This package's public contract is
# stronger (`B_out == B_in * U`), so the same permutations and block transforms
# are also applied to the accumulated `U` below.  That changes no reduction
# decision and makes rank-deficient dense input usable through the public API.

const COND_UNKNOWN_INITIAL_PRECISION = 53
const COND_UNKNOWN_MAX_REFINEMENTS = 10_000

"Half the base-two logarithm of a nonnegative squared norm."
@inline _cond_log_norm_from_square(x::BigFloat) =
    iszero(x) ? -Inf : Float64(log2(x)) / 2

"Apply one packed reflector stored in `factors[:, reflector]` to another matrix column."
function _cond_apply_reflector_column!(target::Matrix{BigFloat}, column::Int,
                                       factors::Matrix{BigFloat}, reflector::Int,
                                       tau::BigFloat, m::Int,
                                       ws::Vector{BigFloat})
    iszero(tau) && return nothing

    inner = ws[_WS_LARF_INNER]
    product = ws[_WS_LARF_PRODUCT]
    _mpfr_set!(inner, target[reflector, column])
    @inbounds for i in (reflector + 1):m
        _mpfr_fma!(inner, factors[i, reflector], target[i, column], inner)
    end
    _mpfr_mul!(inner, inner, tau)

    _mpfr_sub!(target[reflector, column], target[reflector, column], inner)
    @inbounds for i in (reflector + 1):m
        _mpfr_mul!(product, inner, factors[i, reflector])
        _mpfr_sub!(target[i, column], target[i, column], product)
    end
    return nothing
end

"A permutation matrix's source column for every destination column."
function _cond_permutation_sources(P::AbstractMatrix{T}) where {T<:Integer}
    n = size(P, 1)
    size(P, 2) == n || throw(DimensionMismatch("permutation matrix must be square"))
    sources = Vector{Int}(undef, n)
    seen = falses(n)
    for destination in 1:n
        source = 0
        for i in 1:n
            iszero(P[i, destination]) && continue
            P[i, destination] == one(T) || throw(ArgumentError(
                "CondUnknown permutation contains a non-unit entry"))
            source == 0 || throw(ArgumentError(
                "CondUnknown permutation contains two sources for one column"))
            source = i
        end
        source != 0 || throw(ArgumentError(
            "CondUnknown permutation contains an empty destination column"))
        seen[source] && throw(ArgumentError(
            "CondUnknown permutation uses source column $source twice"))
        seen[source] = true
        sources[destination] = source
    end
    return sources
end

"Apply a column permutation to both the exact basis and its accumulated transform."
function _cond_apply_perm!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                           P::AbstractMatrix{T}) where {T<:Integer}
    sources = _cond_permutation_sources(P)
    old_B = Matrix{T}(B)
    old_U = Matrix{T}(U)
    for destination in 1:length(sources)
        source = sources[destination]
        for i in axes(B, 1)
            B[i, destination] = old_B[i, source]
        end
        for i in axes(U, 1)
            U[i, destination] = old_U[i, source]
        end
    end
    return B, U
end

"Apply [[U1,U2],[0,I]] to an exact basis and to its accumulated transform."
function _cond_apply_block_transform!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                                      U1::AbstractMatrix{T},
                                      U2::AbstractMatrix{T}) where {T<:Integer}
    k = size(U1, 1)
    size(U1, 2) == k || throw(DimensionMismatch("U1 must be square"))
    n = size(B, 2)
    size(U2) == (k, n - k) || throw(DimensionMismatch(
        "U2 must be $k x $(n-k), got $(size(U2))"))

    old_left_B = Matrix{T}(view(B, :, 1:k))
    old_left_U = Matrix{T}(view(U, :, 1:k))

    if k < n
        add_B = _reduction_mul(old_left_B, Matrix{T}(U2))
        add_U = _reduction_mul(old_left_U, Matrix{T}(U2))
        @inbounds for jj in 1:(n - k), i in axes(B, 1)
            B[i, k + jj] += add_B[i, jj]
        end
        @inbounds for jj in 1:(n - k), i in axes(U, 1)
            U[i, k + jj] += add_U[i, jj]
        end
    end

    new_left_B = _reduction_mul(old_left_B, Matrix{T}(U1))
    new_left_U = _reduction_mul(old_left_U, Matrix{T}(U1))
    @inbounds for j in 1:k, i in axes(B, 1)
        B[i, j] = new_left_B[i, j]
    end
    @inbounds for j in 1:k, i in axes(U, 1)
        U[i, j] = new_left_U[i, j]
    end
    return B, U
end

"Exact norm-sorting permutation used between failed CondUnknown refinements."
function _cond_sort_permutation(B::AbstractMatrix{T}) where {T<:Integer}
    n = size(B, 2)
    logs = Vector{Float64}(undef, n)
    for j in 1:n
        squared = BigInt(0)
        for i in axes(B, 1)
            value = BigInt(B[i, j])
            squared += value * value
        end
        logs[j] = iszero(squared) ? -Inf : _log2_abs(squared) / 2
    end

    P = zeros(T, n, n)
    first_nonzero = 1
    last_zero = n
    for _ in 1:n
        # `findmin` returns the first minimum, matching std::min_element.
        _, source = findmin(logs)
        if logs[source] == -Inf
            P[source, last_zero] = one(T)
            last_zero -= 1
        else
            P[source, first_nonzero] = one(T)
            first_nonzero += 1
        end
        logs[source] = Inf
    end
    return P
end

function _cond_sort_by_size!(B::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T<:Integer}
    P = _cond_sort_permutation(B)
    _cond_apply_perm!(B, U, P)
    return P
end

"""
    _cond_extract_similar(B, precision)

Port of flatter's `CondUnknown::extract_similar`.

At the requested precision, process columns from left to right.  A column is
accepted when its total norm and its component orthogonal to the already
accepted columns can be distinguished with the safety rule

    2 * spread + 40 <= precision.

Accepted columns are Householdered into a triangular prefix; unresolved columns
are packed at the right.  Returns the integer scaled surrogate, the exact
permutation, the number of accepted independent columns, the common binary
shift, and the observed spread.
"""
function _cond_extract_similar(B::AbstractMatrix{T}, precision::Integer) where {T<:Integer}
    bits = Int(precision)
    bits >= 2 || throw(ArgumentError("CondUnknown precision must be at least two bits"))
    m, n = size(B)
    max_rank = min(m, n)

    return with_precision(bits) do
        R_unsorted = Matrix{BigFloat}(undef, m, n)
        R = Matrix{BigFloat}(undef, m, n)
        for j in 1:n, i in 1:m
            R_unsorted[i, j] = BigFloat(B[i, j])
            R[i, j] = BigFloat(0)
        end

        P = Matrix{T}(undef, n, n)
        _set_identity!(P)
        ws = _fused_temporaries(BigFloat, _FUSED_WORKSPACE_SIZE)

        num_valid = 0
        num_dependent = 0
        curmax = -Inf
        curmin = Inf

        for j in 1:n
            vec_len = BigFloat(0)
            orth_len = BigFloat(0)
            for i in 1:m
                value = R_unsorted[i, j]
                vec_len += value * value
                i > num_valid && (orth_len += value * value)
            end

            log_vec_len = _cond_log_norm_from_square(vec_len)
            log_orth_len = _cond_log_norm_from_square(orth_len)
            new_spread = max(curmax, log_vec_len) - min(curmin, log_orth_len)
            required = isfinite(new_spread) ? trunc(Int, 2 * new_spread + 40) : typemax(Int)

            if isfinite(new_spread) && required <= bits && num_valid < max_rank
                destination = num_valid + 1
                for i in 1:m
                    R[i, destination] = BigFloat(R_unsorted[i, j])
                end
                P[j, j] = zero(T)
                P[j, destination] = one(T)

                tau = _fused_larfg!(R, destination, m, ws)
                if j < n
                    for target in (j + 1):n
                        _cond_apply_reflector_column!(
                            R_unsorted, target, R, destination, tau, m, ws)
                    end
                end

                num_valid += 1
                curmax = max(curmax, log_vec_len)
                curmin = min(curmin, log_orth_len)
            else
                destination = n - num_dependent
                for i in 1:m
                    R[i, destination] = BigFloat(R_unsorted[i, j])
                end
                P[j, j] = zero(T)
                P[j, destination] = one(T)
                num_dependent += 1
            end
        end

        # The all-zero basis is a useful degenerate case.  Upstream reaches this
        # code with curmax=-Inf, for which converting the shift to an integer is
        # undefined; treating its scale and spread as zero is the unique benign
        # extension and leaves every reduction decision unchanged.
        shift_amount = num_valid == 0 ? 0 : bits - trunc(Int, curmax)
        spread = num_valid == 0 ? 0.0 : curmax - curmin

        B_sim = zeros(T, m, n)
        for j in 1:n
            for i in 1:min(j, m)
                scaled = ldexp(R[i, j], shift_amount)
                B_sim[i, j] = round(T, scaled)
            end
        end

        return B_sim, P, num_valid, shift_amount, spread
    end
end


"MPZ precision used by upstream Generic relative size reduction (whole GMP limbs)."
function _cond_mpz_precision(B::AbstractMatrix{<:Integer})
    widest = max(1, _h2_widest_bits(B))
    limb_bits = Sys.WORD_SIZE
    return limb_bits * cld(widest, limb_bits)
end

"Port of RelativeSizeReductionImpl::Generic for CondUnknown's surrogate B2 step."
function _cond_generic_relative_sr!(B1::AbstractMatrix{T}, B2::AbstractMatrix{T},
                                    U::AbstractMatrix{T}) where {T<:Integer}
    width2 = size(B2, 2)
    width2 == 0 && return B2, U

    # Upstream Generic chooses the MPFR precision from the MPZ matrix's
    # allocated limb width, copies B1 exactly to MPFR, QR-factorises it, then
    # dispatches to the orthogonal relative-size-reduction kernel.  In
    # particular, CondUnknown does NOT set `is_B1_upper_triangular`, even
    # though its surrogate prefix is triangular.
    bits = _cond_mpz_precision(B1)
    with_precision(bits) do
        factors = Matrix{BigFloat}(undef, size(B1)...)
        for j in axes(B1, 2), i in axes(B1, 1)
            factors[i, j] = BigFloat(B1[i, j])
        end
        factors, tau = _heuristic3_direct_qr!(factors)
        _relative_orthogonal_reflectors!(B1, B2, U, factors, tau, nothing;
            deadband = RELATIVE_SIZE_REDUCTION_DEADBAND,
            max_passes = RELATIVE_SIZE_REDUCTION_MAX_PASSES)
    end
    return B2, U
end

"Count exact zero columns strictly to the right of the currently valid prefix."
function _cond_zero_tail_columns(B::AbstractMatrix, num_valid::Int)
    zeros_found = 0
    for j in (num_valid + 1):size(B, 2)
        all_zero = true
        for i in axes(B, 1)
            if !iszero(B[i, j])
                all_zero = false
                break
            end
        end
        all_zero && (zeros_found += 1)
    end
    return zeros_found
end

"One CondUnknown precision/refinement iteration."
function _cond_refine_basis!(B::AbstractMatrix{T}, U::AbstractMatrix{T},
                             goal::ReductionGoal, working_precision::Int,
                             max_rank::Int;
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

    started = _tick()
    B_sim, P, num_valid, shift_amount, observed_spread =
        _cond_extract_similar(B, working_precision)
    if telemetry !== nothing
        telemetry.cond_time_extract += _tock(started)
        telemetry.cond_selected_rank = num_valid
        telemetry.cond_max_precision = max(telemetry.cond_max_precision, working_precision)
    end

    profile = Float64[]
    if num_valid > 0
        B_indep = Matrix{T}(view(B_sim, 1:num_valid, 1:num_valid))
        B_dep = num_valid < n ? Matrix{T}(view(B_sim, 1:num_valid, (num_valid + 1):n)) :
                              zeros(T, num_valid, 0)

        # U2 in upstream CondUnknown: first size-reduce the independent
        # triangular surrogate, then relatively size-reduce the unresolved
        # block and express that off-diagonal transform in ORIGINAL independent
        # coordinates as U2i * Urel.
        U2i = Matrix{T}(undef, num_valid, num_valid)
        started = _tick()
        size_reduction_triu!(B_indep, U2i)
        telemetry === nothing || (telemetry.cond_time_size_reduce += _tock(started))

        U2d = zeros(T, num_valid, n - num_valid)
        if num_valid < n
            Urel = zeros(T, num_valid, n - num_valid)
            started = _tick()
            _cond_generic_relative_sr!(B_indep, B_dep, Urel)
            telemetry === nothing || (telemetry.cond_time_relative += _tock(started))
            U2d = _reduction_mul(U2i, Urel)
        end

        # U3 in upstream CondUnknown: the phase-2 reduction is wrapped in
        # LatRedRelSR whenever a dependent block is present.  Reuse the H2
        # wrapper port directly so B_dep and U3d follow the same algebra.
        U3i = Matrix{T}(undef, num_valid, num_valid)
        U3d = zeros(T, num_valid, n - num_valid)
        phase_goal = goal_from_rhf(num_valid, goal_rhf(goal))
        started = _tick()
        info = _heuristic2_relative_child!(B_indep, B_dep, U3i, U3d;
            goal = phase_goal, split = SplitPhase2(num_valid),
            max_iterations = max_iterations, aggressive = aggressive,
            schoenhage_threshold = schoenhage_threshold, base_cutoff = base_cutoff,
            blocksize = blocksize, panelsize = panelsize,
            validate = validate, telemetry = telemetry, depth = depth + 1)
        telemetry === nothing || (telemetry.cond_time_h2 += _tock(started))

        profile = isempty(info.profile) ?
                  [_log2_abs(B_indep[i, i]) - shift_amount for i in 1:num_valid] :
                  Float64[p - shift_amount for p in info.profile]

        # Apply P, U2 and U3 to the TRUE exact basis.  Apply the same operations
        # to accumulated U to preserve B == B_original * U, a contract this
        # Julia package exposes publicly.
        started = _tick()
        _cond_apply_perm!(B, U, P)
        _cond_apply_block_transform!(B, U, U2i, U2d)
        _cond_apply_block_transform!(B, U, U3i, U3d)
        telemetry === nothing || (telemetry.cond_time_apply += _tock(started))
    else
        started = _tick()
        _cond_apply_perm!(B, U, P)
        telemetry === nothing || (telemetry.cond_time_apply += _tock(started))
    end

    zero_columns = _cond_zero_tail_columns(B, num_valid)
    if num_valid + zero_columns == n
        return true, working_precision, max_rank, profile, num_valid, observed_spread
    end

    if num_valid == max_rank
        # Upstream switches from blind doubling to the spread-derived precision
        # once the entire currently possible rank has been resolved.
        spread = isempty(profile) ? observed_spread : profile_spread(profile)
        next_precision = max(2, trunc(Int, 2 * spread + 40))
        return false, next_precision, max_rank, profile, num_valid, observed_spread
    end

    next_max_rank = min(size(B, 1), n - zero_columns)
    return false, 2 * working_precision, next_max_rank, profile, num_valid, observed_spread
end

"""
    cond_unknown_reduce!(B, U; kwargs...) -> (B, U, info)

Faithful port of flatter's `CondUnknown`, used for non-triangular input whose
condition number is not known.

The exact basis is repeatedly approximated at increasing BigFloat precision.
Only columns whose orthogonal component is resolved at that precision are put
through Phase 2; the resulting transforms are applied to the exact basis.  The
loop terminates when every unresolved column has become exactly zero, so
rank-deficient bases are handled rather than rejected.
"""
function cond_unknown_reduce!(B::AbstractMatrix{T}, U::AbstractMatrix{T};
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
                              initial_precision::Integer = COND_UNKNOWN_INITIAL_PRECISION,
                              max_refinements::Integer = COND_UNKNOWN_MAX_REFINEMENTS,
                              _depth::Integer = 0) where {T<:Integer}
    m, n = size(B)
    m >= n || throw(DimensionMismatch(
        "CondUnknown currently expects at least as many rows as columns, got $m x $n"))
    size(U) == (n, n) || throw(DimensionMismatch("U must be $n x $n, got $(size(U))"))
    n >= 1 || throw(ArgumentError("B must have at least one column"))
    initial_precision >= 2 || throw(ArgumentError("initial_precision must be at least two"))
    max_refinements >= 1 || throw(ArgumentError("max_refinements must be at least one"))

    resolved_goal = goal === nothing ? goal_from_rhf(n, rhf) : goal
    resolved_goal.n == n || throw(ArgumentError(
        "the goal has dimension $(resolved_goal.n) but the basis has $n columns"))

    original = validate ? Matrix{T}(B) : nothing
    _set_identity!(U)
    telemetry === nothing || (telemetry.cond_calls += 1)

    working_precision = Int(initial_precision)
    max_rank = min(m, n)
    profile = Float64[]
    rank = 0
    observed_spread = 0.0

    for refinement in 1:Int(max_refinements)
        telemetry === nothing || (telemetry.cond_refinements += 1)
        done, next_precision, next_max_rank, current_profile, current_rank, spread =
            _cond_refine_basis!(B, U, resolved_goal, working_precision, max_rank;
                max_iterations = Int(max_iterations), aggressive = aggressive,
                schoenhage_threshold = Int(schoenhage_threshold),
                base_cutoff = Int(base_cutoff), blocksize = Int(blocksize),
                panelsize = Int(panelsize), validate = validate,
                telemetry = telemetry, depth = Int(_depth))

        profile = current_profile
        rank = current_rank
        observed_spread = spread
        if done
            if validate
                expected = _reduction_mul(original::Matrix{T}, Matrix{T}(U))
                expected == B || throw(ErrorException(
                    "CondUnknown returned a basis inconsistent with its accumulated transform"))
                _exact_determinant(U) in (one(T), -one(T)) || throw(ErrorException(
                    "CondUnknown returned a non-unimodular transform"))
            end
            rank_goal_met = rank == 0 || goal_check(
                goal_from_rhf(rank, goal_rhf(resolved_goal)), profile)
            return B, U, (iterations = refinement, goal_met = rank_goal_met,
                          stopped = rank_goal_met ? :goal : :cap, profile = profile,
                          rank = rank, working_precision = working_precision,
                          observed_spread = observed_spread, condition_resolved = true)
        end

        if next_precision != working_precision && telemetry !== nothing
            telemetry.cond_precision_changes += 1
        end
        working_precision = next_precision
        max_rank = next_max_rank

        started = _tick()
        _cond_sort_by_size!(B, U)
        telemetry === nothing || (telemetry.cond_time_sort += _tock(started))
    end

    throw(ErrorException(
        "CondUnknown did not resolve the basis in $(Int(max_refinements)) refinements"))
end

"Non-mutating form of [`cond_unknown_reduce!`](@ref)."
function cond_unknown_reduce(B::AbstractMatrix{T}; kwargs...) where {T<:Integer}
    reduced = Matrix{T}(B)
    transform = Matrix{T}(undef, size(B, 2), size(B, 2))
    return cond_unknown_reduce!(reduced, transform; kwargs...)
end
