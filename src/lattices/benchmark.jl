# lattices/benchmark.jl
#
# Exploration harness for the recursive reducer. NOT a test: nothing here
# asserts, because the point is to find out how the driver behaves on real
# instances rather than to confirm what we already believe.
#
# Run from the package root with
#
#     julia --project lattices/benchmark.jl
#
# or, for a single family from the REPL,
#
#     include("lattices/benchmark.jl"); run_family("spread", [8, 16, 24])
#
# Four questions it exists to answer:
#
#   1. Does the driver terminate on real instances, at dimensions past the
#      n <= 12 the unit tests cover?
#   2. How does its reduction quality compare to fplll on the same basis?
#   3. Where does the time actually go — fused QR, matrix products,
#      compression, or the two-column base case?
#   4. Is the iteration cap ever the thing that stops it, rather than the goal?
#
# Question 3 is the one that decides what to optimise next. The guesses on the
# table are the untiled update step and recursing all the way to n = 2 instead
# of cutting over to fplll around n = 32; this measures which, if either.

using Flatter
using Random
using Printf

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

column_norm2(B, j) = sum(BigInt(B[i, j])^2 for i in 1:size(B, 1))

shortest_norm2(B) = minimum(column_norm2(B, j) for j in 1:size(B, 2))

"log2 of a non-negative big integer, without overflowing to Inf."
function log2_big(x::BigInt)
    iszero(x) && return -Inf
    bits = ndigits(abs(x); base = 2)
    bits <= 53 && return log2(Float64(abs(x)))
    return log2(Float64(abs(x) >> (bits - 53))) + (bits - 53)
end

"""
Root Hermite factor: `(||b1|| / det^(1/n))^(1/n)`, the standard measure of
reduction quality. Lower is better; around 1.02 is a good LLL result and 1.01
is strong BKZ.
"""
function root_hermite_factor(shortest2::BigInt, log2_det::Float64, n::Int)
    log2_shortest = log2_big(shortest2) / 2
    return 2.0^((log2_shortest - log2_det / n) / n)
end

"""
The Gaussian heuristic prediction for the shortest vector, in log2. What a
random lattice of this determinant and dimension should contain.
"""
gaussian_heuristic_log2(log2_det::Float64, n::Int) =
    log2_det / n + 0.5 * log2(n / (2 * pi * exp(1)))

# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

"""
    measure(bundle; kwargs...) -> NamedTuple

Reduce one basis with the driver and with fplll, and collect the comparison.

Failures are caught rather than thrown: one bad instance should not abandon the
sweep, and a timeout or a stall is itself a result worth recording.
"""
function measure(bundle; max_iterations = 120, aggressive = false,
                 algorithm::Symbol = :teaching,
                 run_fplll = true, verbose = false, repeat_under = 2.0,
                 progress = true, measure_memory = false, low_memory = nothing,
                 gc_dimension = Flatter.REDUCTION_GC_DIMENSION,
                 gc_full = false)
    B = bundle.basis
    n = size(B, 2)
    log2_det = bundle.log2_determinant

    # Repeat cheap runs and keep the FASTEST, together with the telemetry from
    # that same run: a single timing on BigInt work varies by up to 2x with GC,
    # and the time and the phase counters must come from one run or the reported
    # shares do not add up.
    telemetry = Flatter.ReductionTelemetry()
    ours = nothing
    ours_time = NaN
    ours_error = nothing
    info = nothing
    live_growth = 0
    rss_growth = 0

    # Announce each phase before entering it. A run killed by the OS leaves no
    # catchable error, so without this there is no way to tell whether our code
    # or fplll was the one that ran out of memory.
    announce(phase) = progress && (print(stderr, "    [", bundle.name, " dim ",
                                         n, "] ", phase, " ...\n"); flush(stderr))

    try
        for attempt in 1:3
            announce(attempt == 1 ? "reducing" : "reducing (repeat)")
            attempt > 1 && GC.gc()
            attempt_telemetry = Flatter.ReductionTelemetry()
            attempt_telemetry.measure_memory = measure_memory
            GC.gc()
            live_before = Base.gc_live_bytes()
            rss_before = Sys.maxrss()
            started = time_ns()
            if algorithm === :teaching
                basis, _, attempt_info = Flatter.reduce_basis(
                    B; algorithm = :teaching, max_iterations = max_iterations,
                    aggressive = aggressive, low_memory = low_memory,
                    gc_dimension = gc_dimension, gc_full = gc_full,
                    telemetry = attempt_telemetry)
            else
                # low_memory/gc_dimension/gc_full are controls for the teaching
                # recursive driver.  Heuristic2 has its own recursion and does
                # not accept those keywords; only pass the controls shared by
                # the two algorithms here.
                basis, _, attempt_info = Flatter.reduce_basis(
                    B; algorithm = :heuristic, max_iterations = max_iterations,
                    aggressive = aggressive, telemetry = attempt_telemetry)
            end
            elapsed = (time_ns() - started) / 1e9
            attempt_live = Base.gc_live_bytes() - live_before
            attempt_rss = Sys.maxrss() - rss_before

            if isnan(ours_time) || elapsed < ours_time
                ours_time = elapsed
                telemetry = attempt_telemetry
                ours = basis
                info = attempt_info
                live_growth = attempt_live
                rss_growth = attempt_rss
            end

            # Only repeat while it is cheap to do so. On an expensive instance
            # the noise matters less than the cost of measuring again.
            ours_time >= repeat_under && break
        end
    catch err
        ours_error = sprint(showerror, err)
    end

    theirs = nothing
    theirs_time = NaN
    theirs_error = nothing
    if run_fplll
        try
            announce("fplll")
            started = time_ns()
            theirs, _ = Flatter.fplll_reduce(B)
            theirs_time = (time_ns() - started) / 1e9
            if theirs_time < repeat_under
                for _ in 1:2
                    GC.gc()
                    started = time_ns()
                    Flatter.fplll_reduce(B)
                    theirs_time = min(theirs_time, (time_ns() - started) / 1e9)
                end
            end
        catch err
            theirs_error = sprint(showerror, err)
        end
    end

    ours_best = ours === nothing ? nothing : shortest_norm2(ours)
    theirs_best = theirs === nothing ? nothing : shortest_norm2(theirs)

    verbose && telemetry.levels > 0 && println(telemetry)

    return (name = bundle.name,
            dimension = n,
            log2_determinant = log2_det,
            ours_time = ours_time,
            theirs_time = theirs_time,
            ours_norm_log2 = ours_best === nothing ? NaN : log2_big(ours_best) / 2,
            theirs_norm_log2 = theirs_best === nothing ? NaN : log2_big(theirs_best) / 2,
            ours_rhf = (ours_best === nothing || isnan(log2_det)) ? NaN :
                       root_hermite_factor(ours_best, log2_det, n),
            theirs_rhf = (theirs_best === nothing || isnan(log2_det)) ? NaN :
                         root_hermite_factor(theirs_best, log2_det, n),
            gaussian_log2 = isnan(log2_det) ? NaN :
                            gaussian_heuristic_log2(log2_det, n),
            planted_log2 = bundle.planted_norm2 === nothing ? NaN :
                           log2_big(BigInt(bundle.planted_norm2)) / 2,
            found_planted = (bundle.planted_norm2 === nothing || ours_best === nothing) ?
                            missing : ours_best <= BigInt(bundle.planted_norm2),
            path = info === nothing ? :none : get(info, :path, :triangular),
            live_growth = live_growth,
            rss_growth = rss_growth,
            iterations = info === nothing ? -1 : info.iterations,
            goal_met = info === nothing ? false : info.goal_met,
            info = info,
            telemetry = telemetry,
            ours_error = ours_error,
            theirs_error = theirs_error)
end

# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

function print_header()
    @printf("%-18s %4s %10s %10s %9s %9s %8s %6s %5s %4s %5s\n",
            "family", "dim", "ours(ms)", "fplll(ms)", "log2|b1|", "fplll",
            "vs GH", "ratio", "iter", "stop", "path")
    println(repeat("-", 96))
end

function print_row(r)
    if r.ours_error !== nothing
        @printf("%-18s %4d  FAILED: %s\n", r.name, r.dimension,
                first(r.ours_error, 60))
        return
    end
    ratio = (isnan(r.theirs_time) || r.theirs_time <= 0) ? NaN :
            r.ours_time / r.theirs_time
    reason = r.info === nothing ? "err" :
             r.info.stopped === :goal ? "goal" :
             r.info.stopped === :stagnated ? "stag" :
             r.info.stopped === :base_case ? "base" : "CAP"
    marker = r.found_planted === missing ? "" :
             r.found_planted ? "  planted vector FOUND" : "  planted vector missed"
    @printf("%-18s %4d %10.2f %10.2f %9.2f %9.2f %8.2f %6.0fx %5d %4s %5s%s\n",
            r.name, r.dimension, 1000 * r.ours_time, 1000 * r.theirs_time,
            r.ours_norm_log2, r.theirs_norm_log2,
            r.ours_norm_log2 - r.gaussian_log2, ratio,
            r.iterations, reason, string(r.path), marker)
end

"""
Which of the goal's conditions refused, when a run stalls.

`goal_met = false` says only that the loop gave up; this says why. The drop
condition failing means the basis is genuinely not reduced enough, whereas the
middle-span condition failing means the drop is concentrated across the halves'
boundary -- a different problem needing a different fix.
"""
function print_goal_diagnosis(r)
    r.info === nothing && return
    goal = Flatter.goal_from_rhf(r.dimension, 1.02)
    report = Flatter.goal_report(goal, r.info.profile)
    @printf("    goal: drop %.2f / %.2f %s   halves %.2f / %.2f %s   middle %.2f / %.2f %s\n",
            report.drop, report.max_drop, report.drop_ok ? "ok" : "FAIL",
            report.mean_gap, report.mu_sep, report.separation_ok ? "ok" : "FAIL",
            report.mid_drop, report.mid_budget, report.middle_ok ? "ok" : "FAIL")
end

"H2-local kernel time which is disjoint from the generic driver stage timers."
function heuristic2_local_time(t)
    return t.h2_time_qr + t.h2_time_wrapper_qr + t.h2_time_relative +
           t.h2_time_u2_compose + t.h2_time_top_right + t.h2_time_collect +
           t.h2_time_compress + t.h2_time_apply
end

"Where the time went, as a share of the driver's wall clock."
function print_time_breakdown(r)
    t = r.telemetry
    total = r.ours_time
    (isnan(total) || total <= 0) && return
    @printf("    levels=%d depth=%d windows=%d capped=%d base=%d (L%d/S%d/F%d) fusedQR=%d\n",
            t.levels, t.max_depth, t.iterations, t.capped, t.base_cases,
            t.lagrange_calls, t.schoenhage_calls, t.fplll_calls, t.fused_calls)
    generic_accounted = t.time_fused + t.time_matmul + t.time_compress + t.time_base +
                        t.time_finalise + t.time_dense_qr + t.time_profile +
                        t.time_gc + t.time_setup + t.time_push
    h2_local = heuristic2_local_time(t)
    accounted = generic_accounted + h2_local
    @printf("    fusedQR %5.1f%%  matmul %5.1f%%  compress %5.1f%%  base %5.1f%%  finalise %5.1f%%\n",
            100 * t.time_fused / total, 100 * t.time_matmul / total,
            100 * t.time_compress / total, 100 * t.time_base / total,
            100 * t.time_finalise / total)
    if t.time_fused > 0.10 * total
        # The four things this factorisation does respond very differently to
        # blocking: it replaces the trailing update entirely, could be extended
        # to the re-orthogonalisation, and does nothing for the other two.
        @printf("      fusedQR = size-red %4.1f%%  reorth %4.1f%%  reflectors %4.1f%%  trailing %4.1f%%\n",
                100 * t.fused_split[Flatter.FUSED_TIME_REDUCE] / total,
                100 * t.fused_split[Flatter.FUSED_TIME_REORTH] / total,
                100 * t.fused_split[Flatter.FUSED_TIME_REFLECTOR] / total,
                100 * t.fused_split[Flatter.FUSED_TIME_TRAILING] / total)
    end
    if t.profile_calls > 0 && t.time_profile > 0.01 * total
        # Reporting `info.profile` needs a factorisation of the reduced basis.
        # fplll computes no such thing, so this is work the comparison charges
        # to us and not to it; `want_profile = false` skips it.
        @printf("    profile report: %5.1f%% of total (%d factorisation%s)\n",
                100 * t.time_profile / total, t.profile_calls,
                t.profile_calls == 1 ? "" : "s")
    end
    if t.dense_rounds > 0
        # The dense path's own factorisation, paid once per round and outside
        # the driver entirely.
        @printf("    dense path: %d rounds, QR %5.1f%% of total\n",
                t.dense_rounds, 100 * t.time_dense_qr / total)
    end
    # H2's local representation/update kernels are disjoint from the generic
    # recursive-driver timers above.  Include them before calling anything
    # residual; H3 update time is already inside `time_fused` and must not be
    # added again.
    if t.h2_calls > 0
        @printf("    H2 local %5.1f%%  accounted %5.1f%% = generic %5.1f%% + H2-local %5.1f%%; residual %5.1f%%\n",
                100 * h2_local / total, 100 * accounted / total,
                100 * generic_accounted / total, 100 * h2_local / total,
                100 * max(0.0, 1 - accounted / total))
    elseif accounted < 0.75 * total
        # Anything unaccounted is worth seeing: it has hidden a dominant cost twice.
        @printf("    (%.0f%% of the time is unaccounted for)\n",
                100 * (1 - accounted / total))
    end
    # When finalising dominates, break it down: lifting the transforms,
    # applying them to the original basis, and the final size reduction are
    # three quite different costs with three quite different remedies.
    if t.time_finalise > 0.25 * total
        # `fold-in` happens in the loop and `combine` in the finalise block, so
        # only the latter is part of `finalise`; they used to share a counter.
        @printf("      finalise = combine %5.1f%%  apply %5.1f%%  final-SR %5.1f%%  (fold-in %5.1f%% in loop)\n",
                100 * t.time_collect / total, 100 * t.time_apply / total,
                100 * t.time_final_sr / total, 100 * t.time_push / total)
    end
    # Live bytes are what the collector still considers reachable; resident is
    # what the process actually holds. A large gap between them is collection
    # lag rather than a genuine memory requirement, and is worth knowing about
    # before concluding that an instance is too big to run.
    if (r.rss_growth > 64 * 1024 * 1024 || r.live_growth > 64 * 1024 * 1024)
        # Growth figures only. `telemetry.peak_live` is an absolute reading, so
        # in a shared session it reports whatever earlier runs left live and
        # says nothing about this one; `lattices/memory_scaling.jl` measures in
        # a fresh process for that reason.
        @printf("    memory: live %+.0f MB   resident %+.0f MB\n",
                r.live_growth / 1024^2, r.rss_growth / 1024^2)
    end
    if t.capped > 0
        @printf("    WARNING: %d of %d levels stopped at the iteration cap, not the goal\n",
                t.capped, t.levels)
    end
end

# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

"""
    warm_up()

Compile the reduction paths on a tiny instance, so that no measurement carries
the cost of compiling them.

`measure` keeps the fastest of three attempts, which hides the compiling run for
anything quick — but it stops after one attempt once a run exceeds
`repeat_under`, so the expensive cases get a single timing. Those are protected
only by some earlier case happening to compile the same path first, which is
luck. The base triangular, recursive triangular and dense entry paths are all
exercised here, since they compile separately.
"""
function warm_up(; algorithm::Symbol = :teaching)
    algorithm in (:teaching, :heuristic) || throw(ArgumentError(
        "unknown reduction algorithm $algorithm; expected :teaching or :heuristic"))
    rng = MersenneTwister(7)

    if algorithm === :teaching
        # Base triangular path.
        Flatter.reduce_basis(Flatter.random_triangular_lattice(rng, 24).basis;
                             algorithm = :teaching, want_profile = false)

        # Recursive triangular path.
        Flatter.reduce_basis(Flatter.random_triangular_lattice(rng, 48).basis;
                             algorithm = :teaching, want_profile = false)

        # Dense entry path, which compiles independently of the triangular driver.
        Flatter.reduce_basis(Flatter.scrambled_lattice(rng, 16).basis;
                             algorithm = :teaching, want_profile = false)
    else
        # An easy triangular basis can satisfy H2's goal at entry and therefore
        # fail to compile the left/right/all cycle.  Knapsack reliably exercises
        # reorientation, recursive H2, LatRedRelSR and the H2 -> H3 handoff.
        Flatter.reduce_basis(Flatter.knapsack_lattice(rng, 48).basis;
                             algorithm = :heuristic, want_profile = false)
        # Also compile the already-upper-triangular entry branch.
        Flatter.reduce_basis(Flatter.random_triangular_lattice(rng, 48).basis;
                             algorithm = :heuristic, want_profile = false)
    end
    return nothing
end
"""
    run_family(name, sizes; seed=1, kwargs...) -> Vector

Sweep one lattice family over a list of size parameters.

`sizes` defaults to the range the family declares for itself, which differs
between families because they do not cost the same at a given dimension. Pass a
list to override, mostly useful for a quick pass at small dimensions.

The size parameter is not always the dimension: knapsack produces `n+1` columns,
q-ary `2n`, and relation `n+1` columns in `n+2` rows.
"""
function run_family(name::AbstractString, sizes = nothing; seed::Integer = 1,
                    algorithm::Symbol = :teaching,
                    breakdown::Bool = true, time_budget::Real = 60.0,
                    warm::Bool = true, kwargs...)
    families = Flatter.lattice_families()
    index = findfirst(f -> f.name == name, families)
    index === nothing && error("unknown family $name; have " *
                               join((f.name for f in families), ", "))
    family = families[index]
    sizes = sizes === nothing ? family.sizes : sizes
    warm && warm_up(; algorithm = algorithm)

    println("\n=== $(family.name) [$(algorithm)] ===")
    print_header()
    results = []
    for n in sizes
        rng = MersenneTwister(hash((seed, name, n)))
        bundle = family.generate(rng, n)
        r = measure(bundle; algorithm = algorithm, kwargs...)
        print_row(r)
        if breakdown && r.ours_error === nothing
            print_time_breakdown(r)
            if r.info !== nothing && r.info.stopped in (:stagnated, :cap)
                print_goal_diagnosis(r)
            end
        end
        push!(results, r)

        # Stop climbing BEFORE the expensive one, not after it. Cost grows
        # steeply -- roughly an order of magnitude per size step here -- so a
        # guard that only reacts once the budget is already exceeded reacts one
        # size too late, and that next size is the one that gets the process
        # killed rather than merely being slow.
        if !isnan(r.ours_time) && r.ours_time * 8 > time_budget
            @printf("    (stopping this family: %.1fs now, so the next size is projected past the %.0fs budget)\n",
                    r.ours_time, time_budget)
            break
        end
    end
    return results
end

"""
    run_all(; sizes=..., seed=1, kwargs...) -> Vector

Every family over its own declared range, smallest first so a stall shows up
before much time has been spent. Pass `sizes` to override every family at once.
"""
function run_all(; sizes = nothing, seed::Integer = 1,
                 algorithm::Symbol = :teaching, kwargs...)
    warm_up(; algorithm = algorithm)
    results = []
    for family in Flatter.lattice_families()
        # Each family carries the range that suits it; `sizes` overrides them
        # all, which is mostly useful for a quick pass at small dimensions.
        append!(results, run_family(family.name, sizes; seed = seed,
                                    algorithm = algorithm, warm = false, kwargs...))
    end

    println("\n=== summary ===")
    stalled = [r for r in results if r.ours_error === nothing && !r.goal_met]
    capped = [r for r in results
              if r.ours_error === nothing && r.telemetry.capped > 0]
    failed = [r for r in results if r.ours_error !== nothing]
    println("runs: ", length(results),
            "   failed: ", length(failed),
            "   top level stalled: ", length(stalled),
            "   with capped sublevels: ", length(capped))

    stagnant = [r for r in results
                if r.ours_error === nothing && r.telemetry.stagnated > 0]
    if !isempty(stagnant)
        println("\nRuns where a level converged short of its goal (stopped: stag).")
        println("Not a failure -- the profile stopped moving, so more iterations would not help.")
        println("Check the goal line: a near-miss means the target was marginally out of reach.")
        for r in stagnant
            @printf("  %-18s dim %3d : %d of %d levels\n",
                    r.name, r.dimension, r.telemetry.stagnated, r.telemetry.levels)
        end
    end

    if !isempty(capped)
        println("\nRuns where sublevels hit the iteration cap. These stall silently:")
        println("the top-level `goal` column says yes while the recursion beneath it gives up):")
        for r in capped
            @printf("  %-18s dim %3d : %d of %d levels capped\n",
                    r.name, r.dimension, r.telemetry.capped, r.telemetry.levels)
        end
    end

    if !isempty(failed)
        println("\nfailures:")
        for r in failed
            println("  ", r.name, " dim ", r.dimension, ": ", first(r.ours_error, 100))
        end
    end

    slower = [r for r in results
              if r.ours_error === nothing && !isnan(r.theirs_time) && r.theirs_time > 0]
    if !isempty(slower)
        ratios = [r.ours_time / r.theirs_time for r in slower]
        @printf("\ntime vs fplll: median %.1fx, worst %.1fx\n",
                sort(ratios)[cld(length(ratios), 2)], maximum(ratios))
    end

    quality = [r for r in results if r.ours_error === nothing && !isnan(r.ours_rhf)]
    if !isempty(quality)
        gaps = [r.ours_norm_log2 - r.theirs_norm_log2 for r in quality
                if !isnan(r.theirs_norm_log2)]
        if !isempty(gaps)
            # Note: @printf needs a literal format string, so this line stays
            # long rather than being concatenated with `*`.
            @printf("quality vs fplll: median %+.2f bits on log2||b1|| (negative means we found a shorter vector)\n",
                    sort(gaps)[cld(length(gaps), 2)])
        end
    end

    return results
end

# Running the file directly does the default sweep. Small sizes first: if the
# driver is going to stall it should do so cheaply.

const HEURISTIC2_STANDARD_FAMILIES =
    ("random-triangular", "spread", "knapsack", "q-ary")

"""
    run_heuristic2_standard(; sizes=nothing, seed=1, kwargs...) -> Vector

Run the ordinary benchmark harness through the currently ported heuristic
dispatcher, but only on families whose input can already enter Phase 2 faithfully.

The supported set is `random-triangular`, `spread`, `knapsack` (after automatic
reorientation), and `q-ary`.  `scrambled` and `ideal` need `CondUnknown`;
`relation` is rectangular and also needs the general-input path.  Keeping this
as a separate entry point prevents an incomplete heuristic dispatcher from
quietly becoming the meaning of the normal `run_all()` benchmark.
"""
function run_heuristic2_standard(; sizes = nothing, seed::Integer = 1, kwargs...)
    warm_up(; algorithm = :heuristic)
    results = []
    for name in HEURISTIC2_STANDARD_FAMILIES
        append!(results, run_family(name, sizes; seed = seed, algorithm = :heuristic,
                                    warm = false, kwargs...))
    end
    return results
end

"""
    precision_tradeoff(; sizes, seed)

Time and quality at the two working-precision policies, on identical instances.

`lll_precision` offers `2*spread + 30 + 2n` by default and `spread + 30` under
`aggressive`. Ball-arithmetic measurement (`lattices/arb_probe.jl`) suggests the
default carries several hundred bits of headroom, but also that `aggressive` may
undershoot: at dimension 32 with a spread of 256 the two are 606 and 286 bits,
and the certifying threshold measured around 406.

MPFR cost scales with precision, so the time difference should be large. What
matters is whether the quality survives, which is why this reports both and
compares them on the same basis rather than across runs:

  * `log2|b1|` is what the reduction actually achieved. A higher figure at lower
    precision means the coordinates were too coarse to size-reduce properly.
  * `stopped` says whether the goal was met. A run that starts stagnating at
    lower precision has run out of precision, not out of progress -- exactly
    what the guard exists to catch.

A faster policy that leaves both unchanged is free; one that degrades either is
not, however good the timing looks.
"""
function precision_tradeoff(; sizes = nothing, seed::Integer = 1)
    warm_up()
    println("\nWorking precision: default against aggressive, same instances\n")
    @printf("%-18s %5s %11s %11s %7s %11s %11s %s\n",
            "family", "dim", "default(ms)", "aggr.(ms)", "speedup",
            "default|b1|", "aggr.|b1|", "verdict")
    println(repeat("-", 100))

    for family in Flatter.lattice_families()
        for n in (sizes === nothing ? family.sizes : sizes)
            bundle = family.generate(MersenneTwister(hash((seed, family.name, n))), n)

            standard = measure(bundle; aggressive = false, progress = false,
                               run_fplll = false)
            reduced = measure(bundle; aggressive = true, progress = false,
                              run_fplll = false)
            (standard.ours_error !== nothing || reduced.ours_error !== nothing) && continue
            (isnan(standard.ours_norm_log2) || isnan(reduced.ours_norm_log2)) && continue

            quality_loss = reduced.ours_norm_log2 - standard.ours_norm_log2
            stalled = reduced.info !== nothing && standard.info !== nothing &&
                      reduced.info.stopped !== standard.info.stopped

            verdict = quality_loss > 0.01 ? "WORSE by $(round(quality_loss; digits = 2)) bits" :
                      stalled ? "same length, different stop" :
                      reduced.ours_time < 0.9 * standard.ours_time ? "free speedup" :
                      "no difference"

            @printf("%-18s %5d %11.2f %11.2f %6.1fx %11.2f %11.2f %s\n",
                    bundle.name, bundle.dimension,
                    1000 * standard.ours_time, 1000 * reduced.ours_time,
                    standard.ours_time / max(reduced.ours_time, eps()),
                    standard.ours_norm_log2, reduced.ours_norm_log2, verdict)

            standard.ours_time > 20 && break     # keep the sweep bounded
        end
    end

    println("\nA policy is only worth adopting where the verdict is a free")
    println("speedup across every family: a shorter vector missed on one")
    println("instance costs more than the time saved on the others.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
