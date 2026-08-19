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
                 run_fplll = true, verbose = false)
    B = bundle.basis
    n = size(B, 2)
    log2_det = bundle.log2_determinant

    telemetry = Flatter.ReductionTelemetry()

    ours = nothing
    ours_time = NaN
    ours_error = nothing
    info = nothing
    try
        started = time_ns()
        ours, _, info = Flatter.lattice_reduce(B; max_iterations = max_iterations,
                                               aggressive = aggressive,
                                               telemetry = telemetry)
        ours_time = (time_ns() - started) / 1e9
    catch err
        ours_error = sprint(showerror, err)
    end

    theirs = nothing
    theirs_time = NaN
    theirs_error = nothing
    if run_fplll
        try
            started = time_ns()
            theirs, _ = Flatter.fplll_reduce(B)
            theirs_time = (time_ns() - started) / 1e9
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
            ours_rhf = ours_best === nothing ? NaN :
                       root_hermite_factor(ours_best, log2_det, n),
            theirs_rhf = theirs_best === nothing ? NaN :
                         root_hermite_factor(theirs_best, log2_det, n),
            gaussian_log2 = gaussian_heuristic_log2(log2_det, n),
            planted_log2 = bundle.planted_norm2 === nothing ? NaN :
                           log2_big(BigInt(bundle.planted_norm2)) / 2,
            found_planted = (bundle.planted_norm2 === nothing || ours_best === nothing) ?
                            missing : ours_best <= BigInt(bundle.planted_norm2),
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
    @printf("%-18s %4s %10s %10s %9s %9s %8s %6s %5s %4s\n",
            "family", "dim", "ours(ms)", "fplll(ms)", "log2|b1|", "fplll",
            "vs GH", "ratio", "iter", "stop")
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
    @printf("%-18s %4d %10.2f %10.2f %9.2f %9.2f %8.2f %6.0fx %5d %4s\n",
            r.name, r.dimension, 1000 * r.ours_time, 1000 * r.theirs_time,
            r.ours_norm_log2, r.theirs_norm_log2,
            r.ours_norm_log2 - r.gaussian_log2, ratio,
            r.iterations, reason)
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

"Where the time went, as a share of the driver's wall clock."
function print_time_breakdown(r)
    t = r.telemetry
    total = r.ours_time
    (isnan(total) || total <= 0) && return
    @printf("    levels=%d depth=%d windows=%d capped=%d base=%d (L%d/S%d/F%d) fusedQR=%d\n",
            t.levels, t.max_depth, t.iterations, t.capped, t.base_cases,
            t.lagrange_calls, t.schoenhage_calls, t.fplll_calls, t.fused_calls)
    @printf("    fusedQR %5.1f%%  matmul %5.1f%%  compress %5.1f%%  base %5.1f%%  finalise %5.1f%%\n",
            100 * t.time_fused / total, 100 * t.time_matmul / total,
            100 * t.time_compress / total, 100 * t.time_base / total,
            100 * t.time_finalise / total)
    # When finalising dominates, break it down: lifting the transforms,
    # applying them to the original basis, and the final size reduction are
    # three quite different costs with three quite different remedies.
    if t.time_finalise > 0.25 * total
        @printf("      finalise = lift %5.1f%%  apply %5.1f%%  final-SR %5.1f%%\n",
                100 * t.time_collect / total, 100 * t.time_apply / total,
                100 * t.time_final_sr / total)
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
    run_family(name, sizes; seed=1, kwargs...) -> Vector

Sweep one lattice family over a list of size parameters.

`sizes` is the generator's size parameter, which is not always the dimension:
the knapsack family produces `n+1` columns and the q-ary family `2n`.
"""
function run_family(name::AbstractString, sizes; seed::Integer = 1,
                    breakdown::Bool = true, time_budget::Real = 60.0, kwargs...)
    families = Flatter.lattice_families()
    index = findfirst(f -> f.name == name, families)
    index === nothing && error("unknown family $name; have " *
                               join((f.name for f in families), ", "))
    family = families[index]

    println("\n=== $(family.name) ===")
    print_header()
    results = []
    for n in sizes
        rng = MersenneTwister(hash((seed, name, n)))
        bundle = family.generate(rng, n)
        r = measure(bundle; kwargs...)
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

The default sweep: every family at a range of sizes, small first so a stall
shows up before much time has been spent.
"""
function run_all(; sizes = [8, 16, 32, 48], seed::Integer = 1, kwargs...)
    results = []
    for family in Flatter.lattice_families()
        append!(results, run_family(family.name, sizes; seed = seed, kwargs...))
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
if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
