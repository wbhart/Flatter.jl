# lattices/profile_lift.jl
#
# A focused look inside one reduction, for when the benchmark's phase timers
# are too coarse to say what is actually slow. They report which phase costs
# what; Julia's sampling profiler can see inside a phase.
#
#     julia --project src/lattices/profile_lift.jl
#
# Reading the output: look for whether time sits in `conjugate_transform` (the
# shifting), in `strassen`/`mul!` (the products), or in GMP allocation. Those
# imply quite different fixes -- respectively avoiding the copy, tuning the
# Strassen cutoff for BigInt, and reducing temporaries.

using Flatter
using Random
using Profile
using Printf

const CASES = [
    ("spread-48",    () -> Flatter.spread_lattice(MersenneTwister(1), 48)),
    ("knapsack-33",  () -> Flatter.knapsack_lattice(MersenneTwister(1), 32)),
    ("scrambled-48", () -> Flatter.scrambled_lattice(MersenneTwister(1), 48)),
]

"""
Report the split within `finalise` for one case, then a sampling profile of the
whole reduction. Runs once first so compilation is not what gets profiled.
"""
function inspect(name, generate; samples::Bool = true)
    println("\n", repeat("=", 70))
    println(name)
    println(repeat("=", 70))

    bundle = generate()
    telemetry = Flatter.ReductionTelemetry()
    elapsed = @elapsed Flatter.lattice_reduce(bundle.basis; telemetry = telemetry)

    @printf("total %.3f s\n", elapsed)
    println(telemetry)

    share(x) = elapsed > 0 ? 100 * x / elapsed : 0.0
    @printf("\n  lift %.1f%%  apply %.1f%%  final-SR %.1f%%  (finalise %.1f%% overall)\n",
            share(telemetry.time_collect), share(telemetry.time_apply),
            share(telemetry.time_final_sr), share(telemetry.time_finalise))

    if samples
        println("\n--- sampling profile ---")
        Profile.clear()
        @profile Flatter.lattice_reduce(bundle.basis)
        Profile.print(; format = :flat, sortedby = :count, mincount = 20)
    end
    return telemetry
end

"""
    strassen_cutoff_sweep(n, bits)

How the Strassen cutoff affects a single BigInt matrix product of the size and
entry width the lift phase actually sees.

The default cutoff was chosen for floating point work. BigInt multiplication has
a completely different cost profile -- the entries are the expensive part, not
the index arithmetic -- so the crossover may well sit somewhere else.
"""
function strassen_cutoff_sweep(n::Int = 48, bits::Int = 1200; repeats::Int = 15)
    rng = MersenneTwister(7)
    make() = BigInt[(big(1) << bits) + rand(rng, 1:1000) for _ in 1:n, _ in 1:n]
    A, B = make(), make()

    println("\n", repeat("=", 70))
    @printf("Strassen cutoff sweep: %d x %d BigInt, ~%d-bit entries, best of %d\n",
            n, n, bits, repeats)
    println(repeat("=", 70))
    println("Cutoffs at or above $n mean no recursion at all, so they should all")
    println("agree; if they do not, the spread between them is the noise floor")
    println("and any smaller difference elsewhere in the table means nothing.")
    println()

    # Minimum of several runs, not a single sample: BigInt work allocates
    # heavily, so one timing is mostly a measurement of when GC happened.
    # BigInt work allocates heavily, so a single timing mostly measures when
    # GC happened. Collect first, then take the minimum of several runs, and
    # report the spread so the noise floor is visible rather than assumed.
    function best(f)
        f()
        samples = Float64[]
        for _ in 1:repeats
            GC.gc()
            push!(samples, @elapsed f())
        end
        return minimum(samples), maximum(samples)
    end

    reference = A * B
    baseline, baseline_worst = best(() -> A * B)

    for cutoff in (8, 16, 32, 64, 128, 4096)
        result = Flatter.strassen(A, B; cutoff = cutoff)
        result == reference || println("  MISMATCH at cutoff $cutoff")
        fastest, slowest = best(() -> Flatter.strassen(A, B; cutoff = cutoff))
        @printf("  cutoff %5d : %7.1f ms (worst %7.1f) %5.2fx plain   %s\n",
                cutoff, 1000 * fastest, 1000 * slowest, fastest / baseline,
                cutoff >= n ? "(no recursion)" : "")
    end
    @printf("  plain *      : %7.1f ms (worst %7.1f)\n",
            1000 * baseline, 1000 * baseline_worst)
    println()
    println("Sanity check: every row marked (no recursion) runs identical code.")
    println("Any spread between those rows is pure measurement noise, and")
    println("differences smaller than that spread mean nothing.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    for (name, generate) in CASES
        inspect(name, generate)
    end
    strassen_cutoff_sweep()
end
