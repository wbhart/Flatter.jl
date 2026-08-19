# lattices/profile_lift.jl
#
# A focused look inside the `lift` phase, which the benchmark identified as the
# dominant cost on most families (77% of runtime on spread-48).
#
# The benchmark's timers are coarse: they say `_collect_transform` is expensive
# but not which part of it. Julia's sampling profiler can see inside, and at
# this point that is a better instrument than more hand-placed timers.
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
function strassen_cutoff_sweep(n::Int = 48, bits::Int = 1200)
    rng = MersenneTwister(7)
    make() = BigInt[(big(1) << bits) + rand(rng, 1:1000) for _ in 1:n, _ in 1:n]
    A, B = make(), make()

    println("\n", repeat("=", 70))
    @printf("Strassen cutoff sweep: %d x %d BigInt, ~%d-bit entries\n", n, n, bits)
    println(repeat("=", 70))

    reference = A * B
    for cutoff in (8, 16, 32, 64, 128, 4096)
        Flatter.strassen(A, B; cutoff = cutoff)               # warm up
        elapsed = @elapsed result = Flatter.strassen(A, B; cutoff = cutoff)
        @printf("  cutoff %5d : %7.1f ms %s\n", cutoff, 1000 * elapsed,
                result == reference ? "" : "  MISMATCH")
    end
    elapsed = @elapsed A * B
    @printf("  plain *      : %7.1f ms\n", 1000 * elapsed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for (name, generate) in CASES
        inspect(name, generate)
    end
    strassen_cutoff_sweep()
end
