# lattices/large_benchmark.jl
#
# The few cases at a dimension where flatter's design decisions start to matter.
#
#     julia --project src/lattices/large_benchmark.jl
#
# WHY THIS EXISTS SEPARATELY
#
# The tiled size reduction crosses over against the elementary one at about
# dimension 128 (see `lattices/size_reduction_probe.jl`). Below that the driver's
# structure barely matters; above it, it decides everything. The main benchmark
# runs at 8 to 128 because that is where iteration is quick, so it cannot see any
# of this.
#
# Running everything at 256 would be intolerable as a routine check, so this
# holds only the cases that could actually move, and it CACHES the timings that
# will not change:
#
#   * fpLLL's time, which depends on the machine and the instance but not on
#     anything in this package;
#   * a baseline for our own time, recorded before a change, so a later run has
#     something to compare against.
#
# Both live in `lattices/reference_times.tsv`, keyed by machine and by a hash of
# the basis itself, so a changed generator invalidates its own entries rather
# than quietly comparing against something else.
#
# WHICH FAMILIES ARE HERE, AND WHY NOT THE OTHERS
#
# `Heuristic3` tiles the size reduction inside the factorisation, so it can only
# help where that is a meaningful share of the run. Measured at dimension 96-128:
#
#   q-ary              fusedQR 44%, size reduction 30%   -- yes
#   random-triangular  fusedQR 38%, size reduction 26%   -- yes
#   relation           fusedQR 39%, size reduction 16%   -- yes
#   knapsack           fusedQR 36%, size reduction 16%   -- yes
#   scrambled          fusedQR 24%, size reduction 11%   -- marginal, included
#   spread             fusedQR 5.5%  -- no, `finalise` is 50% of it
#   ideal              fusedQR 6-12% -- no, `matmul` and `finalise` are 62%
#
# `spread` and `ideal` are excluded deliberately: they are expensive at this
# dimension and nothing `Heuristic3` does can move them.

using Flatter
using Printf
using Random
using Dates

const REFERENCE_FILE = joinpath(@__DIR__, "reference_times.tsv")

# The size parameter, not the dimension: q-ary doubles it, knapsack and relation
# add one. These land everything between 256 and 258 columns.
const LARGE_CASES = [
    (family = "random-triangular", size = 256),
    (family = "scrambled",         size = 256),
    (family = "knapsack",          size = 255),
    (family = "q-ary",             size = 128),
    (family = "relation",          size = 255),
]

"""
    machine_tag() -> String

Something stable enough to notice that a cached timing came from elsewhere.

Not a serious fingerprint — CPU model and thread count, which is enough to catch
a file copied between machines. A mismatch warns rather than fails, since the
comparison is still meaningful if you know it happened.
"""
function machine_tag()
    cpu = try
        first(Sys.cpu_info()).model
    catch
        "unknown"
    end
    return replace("$(cpu)-$(Sys.CPU_THREADS)t", r"\s+" => "_")
end

"A hash of the basis, so a changed generator invalidates its own cached rows."
basis_tag(B) = string(hash(B); base = 16)

struct Reference
    machine::String
    family::String
    size::Int
    kind::String          # "fplll" or "ours"
    basis::String
    seconds::Float64
    norm_log2::Float64
    recorded::String
end

function load_references()
    isfile(REFERENCE_FILE) || return Reference[]
    entries = Reference[]
    for line in eachline(REFERENCE_FILE)
        (isempty(line) || startswith(line, '#')) && continue
        f = split(line, '\t')
        length(f) == 8 || continue
        push!(entries, Reference(f[1], f[2], parse(Int, f[3]), f[4], f[5],
                                 parse(Float64, f[6]), parse(Float64, f[7]), f[8]))
    end
    return entries
end

function save_reference(entry::Reference)
    fresh = !isfile(REFERENCE_FILE)
    open(REFERENCE_FILE, "a") do io
        if fresh
            println(io, "# Reference timings for lattices/large_benchmark.jl.")
            println(io, "# Cached because they are expensive and do not change:")
            println(io, "# fpLLL's time depends on the machine, not on this package,")
            println(io, "# and a recorded baseline is what a later run is compared against.")
            println(io, "# Delete a row to have it measured again.")
            println(io, "#machine\tfamily\tsize\tkind\tbasis\tseconds\tlog2_b1\trecorded")
        end
        println(io, join((entry.machine, entry.family, entry.size, entry.kind,
                          entry.basis, entry.seconds, entry.norm_log2,
                          entry.recorded), '\t'))
    end
    return entry
end

function find_reference(entries, machine, family, size, kind, basis)
    for e in entries
        e.family == family && e.size == size && e.kind == kind &&
            e.basis == basis || continue
        e.machine == machine ||
            @warn "cached timing came from another machine" family size kind e.machine
        return e
    end
    return nothing
end

"""
    run_large(; record_baseline = false, force_fplll = false)

Time the large cases, reusing cached reference timings where they exist.

`record_baseline` stores our own time as the figure future runs compare against.
Do that once, on the current implementation, BEFORE changing anything — a
baseline captured afterwards measures nothing.
"""
function run_large(; record_baseline::Bool = false, force_fplll::Bool = false,
                   seed::Integer = 1)
    machine = machine_tag()
    entries = load_references()
    families = Dict(f.name => f for f in Flatter.lattice_families())

    println("Large cases, dimension ~256. Machine: ", machine)
    println("Cached timings in ", REFERENCE_FILE, "\n")
    @printf("%-18s %5s %11s %11s %9s %11s %s\n",
            "family", "dim", "ours(s)", "fplll(s)", "faster", "baseline(s)", "change")
    println(repeat("-", 84))

    for case in LARGE_CASES
        family = get(families, case.family, nothing)
        family === nothing && continue

        bundle = family.generate(MersenneTwister(hash((seed, case.family, case.size))),
                                 case.size)
        tag = basis_tag(bundle.basis)

        # Ours, always: this is the number the run exists to produce.
        GC.gc()
        telemetry = Flatter.ReductionTelemetry()
        elapsed = @elapsed reduced, _, _ = Flatter.reduce_basis(
            bundle.basis; telemetry = telemetry, want_profile = false)
        shortest = shortest_norm_log2(reduced)

        # fpLLL, only if we have not already paid for it on this machine.
        cached = force_fplll ? nothing :
                 find_reference(entries, machine, case.family, case.size, "fplll", tag)
        if cached === nothing
            println("    measuring fplll for ", case.family, " (once; then cached)")
            GC.gc()
            theirs_time = @elapsed theirs, _ = Flatter.fplll_reduce(bundle.basis)
            cached = save_reference(Reference(machine, case.family, case.size,
                                              "fplll", tag, theirs_time,
                                              shortest_norm_log2(theirs),
                                              string(Dates.now())))
        end

        baseline = find_reference(entries, machine, case.family, case.size, "ours", tag)
        if record_baseline && baseline === nothing
            baseline = save_reference(Reference(machine, case.family, case.size,
                                                "ours", tag, elapsed, shortest,
                                                string(Dates.now())))
        end

        change = baseline === nothing ? "no baseline" :
                 @sprintf("%.2fx %s", baseline.seconds / elapsed,
                          shortest <= baseline.norm_log2 + 1e-9 ? "" : "QUALITY WORSE")

        columns = Base.size(bundle.basis, 2)
        # Show which way round it is rather than a ratio that rounds to zero:
        # at dimension 256 on q-ary we are twenty times FASTER than fpLLL, and
        # "0.0x" hid that completely.
        ratio = elapsed <= cached.seconds ?
                @sprintf("%.1fx us", cached.seconds / elapsed) :
                @sprintf("%.1fx them", elapsed / cached.seconds)
        @printf("%-18s %5d %11.2f %11.2f %9s %11s %s\n",
                case.family, columns, elapsed, cached.seconds, ratio,
                baseline === nothing ? "-" : @sprintf("%.2f", baseline.seconds),
                change)

        print_fused_split(telemetry, elapsed)
    end

    println("\nRecord a baseline with `run_large(record_baseline = true)` BEFORE")
    println("changing the implementation; afterwards the change column is what")
    println("the work bought. A quality warning means the reduction got worse,")
    println("which no amount of speed makes acceptable.")
end

function shortest_norm_log2(B)
    best = nothing
    for j in 1:Base.size(B, 2)
        total = sum(B[i, j]^2 for i in 1:Base.size(B, 1))
        iszero(total) && continue
        (best === nothing || total < best) && (best = total)
    end
    best === nothing && return NaN
    return Float64(log2(big(best))) / 2
end

function print_fused_split(t, total)
    total > 0 && t.time_fused > 0 || return nothing
    @printf("      fusedQR %5.1f%%  = size-red %4.1f%%  reorth %4.1f%%  trailing %4.1f%%\n",
            100 * t.time_fused / total,
            100 * t.fused_split[Flatter.FUSED_TIME_REDUCE] / total,
            100 * t.fused_split[Flatter.FUSED_TIME_REORTH] / total,
            100 * t.fused_split[Flatter.FUSED_TIME_TRAILING] / total)
    accounted = t.time_fused + t.time_matmul + t.time_finalise + t.time_base +
                t.time_compress + t.time_dense_qr
    @printf("      matmul %5.1f%%  finalise %5.1f%%  base %5.1f%%  compress %5.1f%%",
            100 * t.time_matmul / total, 100 * t.time_finalise / total,
            100 * t.time_base / total, 100 * t.time_compress / total)
    if t.time_dense_qr > 0.01 * total
        @printf("  denseQR %5.1f%%", 100 * t.time_dense_qr / total)
    end
    if accounted < 0.9 * total
        @printf("  [%.0f%% unaccounted]", 100 * (1 - accounted / total))
    end
    println()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_large()
end
