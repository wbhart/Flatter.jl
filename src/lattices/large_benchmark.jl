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
# than quietly comparing against something else.  Historical `ours` rows are
# the teaching reducer; `ours-heuristic` rows are the current default heuristic
# dispatcher, so both baselines remain available.
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
    (family = "h1-known-cond",     size = 256),
    (family = "q-ary",             size = 128),
    (family = "relation",          size = 255),
]

"""
The subset that finishes in seconds rather than minutes: about 17s and 14s
against 96, 506 and 1054 for the others. Used by [`run_cheap`](@ref) for
iterating without paying for the full set each time.
"""
const CHEAP_CASES = [
    (family = "random-triangular", size = 256),
    (family = "knapsack",          size = 255),
]

# A cheap dedicated phase-1 Heuristic1 probe.  The normal LARGE_CASES include
# the same construction at dimension 256; this 128-dimensional version remains
# useful for quick H1-only regression checks.  Unlike the ordinary dense
# families, it supplies a rigorous positive condition bound and therefore
# deliberately selects Heuristic1 instead of CondUnknown.
const HEURISTIC1_CASES = [
    (family = "h1-known-cond", size = 128),
]

# Historical subset used while only H2/H3 were available.  CondUnknown now
# allows the heuristic dispatcher to handle dense and rectangular large cases too.
const HEURISTIC2_LARGE_CASES = [
    (family = "random-triangular", size = 256),
    (family = "knapsack",          size = 255),
    (family = "q-ary",             size = 128),
]

"""
`relation` at a dimension small enough to sweep but large enough to be
representative — the full case at 256 takes about 1050 seconds, this one a
couple of minutes.

It is the family where the trailing update is large (19.7% at dimension 256
against 2.9% for random-triangular), so it is the one where panelling could
plausibly pay.
"""
const RELATION_PROBE = [(family = "relation", size = 128)]

"""
Modest sizes for comparing the tiled representation update against the
monolithic one.

The tiled route is a different computation and may be far slower before it is
tuned, so the comparison should be at a size where a tenfold loss still finishes
in reasonable time. `CHEAP_CASES` at dimension 256 is not that: a run taking
twenty minutes is indistinguishable from a hang.

Dimension 128 rather than 64: at 64 every configuration finished in under a
tenth of a second, which is below the noise floor and says nothing about which
is faster.
"""
const TILED_PROBE = [
    (family = "random-triangular", size = 128),
    (family = "knapsack",          size = 127),
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

"""
    heuristic1_known_condition_lattice(rng, n) -> NamedTuple

A dense square integer basis with an exact, cheap condition-number bound for
exercising Heuristic1.  Let `v` have random entries in `{+1, -1}` and set

    B = I + v*v'.

`B` is symmetric positive definite.  Its eigenvalues, hence singular values,
are `1` with multiplicity `n-1` and `n+1` in the direction `v`.  Consequently
`cond_2(B) = n+1`; for a square QR factorisation `B = Q*R`, orthogonality of `Q`
gives the same `cond_2(R)`.  The returned `log_cond` is therefore a genuine
upper bound, nudged upward by one Float64 ulp so rounding cannot turn equality
into an accidental underestimate.

For `n >= 2` every entry is nonzero, so no row/column reversal can make the
basis triangular: the public heuristic dispatcher must take its dense phase-1
route, and the positive bound selects Heuristic1 rather than CondUnknown.
"""
function heuristic1_known_condition_lattice(rng, n::Integer)
    n >= 2 || throw(ArgumentError("Heuristic1 probe needs dimension at least two"))
    v = BigInt[rand(rng, Bool) ? 1 : -1 for _ in 1:n]
    basis = Matrix{BigInt}(undef, n, n)
    for j in 1:n, i in 1:n
        basis[i, j] = v[i] * v[j] + (i == j ? 1 : 0)
    end

    log_cond = nextfloat(log2(Float64(n + 1)))
    return (basis = basis, name = "h1-known-cond", dimension = Int(n),
            log2_determinant = log2(Float64(n + 1)),
            planted_norm2 = big(2), log_cond = log_cond)
end

"Reference kind for one of our reducers.  Legacy `ours` rows are teaching baselines."
function ours_reference_kind(algorithm::Symbol)
    algorithm === :teaching && return "ours"
    algorithm === :heuristic && return "ours-heuristic"
    throw(ArgumentError("algorithm must be :teaching or :heuristic"))
end

struct Reference
    machine::String
    family::String
    size::Int
    kind::String          # "fplll", legacy teaching "ours", or "ours-heuristic"
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

"""
The LAST matching row wins, so a refreshed baseline supersedes a stale one
without the old row having to be deleted by hand. Rows are only ever appended,
so the file keeps its history.
"""
function find_reference(entries, machine, family, size, kind, basis)
    found = nothing
    for e in entries
        e.family == family && e.size == size && e.kind == kind &&
            e.basis == basis || continue
        e.machine == machine ||
            @warn "cached timing came from another machine" family size kind e.machine
        found = e
    end
    return found
end

"""
    run_cheap(; seed = 1)

The two quick cases only, touching nothing on disk.

For iterating on instrumentation or on a change whose effect should show up
everywhere. Roughly half a minute, against half an hour for the full set. It
never writes `reference_times.tsv` and never runs fpLLL, so a baseline recorded
earlier stays exactly as it was.

`scrambled`, `q-ary` and `relation` are excluded because they take 96, 506 and
1054 seconds respectively. Run [`run_large`](@ref) when the change is ready to
be judged.
"""
run_cheap(; verbose::Bool = false, seed::Integer = 1) =
    run_large(; cases = CHEAP_CASES, record_baseline = false,
              skip_fplll = true, verbose = verbose, seed = seed)

"""
    run_heuristic1_probe(; kwargs...)

Run the dense known-condition case that deliberately selects Heuristic1.  This
uses the ordinary large-benchmark reference cache, including fpLLL and
`ours-heuristic` rows, but is kept separate from the standard large/cheap sets
until its runtime is known.  Pass `record_baseline = true` after the first run
looks sensible.
"""
run_heuristic1_probe(; kwargs...) =
    run_large(; cases = HEURISTIC1_CASES, algorithm = :heuristic, kwargs...)

"""
    run_large(; cases, record_baseline = false, force_fplll = false, skip_fplll = false)

Time the large cases, reusing cached reference timings where they exist.

`record_baseline` stores our own time as the figure future runs compare against.
Do that once, on the current implementation, BEFORE changing anything — a
baseline captured afterwards measures nothing.

`refresh_baseline` appends a fresh row even when one already exists.  Use it
when the existing baseline is known to be stale (for example, because it was
recorded before the recursive path was warmed).  `find_reference` deliberately
uses the last matching row, so the refreshed row supersedes the old one without
editing the TSV by hand.
"""
function _warm_large_benchmark!(algorithm::Symbol = :heuristic, warm_h1::Bool = false)
    ours_reference_kind(algorithm)  # validate early
    rng = MersenneTwister(7)

    # IMPORTANT: keep the keyword signature of these calls identical to the
    # timed call in `run_large`. Julia specialises keyword-call wrappers on the
    # keyword NamedTuple type; omitting `telemetry` here leaves the telemetry-
    # enabled `reduce_basis` path to compile inside the first measured run.

    if algorithm === :teaching
        # Exercise the base triangular entry path.
        Flatter.reduce_basis(Flatter.random_triangular_lattice(rng, 24).basis;
                             algorithm = :teaching, telemetry = Flatter.ReductionTelemetry(),
                             want_profile = false)

        # Cross both the recursive cutoff (32) and the blocked triangular
        # size-reduction cutoff (64).  Warming at dimension 48 leaves the
        # blocked kernel cold, so the first dimension-256 triangular timing
        # otherwise pays several seconds of JIT compilation.
        Flatter.reduce_basis(Flatter.random_triangular_lattice(rng, 80).basis;
                             algorithm = :teaching, telemetry = Flatter.ReductionTelemetry(),
                             want_profile = false)

        # The teaching dense/irregular entry path compiles independently.
        Flatter.reduce_basis(Flatter.scrambled_lattice(rng, 16).basis;
                             algorithm = :teaching, telemetry = Flatter.ReductionTelemetry(),
                             want_profile = false)
    else
        # An easy triangular basis can satisfy H2's goal at entry and therefore
        # fail to compile the left/right/all cycle.  Knapsack reliably exercises
        # reorientation, recursive H2, LatRedRelSR and the H2 -> H3 handoff.
        Flatter.reduce_basis(Flatter.knapsack_lattice(rng, 48).basis;
                             algorithm = :heuristic, telemetry = Flatter.ReductionTelemetry(),
                             want_profile = false)

        # Also compile the already-upper-triangular entry branch.  Dimension 80
        # is deliberately above the blocked triangular size-reduction cutoff of
        # 64; dimension 48 only warms the elementary :zz kernel.
        Flatter.reduce_basis(Flatter.random_triangular_lattice(rng, 80).basis;
                             algorithm = :heuristic, telemetry = Flatter.ReductionTelemetry(),
                             want_profile = false)

        # Phase 1 / CondUnknown compiles independently.  The knapsack probe above
        # has already warmed recursive phase 2, so a small dense case is enough
        # here to compile selection, surrogate reduction and the H2 handoff.
        Flatter.reduce_basis(Flatter.scrambled_lattice(rng, 16).basis;
                             algorithm = :heuristic, telemetry = Flatter.ReductionTelemetry(),
                             want_profile = false)

        if warm_h1
            # `log_cond` changes the keyword-call specialisation and selects a
            # wholly different phase-1 implementation, so warm it explicitly
            # when the requested cases contain the H1 probe.  Dimension 48 is
            # above the base cutoff and therefore compiles recursive phase 1.
            h1 = heuristic1_known_condition_lattice(rng, 48)
            Flatter.reduce_basis(h1.basis; algorithm = :heuristic,
                                 log_cond = h1.log_cond,
                                 telemetry = Flatter.ReductionTelemetry(),
                                 want_profile = false)
        end
    end
    return nothing
end

function run_large(; cases = LARGE_CASES, algorithm::Symbol = :heuristic,
                   record_baseline::Bool = false, refresh_baseline::Bool = false,
                   force_fplll::Bool = false, skip_fplll::Bool = false,
                   verbose::Bool = false, seed::Integer = 1)
    machine = machine_tag()
    baseline_kind = ours_reference_kind(algorithm)
    entries = load_references()
    families = Dict(f.name => f for f in Flatter.lattice_families())

    # Compile the base, recursive and dense entry paths before timing anything.
    # In particular, the recursive warm-up must cross DEFAULT_BASE_CUTOFF = 32;
    # otherwise the first dimension-256 case pays the recursive JIT cost.  The
    # H1 probe has an additional `log_cond` keyword specialisation, warmed only
    # when one of the requested cases actually needs it.
    warm_h1 = algorithm === :heuristic &&
              any(case -> case.family == "h1-known-cond", cases)
    _warm_large_benchmark!(algorithm, warm_h1)

    println("Large/probe cases [", algorithm, "]. Machine: ", machine)
    println("Cached timings in ", REFERENCE_FILE, "\n")
    @printf("%-18s %5s %11s %11s %9s %11s %s\n",
            "family", "dim", "ours(s)", "fplll(s)", "faster", "baseline(s)", "change")
    println(repeat("-", 84))

    for case in cases
        rng = MersenneTwister(hash((seed, case.family, case.size)))
        bundle = if case.family == "h1-known-cond"
            heuristic1_known_condition_lattice(rng, case.size)
        else
            family = get(families, case.family, nothing)
            family === nothing && continue
            family.generate(rng, case.size)
        end
        tag = basis_tag(bundle.basis)

        # Ours, always: this is the number the run exists to produce.  Preserve
        # the old keyword signature for ordinary cases; adding `log_cond = 0`
        # everywhere would create a new Julia keyword specialisation and revive
        # the first-run JIT problem this benchmark explicitly avoids.
        GC.gc()
        telemetry = Flatter.ReductionTelemetry()
        if algorithm === :heuristic && hasproperty(bundle, :log_cond)
            elapsed = @elapsed reduced, _, _ = Flatter.reduce_basis(
                bundle.basis; algorithm = :heuristic, log_cond = bundle.log_cond,
                telemetry = telemetry, want_profile = false)
        else
            elapsed = @elapsed reduced, _, _ = Flatter.reduce_basis(
                bundle.basis; algorithm = algorithm, telemetry = telemetry,
                want_profile = false)
        end
        if case.family == "h1-known-cond" && algorithm === :heuristic
            telemetry.h1_calls > 0 || error(
                "h1-known-cond probe did not enter Heuristic1")
            telemetry.cond_calls == 0 || error(
                "h1-known-cond probe unexpectedly entered CondUnknown")
        end
        shortest = shortest_norm_log2(reduced)

        # fpLLL, only if we have not already paid for it on this machine.
        cached = force_fplll ? nothing :
                 find_reference(entries, machine, case.family, case.size, "fplll", tag)
        if cached === nothing && skip_fplll
            # Nothing on disk and not allowed to measure: report our own time
            # alone rather than spending minutes on a comparison not asked for.
            cached = Reference(machine, case.family, case.size, "fplll", tag,
                               NaN, NaN, "")
        elseif cached === nothing
            println("    measuring fplll for ", case.family, " (once; then cached)")
            GC.gc()
            theirs_time = @elapsed theirs, _ = Flatter.fplll_reduce(bundle.basis)
            cached = save_reference(Reference(machine, case.family, case.size,
                                              "fplll", tag, theirs_time,
                                              shortest_norm_log2(theirs),
                                              string(Dates.now())))
        end

        baseline = refresh_baseline ? nothing :
                   find_reference(entries, machine, case.family, case.size, baseline_kind, tag)
        if (record_baseline || refresh_baseline) && baseline === nothing
            baseline = save_reference(Reference(machine, case.family, case.size,
                                                baseline_kind, tag, elapsed, shortest,
                                                string(Dates.now())))
        end

        change = baseline === nothing ? "no baseline" :
                 @sprintf("%.2fx %s", baseline.seconds / elapsed,
                          shortest <= baseline.norm_log2 + 1e-9 ? "" : "QUALITY WORSE")

        columns = Base.size(bundle.basis, 2)
        # Show which way round it is rather than a ratio that rounds to zero:
        # at dimension 256 on q-ary we are twenty times FASTER than fpLLL, and
        # "0.0x" hid that completely.
        ratio = isnan(cached.seconds) ? "-" :
                elapsed <= cached.seconds ?
                @sprintf("%.1fx us", cached.seconds / elapsed) :
                @sprintf("%.1fx them", elapsed / cached.seconds)
        @printf("%-18s %5d %11.2f %11.2f %9s %11s %s\n",
                case.family, columns, elapsed, cached.seconds, ratio,
                baseline === nothing ? "-" : @sprintf("%.2f", baseline.seconds),
                change)

        print_large_telemetry(telemetry, elapsed; algorithm = algorithm)
        if verbose
            # Every counter, including the ones the summary does not add up.
            # When a residual will not close, the field that is large and
            # missing from `accounted` is the one to look at.
            println()
            show(stdout, telemetry)
            @printf("      total elapsed          : %.3f s\n", elapsed)
            println()
        end
    end

    println("\nBaselines are algorithm-specific: legacy `ours` rows are teaching, while")
    println("heuristic rows use kind `ours-heuristic`. Record with")
    println("`run_large(algorithm = :", algorithm, ", record_baseline = true)`; replace")
    println("a stale same-algorithm baseline with `refresh_baseline = true`. The other")
    println("algorithm's history is left untouched.")
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

function print_fused_split(t, total; extra_accounted::Real = 0.0,
                           extra_label::AbstractString = "")
    total > 0 || return nothing
    if t.time_fused > 0
        @printf("      fusedQR %5.1f%%  = size-red %4.1f%%  reorth %4.1f%%  trailing %4.1f%%\n",
                100 * t.time_fused / total,
                100 * t.fused_split[Flatter.FUSED_TIME_REDUCE] / total,
                100 * t.fused_split[Flatter.FUSED_TIME_REORTH] / total,
                100 * t.fused_split[Flatter.FUSED_TIME_TRAILING] / total)
    end
    generic_accounted = t.time_fused + t.time_matmul + t.time_finalise + t.time_base +
                        t.time_compress + t.time_dense_qr + t.time_gc + t.time_setup +
                        t.time_push
    @printf("      matmul %5.1f%%  finalise %5.1f%%  base %5.1f%%  compress %5.1f%%",
            100 * t.time_matmul / total, 100 * t.time_finalise / total,
            100 * t.time_base / total, 100 * t.time_compress / total)
    if t.time_dense_qr > 0.01 * total
        @printf("  denseQR %5.1f%%", 100 * t.time_dense_qr / total)
    end
    @printf("\n      gc %5.1f%%  setup %5.1f%%  fold-in %5.1f%%",
            100 * t.time_gc / total, 100 * t.time_setup / total,
            100 * t.time_push / total)
    println()
    combined_accounted = generic_accounted + Float64(extra_accounted)
    if extra_accounted > 0
        @printf("      accounted %5.1f%% = generic %5.1f%% + %s %5.1f%%; residual %5.1f%%\n",
                100 * combined_accounted / total,
                100 * generic_accounted / total, extra_label,
                100 * Float64(extra_accounted) / total,
                100 * (1 - combined_accounted / total))
    elseif generic_accounted < 0.9 * total
        @printf("      [%.0f%% unaccounted]\n", 100 * (1 - generic_accounted / total))
    end
    return nothing
end

function print_heuristic3_split(t, total)
    t.h3_calls > 0 || return nothing
    h3_total = t.h3_time_precheck + t.h3_time_setup + t.h3_time_basis +
               t.h3_time_qr + t.h3_time_sr_setup + t.h3_time_sr
    @printf("        H3 %d calls, %.2fs (%.1f%% wall): pre %.2f  setup %.2f  basis %.2f  QR %.2f  SRsetup %.2f  SR %.2f\n",
            t.h3_calls, h3_total, 100 * h3_total / total,
            t.h3_time_precheck, t.h3_time_setup, t.h3_time_basis,
            t.h3_time_qr, t.h3_time_sr_setup, t.h3_time_sr)
    if t.h3_time_basis > 0
        @printf("           basis: materialise %.2f (%.0f%%)  multiply %.2f (%.0f%%)  write %.2f (%.0f%%)\n",
                t.h3_time_basis_materialise, 100 * t.h3_time_basis_materialise / t.h3_time_basis,
                t.h3_time_basis_mul, 100 * t.h3_time_basis_mul / t.h3_time_basis,
                t.h3_time_basis_write, 100 * t.h3_time_basis_write / t.h3_time_basis)
    end
    @printf("           SR pairs %d: reuseQR %d  refactor %d  triangular %d\n",
            t.h3_sr_pairs, t.h3_sr_reuse_qr, t.h3_sr_refactor, t.h3_sr_triangular)
    if t.h3_time_sr > 0
        @printf("           SR: materialise %.2f (%.0f%%)  prec %.2f (%.0f%%)  WY %.2f (%.0f%%)\n",
                t.h3_time_sr_materialise, 100 * t.h3_time_sr_materialise / t.h3_time_sr,
                t.h3_time_sr_precision, 100 * t.h3_time_sr_precision / t.h3_time_sr,
                t.h3_time_sr_factorprep, 100 * t.h3_time_sr_factorprep / t.h3_time_sr)
        @printf("               reduce %.2f (%.0f%%): orth %.2f  triangular %.2f\n",
                t.h3_time_sr_reduce, 100 * t.h3_time_sr_reduce / t.h3_time_sr,
                t.h3_time_sr_orthogonal, t.h3_time_sr_triangular)
        @printf("               write %.2f (%.0f%%)  propagate %.2f (%.0f%%)\n",
                t.h3_time_sr_writeback, 100 * t.h3_time_sr_writeback / t.h3_time_sr,
                t.h3_time_sr_propagate, 100 * t.h3_time_sr_propagate / t.h3_time_sr)
    end
    return nothing
end

function heuristic1_local_time(t)
    return t.h1_time_qr + t.h1_time_wrapper_qr + t.h1_time_relative +
           t.h1_time_u2_compose + t.h1_time_top_right + t.h1_time_collect +
           t.h1_time_compress + t.h1_time_apply
end

function heuristic2_local_time(t)
    return t.h2_time_qr + t.h2_time_wrapper_qr + t.h2_time_relative +
           t.h2_time_u2_compose + t.h2_time_top_right + t.h2_time_collect +
           t.h2_time_compress + t.h2_time_apply
end

"CondUnknown outer-loop time disjoint from H2 and the generic driver."
function cond_unknown_local_time(t)
    return t.cond_time_extract + t.cond_time_size_reduce + t.cond_time_relative +
           t.cond_time_apply + t.cond_time_sort
end

"Public irregular-entry work outside H2, CondUnknown and the generic driver."
function irregular_entry_local_time(t)
    return t.irregular_time_orientation + t.irregular_time_triangular_sr +
           t.irregular_time_transform_compose + t.irregular_time_flip
end

function print_irregular_entry_split(t, total)
    local_time = irregular_entry_local_time(t)
    local_time > 0 || return nothing
    @printf("        entry %.2fs (%.1f%% wall): orientation %.2f  triangular-SR %.2f  U-compose %.2f  flips %.2f  goal-exit %d\n",
            local_time, 100 * local_time / total,
            t.irregular_time_orientation, t.irregular_time_triangular_sr,
            t.irregular_time_transform_compose, t.irregular_time_flip,
            t.irregular_triangular_goal_exits)
    return nothing
end

function print_cond_unknown_split(t, total)
    t.cond_calls > 0 || return nothing
    local_time = cond_unknown_local_time(t)
    @printf("        CondUnknown %d call%s, %d refinement%s, rank %d, maxprec %d; %.2fs local (%.1f%% wall)\n",
            t.cond_calls, t.cond_calls == 1 ? "" : "s",
            t.cond_refinements, t.cond_refinements == 1 ? "" : "s",
            t.cond_selected_rank, t.cond_max_precision, local_time,
            100 * local_time / total)
    @printf("           extract %.2f  surrogate-SR %.2f  relative %.2f  apply %.2f  sort %.2f\n",
            t.cond_time_extract, t.cond_time_size_reduce, t.cond_time_relative,
            t.cond_time_apply, t.cond_time_sort)
    if t.cond_time_h2 > 0
        @printf("           phase-2 child wall %.2fs (overlaps H2; diagnostic only)\n",
                t.cond_time_h2)
    end
    return nothing
end

function print_large_telemetry(t, total; algorithm::Symbol = :heuristic)
    entry_local = irregular_entry_local_time(t)
    if algorithm === :heuristic
        print_irregular_entry_split(t, total)
        print_cond_unknown_split(t, total)
        print_heuristic1_split(t, total)
        print_heuristic2_split(t, total)
        print_heuristic3_split(t, total)
        h1_local = heuristic1_local_time(t)
        h2_local = heuristic2_local_time(t)
        cond_local = cond_unknown_local_time(t)
        extra = entry_local + h1_local + h2_local + cond_local
        label = h1_local > 0 ? "entry+H1+H2-local" :
                cond_local > 0 ? "entry+H2+Cond-local" : "entry+H2-local"
        print_fused_split(t, total; extra_accounted = extra, extra_label = label)
        if t.time_recursion > 0
            @printf("        recursive-call wall %.2fs (overlapping parent views; diagnostic only)\n",
                    t.time_recursion)
        end
    else
        print_irregular_entry_split(t, total)
        print_fused_split(t, total; extra_accounted = entry_local,
                          extra_label = "entry-local")
    end
    return nothing
end

function print_heuristic1_split(t, total)
    t.h1_calls > 0 || return nothing
    h1_total = heuristic1_local_time(t)
    @printf("        H1 %d calls, %d/%d/%d L/R/all, %d H3 handoffs, %.2fs measured (%.1f%% wall)\n",
            t.h1_calls, t.h1_left_steps, t.h1_right_steps, t.h1_all_steps,
            t.h1_phase3_calls, h1_total, 100 * h1_total / total)
    @printf("           repQR %.2f  relQR %.2f  relative-B2 %.2f  U2mul %.2f\n",
            t.h1_time_qr, t.h1_time_wrapper_qr, t.h1_time_relative,
            t.h1_time_u2_compose)
    @printf("           topR %.2f  collectU %.2f  compress %.2f  exact-apply %.2f\n",
            t.h1_time_top_right, t.h1_time_collect,
            t.h1_time_compress, t.h1_time_apply)
    return nothing
end

function print_heuristic2_split(t, total)
    t.h2_calls > 0 || return nothing
    h2_total = heuristic2_local_time(t)
    @printf("        H2 %d calls, %d/%d/%d L/R/all, %d H3 handoffs, %.2fs measured (%.1f%% wall)\n",
            t.h2_calls, t.h2_left_steps, t.h2_right_steps, t.h2_all_steps,
            t.h2_phase3_calls, h2_total, 100 * h2_total / total)
    @printf("           repQR %.2f  relQR %.2f  relative-B2 %.2f  U2mul %.2f\n",
            t.h2_time_qr, t.h2_time_wrapper_qr, t.h2_time_relative,
            t.h2_time_u2_compose)
    @printf("           topR %.2f  collectU %.2f  compress %.2f  exact-apply %.2f\n",
            t.h2_time_top_right, t.h2_time_collect,
            t.h2_time_compress, t.h2_time_apply)
    return nothing
end

"""
    schedule_compare(; cases, seed, budget)

All four combinations of schedule and representation update, on one instance
each, writing nothing to disk.

The two settings are not independent, which is why they belong on the same line
rather than in two separate comparisons:

  * `:legacy` is the hand-rolled middle/left/right cycle, which yields ONE
    window per iteration. The tiled update then sees one reduced tile with a gap
    either side — barely any structure to exploit.
  * `:split` is flatter's phase 3 tree, which yields TWO windows on odd
    iterations, covering the matrix with no gap between them. That is the shape
    `Heuristic3` was written around, and until now nothing has measured it.

Quality is reported alongside, because the tiled update assembles `R` locally
and reaches a different representative: a faster route to a longer vector is not
an improvement.
"""
function schedule_compare(; cases = TILED_PROBE, seed::Integer = 1,
                          budget::Real = 300.0)
    families = Dict(f.name => f for f in Flatter.lattice_families())

    # Warm up EVERY keyword combination that will be timed. Julia specialises
    # per combination, so a warm-up with different keywords compiles a different
    # method and leaves the measured one cold. That is not a small effect here:
    # it made a dimension-64 run read 0.79s where the compiled figure is 0.06s,
    # and produced two reported "speedups" that were entirely compilation.
    warm = Flatter.random_triangular_lattice(MersenneTwister(7), 24).basis
    for schedule in (:legacy, :split), tiled in (false, true)
        Flatter.reduce_basis(warm; algorithm = :teaching, schedule = schedule, tiled = tiled,
                             want_profile = false)
    end

    @printf("%-18s %5s %9s %9s %11s %11s\n",
            "family", "dim", "schedule", "update", "time(s)", "log2|b1|")
    println(repeat("-", 70))

    for case in cases
        family = get(families, case.family, nothing)
        family === nothing && continue
        bundle = family.generate(MersenneTwister(hash((seed, case.family, case.size))),
                                 case.size)
        columns = Base.size(bundle.basis, 2)
        best = Inf

        for schedule in (:legacy, :split), tiled in (false, true)
            print(stderr, "    [", case.family, " dim ", columns, "] ",
                  schedule, "/", tiled ? "tiled" : "plain", " ...\n")
            flush(stderr)
            # Best of three. At these sizes a single timing varies by more than
            # the differences being measured -- one earlier run showed a 3.98x
            # ratio where a repeat gave 2.28x, with nothing changed between
            # them, because the reference half happened to be slow.
            # A configuration can fail rather than merely be slow: the tiled
            # update assembles `R` locally, so the profile it yields can
            # under-state the spread and the precision derived from it is then
            # too low for a sub-problem. That is a result about the
            # configuration, so it is reported and the sweep continues.
            elapsed = Inf
            best_telemetry = nothing
            local reduced
            failure = nothing
            for _ in 1:3
                GC.gc()
                telemetry = Flatter.ReductionTelemetry()
                try
                    attempt = @elapsed reduced, _, _ = Flatter.reduce_basis(
                        bundle.basis; algorithm = :teaching, schedule = schedule, tiled = tiled,
                        telemetry = telemetry, want_profile = false)
                    if attempt < elapsed
                        elapsed = attempt
                        best_telemetry = telemetry
                    end
                catch problem
                    failure = problem
                    break
                end
            end

            if failure !== nothing
                @printf("%-18s %5d %9s %9s %11s %11s\n",
                        case.family, columns, schedule,
                        tiled ? "tiled" : "plain", "FAILED", "-")
                println("        ", first(sprint(showerror, failure), 100))
                continue
            end

            shortest = shortest_norm_log2(reduced)
            best = min(best, elapsed)

            @printf("%-18s %5d %9s %9s %11.2f %11.2f\n",
                    case.family, columns, schedule, tiled ? "tiled" : "plain",
                    elapsed, shortest)
            if best_telemetry !== nothing
                if tiled
                    print_heuristic3_split(best_telemetry, elapsed)
                else
                    print_fused_split(best_telemetry, elapsed)
                end
            end
            elapsed > budget && break
        end
        println()
    end

    println("All four are the same lattice; only the route and the")
    println("representative differ. Compare within a family, not across runs.")
end


"""
    heuristic2_compare(; cases = CHEAP_CASES, seed = 1, repeats = 3)

Compare the retained teaching reducer with the default heuristic dispatcher on
triangular/reoriented inputs, writing nothing to disk.

This is the short two-family probe retained for iteration.  For the complete
large comparison now that CondUnknown is available, use
[`compare_large_algorithms`](@ref).
"""
function heuristic2_compare(; cases = CHEAP_CASES, seed::Integer = 1,
                            repeats::Integer = 3)
    repeats >= 1 || throw(ArgumentError("repeats must be positive"))
    families = Dict(f.name => f for f in Flatter.lattice_families())

    # Julia specialises the two algorithm keywords independently.  Cross the
    # recursive cutoff for both before timing either one.
    warm = Flatter.random_triangular_lattice(MersenneTwister(7), 48).basis
    Flatter.reduce_basis(warm; algorithm = :teaching, want_profile = false)
    Flatter.reduce_basis(warm; algorithm = :heuristic)

    @printf("%-18s %5s %11s %11s %9s %11s %11s\n",
            "family", "dim", "teaching(s)", "heuristic(s)", "speedup",
            "plain b1", "heur b1")
    println(repeat("-", 92))

    for case in cases
        family = get(families, case.family, nothing)
        family === nothing && continue
        bundle = family.generate(MersenneTwister(hash((seed, case.family, case.size))),
                                 case.size)

        teaching_time = Inf
        heuristic_time = Inf
        teaching_reduced = nothing
        heuristic_reduced = nothing
        heuristic_info = nothing
        heuristic_telemetry = nothing

        for algorithm in (:teaching, :heuristic)
            for _ in 1:Int(repeats)
                GC.gc()
                telemetry = Flatter.ReductionTelemetry()
                local reduced, info
                elapsed = @elapsed reduced, _, info = Flatter.reduce_basis(
                    bundle.basis; algorithm = algorithm,
                    telemetry = telemetry, want_profile = false)
                if algorithm === :teaching
                    if elapsed < teaching_time
                        teaching_time = elapsed
                        teaching_reduced = reduced
                    end
                elseif elapsed < heuristic_time
                    heuristic_time = elapsed
                    heuristic_reduced = reduced
                    heuristic_info = info
                    heuristic_telemetry = telemetry
                end
            end
        end

        plain_b1 = shortest_norm_log2(teaching_reduced)
        heur_b1 = shortest_norm_log2(heuristic_reduced)
        speedup = teaching_time / heuristic_time
        # Heuristic2 follows flatter and may legitimately stop at iteration
        # zero when the requested RHF/profile goal is already met.  The
        # teaching reducer does not check its legacy goal at entry and can
        # therefore over-reduce the same input.  A longer b1 than the teaching
        # result is informative, but is not a quality failure if the requested
        # flatter goal was met.
        quality = if heur_b1 <= plain_b1 + 1e-9
            ""
        elseif heuristic_info !== nothing && heuristic_info.goal_met
            " GOAL MET"
        else
            " QUALITY WORSE"
        end

        @printf("%-18s %5d %11.2f %11.2f %8.2fx %11.2f %11.2f%s\n",
                case.family, Base.size(bundle.basis, 2), teaching_time,
                heuristic_time, speedup, plain_b1, heur_b1, quality)
        if heuristic_telemetry !== nothing
            print_large_telemetry(heuristic_telemetry, heuristic_time;
                                  algorithm = :heuristic)
        end
    end

    println("\nNo reference rows are read or written.  A speedup above 1 favours the")
    println("Heuristic2 -> Heuristic3 dispatcher; quality must remain comparable.")
end


"""
    compare_large_algorithms(; cases = LARGE_CASES, repeats = 1,
                             force_fplll = false, skip_fplll = false)

Run only the heuristic reducer and compare it with the cached teaching baseline
and cached fpLLL result for the same exact basis.  If a teaching or fpLLL row is
missing it is measured once for this comparison, but it is NOT written to disk.
The function itself never changes `reference_times.tsv`.

It returns the heuristic measurements.  Keep the Julia process open and pass
that result to [`record_heuristic_baseline!`](@ref) if, after inspecting the
comparison, these are the measurements you want to preserve as the new
heuristic baseline.  This avoids paying for q-ary/relation twice.
"""
function compare_large_algorithms(; cases = LARGE_CASES, repeats::Integer = 1,
                                  force_fplll::Bool = false,
                                  skip_fplll::Bool = false,
                                  verbose::Bool = false, seed::Integer = 1)
    repeats >= 1 || throw(ArgumentError("repeats must be positive"))
    machine = machine_tag()
    entries = load_references()
    families = Dict(f.name => f for f in Flatter.lattice_families())

    # Compile Phase 1/CondUnknown and H2/H3 before the first heuristic timing.
    _warm_large_benchmark!(:heuristic)

    println("Large teaching vs heuristic comparison, dimension ~256. Machine: ", machine)
    println("Teaching and fpLLL use cached rows when available; nothing is written.\n")
    @printf("%-18s %5s %10s %10s %8s %10s %9s %10s %10s %10s %s\n",
            "family", "dim", "teach(s)", "heur(s)", "vs teach", "fplll(s)",
            "vs fplll", "teach b1", "heur b1", "fplll b1", "quality")
    println(repeat("-", 132))

    results = NamedTuple[]

    for case in cases
        family = get(families, case.family, nothing)
        family === nothing && continue
        bundle = family.generate(MersenneTwister(hash((seed, case.family, case.size))),
                                 case.size)
        tag = basis_tag(bundle.basis)
        columns = Base.size(bundle.basis, 2)

        teaching = find_reference(entries, machine, case.family, case.size, "ours", tag)
        if teaching === nothing
            println("    measuring uncached teaching reference for ", case.family,
                    " (comparison only; not saved)")
            GC.gc()
            teach_time = @elapsed teach_reduced, _, _ = Flatter.reduce_basis(
                bundle.basis; algorithm = :teaching, want_profile = false)
            teaching = Reference(machine, case.family, case.size, "ours", tag,
                                 teach_time, shortest_norm_log2(teach_reduced), "")
        end

        fp = force_fplll ? nothing :
             find_reference(entries, machine, case.family, case.size, "fplll", tag)
        if fp === nothing && skip_fplll
            fp = Reference(machine, case.family, case.size, "fplll", tag,
                           NaN, NaN, "")
        elseif fp === nothing
            println("    measuring uncached fpLLL reference for ", case.family,
                    " (comparison only; not saved)")
            GC.gc()
            fp_time = @elapsed fp_reduced, _ = Flatter.fplll_reduce(bundle.basis)
            fp = Reference(machine, case.family, case.size, "fplll", tag,
                           fp_time, shortest_norm_log2(fp_reduced), "")
        end

        heuristic_time = Inf
        heuristic_norm = NaN
        heuristic_info = nothing
        heuristic_telemetry = nothing
        for _ in 1:Int(repeats)
            GC.gc()
            telemetry = Flatter.ReductionTelemetry()
            local reduced, info
            attempt = @elapsed reduced, _, info = Flatter.reduce_basis(
                bundle.basis; algorithm = :heuristic,
                telemetry = telemetry, want_profile = false)
            if attempt < heuristic_time
                heuristic_time = attempt
                heuristic_norm = shortest_norm_log2(reduced)
                heuristic_info = info
                heuristic_telemetry = telemetry
            end
        end

        vs_teach = teaching.seconds / heuristic_time
        vs_fp = isnan(fp.seconds) ? "-" :
                heuristic_time <= fp.seconds ?
                @sprintf("%.1fx us", fp.seconds / heuristic_time) :
                @sprintf("%.1fx them", heuristic_time / fp.seconds)
        quality = if heuristic_norm <= teaching.norm_log2 + 1e-9
            ""
        elseif heuristic_info !== nothing && heuristic_info.goal_met
            "GOAL MET"
        else
            "QUALITY WORSE"
        end

        @printf("%-18s %5d %10.2f %10.2f %7.2fx %10.2f %9s %10.2f %10.2f %10.2f %s\n",
                case.family, columns, teaching.seconds, heuristic_time, vs_teach,
                fp.seconds, vs_fp, teaching.norm_log2, heuristic_norm,
                fp.norm_log2, quality)

        heuristic_telemetry !== nothing &&
            print_large_telemetry(heuristic_telemetry, heuristic_time;
                                  algorithm = :heuristic)
        if verbose && heuristic_telemetry !== nothing
            println()
            show(stdout, heuristic_telemetry)
            @printf("      heuristic elapsed      : %.3f s\n\n", heuristic_time)
        end

        push!(results, (machine = machine, family = case.family, size = case.size,
                        basis = tag, seconds = heuristic_time,
                        norm_log2 = heuristic_norm,
                        recorded = string(Dates.now())))
    end

    println("\nNo baseline rows were written.  `vs teach > 1` favours the heuristic path.")
    println("If these exact heuristic measurements are the baseline you want to keep,")
    println("call `record_heuristic_baseline!(results)` in this same Julia process.")
    return results
end

"""
    record_heuristic_baseline!(results; refresh = false)

Append measurements returned by [`compare_large_algorithms`](@ref) as
`ours-heuristic` baseline rows.  Existing teaching (`ours`) rows are never
changed.  With `refresh=false`, an existing heuristic baseline for the same
machine/family/basis is left alone; `refresh=true` appends a new row and the
usual last-row-wins rule makes it current.
"""
function record_heuristic_baseline!(results; refresh::Bool = false)
    entries = load_references()
    saved = 0
    for r in results
        existing = find_reference(entries, r.machine, r.family, r.size,
                                  "ours-heuristic", r.basis)
        if existing !== nothing && !refresh
            println("    keeping existing heuristic baseline for ", r.family,
                    " (use refresh = true to supersede it)")
            continue
        end
        save_reference(Reference(r.machine, r.family, r.size, "ours-heuristic",
                                 r.basis, r.seconds, r.norm_log2, r.recorded))
        saved += 1
    end
    println("Recorded ", saved, " heuristic baseline row", saved == 1 ? "." : "s.")
    return saved
end

"""
    panel_sweep(; cases, panels, blocks)

Time the cheap cases across panel and block sizes, writing nothing to disk.

Both parameters are exact — the integer results do not depend on either — so the
only question is speed, and the best value is a property of the machine and the
dimension rather than a constant worth guessing. A panel too narrow spends its
time building compact-WY factors for a few reflectors; too wide and each
deferred column still has to be brought up to date elementwise when its turn
comes.
"""
function panel_sweep(; cases = CHEAP_CASES, panels = (0, 8, 16, 32, 64, 128),
                     blocks = (16,), seed::Integer = 1)
    warmed = false
    families = Dict(f.name => f for f in Flatter.lattice_families())

    for case in cases
        family = get(families, case.family, nothing)
        family === nothing && continue
        bundle = family.generate(MersenneTwister(hash((seed, case.family, case.size))),
                                 case.size)
        if !warmed
            Flatter.reduce_basis(Flatter.random_triangular_lattice(
                MersenneTwister(7), 24).basis; want_profile = false)
            warmed = true
        end

        println("\n--- ", case.family, ", dimension ", Base.size(bundle.basis, 2), " ---")
        @printf("%8s %8s %11s %11s\n", "panel", "block", "time(s)", "fusedQR")
        println(repeat("-", 42))

        for block in blocks, panel in panels
            telemetry = Flatter.ReductionTelemetry()
            GC.gc()
            elapsed = @elapsed Flatter.reduce_basis(
                bundle.basis; blocksize = block, panelsize = panel,
                telemetry = telemetry, want_profile = false)
            @printf("%8s %8d %11.2f %10.1f%%\n",
                    panel == 0 ? "off" : string(panel), block, elapsed,
                    100 * telemetry.time_fused / elapsed)
        end
    end

    println("\nBoth settings are exact, so the fastest is simply the best.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_large()
end
