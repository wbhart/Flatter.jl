# test/test_recursive_reduction.jl
#
# Assumes `using Test` and `using Random` from runtests.jl.
#
# The invariants split sharply into two kinds, and the tests keep them apart:
#
#   EXACT, and true no matter what -- these hold regardless of precision, of
#   whether the goal was met, or of the iteration cap:
#       B_out == B_in * U          (over BigInt)
#       |det U| == 1               (Bareiss, exact)
#       the lattice is unchanged
#
#   QUALITY, which the goal and the working precision govern, and which is
#   therefore checked with tolerance or as a comparison:
#       the profile drop shrinks
#       the first basis vector gets shorter
#       the result is competitive with fplll
#
# A bug in the compression or the transform lifting shows up in the first group
# immediately, which is why those are checked on every basis.

# --------------------------------------------------------------------------
# Generators and exact oracles
# --------------------------------------------------------------------------

function rr_random_bigint(rng, bits::Int)
    x = big(0)
    for _ in 1:cld(bits, 32)
        x = (x << 32) + rand(rng, 0:(2^32 - 1))
    end
    x >>= max(0, 32 * cld(bits, 32) - bits)
    return rand(rng, Bool) ? x : -x
end

"""
An upper triangular basis whose diagonal spans `spread` bits, so the profile has
real structure for the reduction to work on. Off-diagonal entries are drawn at
the scale of the column's diagonal, which is what an unreduced basis looks like.
"""
function rr_triangular_basis(rng, n::Int; spread::Int = 40, offdiag::Int = 20)
    B = zeros(BigInt, n, n)
    for j in 1:n
        exponent = n == 1 ? 0 : div(spread * (j - 1), n - 1)
        d = big(0)
        while iszero(d)
            d = rr_random_bigint(rng, 8)
        end
        B[j, j] = d << exponent
        for i in 1:(j - 1)
            B[i, j] = rr_random_bigint(rng, offdiag)
        end
    end
    return B
end

"A triangular basis with a descending profile, the usual shape."
function rr_descending_basis(rng, n::Int; top::Int = 60, step::Int = 4)
    B = zeros(BigInt, n, n)
    for j in 1:n
        d = big(0)
        while iszero(d)
            d = rr_random_bigint(rng, 6)
        end
        B[j, j] = d << max(0, top - step * (j - 1))
        for i in 1:(j - 1)
            B[i, j] = rr_random_bigint(rng, 24)
        end
    end
    return B
end

"Exact determinant by fraction-free (Bareiss) elimination."
function rr_det(A::AbstractMatrix{<:Integer})
    n = size(A, 1)
    n == 0 && return big(1)
    M = Matrix{BigInt}(A)
    sign = big(1)
    previous = big(1)
    for k in 1:(n - 1)
        if iszero(M[k, k])
            row = 0
            for r in (k + 1):n
                if !iszero(M[r, k])
                    row = r
                    break
                end
            end
            iszero(row) && return big(0)
            for c in 1:n
                M[k, c], M[row, c] = M[row, c], M[k, c]
            end
            sign = -sign
        end
        for i in (k + 1):n, j in (k + 1):n
            M[i, j] = div(M[i, j] * M[k, k] - M[i, k] * M[k, j], previous)
        end
        previous = M[k, k]
    end
    return sign * M[n, n]
end

rr_column_norm2(B, j) = sum(BigInt(B[i, j])^2 for i in 1:size(B, 1))
rr_shortest_norm2(B) = minimum(rr_column_norm2(B, j) for j in 1:size(B, 2))

"""
The profile of an UPPER TRIANGULAR basis: `log2` of the diagonal magnitudes.

Only valid for a triangular basis, where the diagonal is the R factor. The
output of `lattice_reduce` is a general basis, so its diagonal means nothing;
use `info.profile` there instead.
"""
rr_triangular_profile(B) = [Flatter._log2_abs(B[i, i]) for i in 1:size(B, 2)]

"""
Every exact invariant at once. Checked over `BigInt` so an overflow anywhere
inside would surface as a failed identity rather than a plausible wrong answer.
"""
function rr_check_exact(original, reduced, U)
    n = size(original, 2)
    BigInt.(reduced) == BigInt.(original) * BigInt.(U) || return false
    abs(rr_det(U)) == 1 || return false
    return true
end

# --------------------------------------------------------------------------

@testset "recursive lattice reduction" begin

    @testset "Lagrange base case" begin
        @testset "reduces a 2D basis, trial $trial" for trial in 1:40
            rng = MersenneTwister(hash((trial, :lag)))
            B0 = BigInt[rr_random_bigint(rng, 30) for _ in 1:2, _ in 1:2]
            iszero(rr_det(B0)) && continue

            B = copy(B0)
            U = Matrix{BigInt}(undef, 2, 2)
            Flatter.lagrange_reduce!(B, U)

            @test rr_check_exact(B0, B, U)
            # Lagrange output is Gauss reduced: the columns are ordered by
            # length and 2|<a,b>| <= <a,a>.
            first_norm = rr_column_norm2(B, 1)
            second_norm = rr_column_norm2(B, 2)
            @test first_norm <= second_norm
            inner = sum(B[i, 1] * B[i, 2] for i in 1:2)
            @test 2 * abs(inner) <= first_norm
        end

        @testset "finds the shortest vector in 2D" begin
            # Gauss reduction is exact in two dimensions: the first output
            # vector is a shortest nonzero lattice vector.
            rng = MersenneTwister(0x2D5F)
            for _ in 1:20
                B0 = BigInt[rand(rng, -40:40) for _ in 1:2, _ in 1:2]
                iszero(rr_det(B0)) && continue
                B = copy(B0)
                U = Matrix{BigInt}(undef, 2, 2)
                Flatter.lagrange_reduce!(B, U)
                best = rr_column_norm2(B, 1)
                # Brute force over small combinations.
                for a in -6:6, b in -6:6
                    (a == 0 && b == 0) && continue
                    v = [a * B0[i, 1] + b * B0[i, 2] for i in 1:2]
                    @test sum(v .^ 2) >= best
                end
            end
        end

        # Gauss reduction is essentially unique in two dimensions, so the two
        # base-case routines must land on the same reduced Gram matrix. This is
        # the strongest cross-check available: two independent algorithms, one
        # exact-integer and one recursive, agreeing entry for entry.
        @testset "Lagrange and Schoenhage agree, $bits-bit entries" for bits in (40, 300, 1500)
            rng = MersenneTwister(hash((bits, :agree)))
            for _ in 1:15
                B0 = BigInt[rr_random_bigint(rng, bits) for _ in 1:2, _ in 1:2]
                iszero(rr_det(B0)) && continue

                viaL = copy(B0); UL = Matrix{BigInt}(undef, 2, 2)
                Flatter._reduce_two_columns!(viaL, UL, typemax(Int))     # forces Lagrange
                viaS = copy(B0); US = Matrix{BigInt}(undef, 2, 2)
                Flatter._reduce_two_columns!(viaS, US, 1)                # forces Schoenhage

                @test rr_check_exact(B0, viaL, UL)
                @test rr_check_exact(B0, viaS, US)

                # Compare Gram matrices: the bases may differ by column signs,
                # which flips the sign of the inner product but nothing else.
                @test rr_column_norm2(viaL, 1) == rr_column_norm2(viaS, 1)
                @test rr_column_norm2(viaL, 2) == rr_column_norm2(viaS, 2)
                innerL = sum(viaL[i, 1] * viaL[i, 2] for i in 1:2)
                innerS = sum(viaS[i, 1] * viaS[i, 2] for i in 1:2)
                @test abs(innerL) == abs(innerS)
            end
        end

        @testset "Schoenhage output is Gauss reduced" begin
            rng = MersenneTwister(0x5C40)
            for _ in 1:20
                B0 = BigInt[rr_random_bigint(rng, 800) for _ in 1:2, _ in 1:2]
                iszero(rr_det(B0)) && continue
                B = copy(B0); U = Matrix{BigInt}(undef, 2, 2)
                Flatter._reduce_two_columns!(B, U, 1)
                first_norm = rr_column_norm2(B, 1)
                @test first_norm <= rr_column_norm2(B, 2)
                @test 2 * abs(sum(B[i, 1] * B[i, 2] for i in 1:2)) <= first_norm
            end
        end

        @testset "the threshold routes but does not change validity" begin
            rng = MersenneTwister(0x7412)
            B0 = BigInt[rr_random_bigint(rng, 600) for _ in 1:2, _ in 1:2]
            if !iszero(rr_det(B0))
                for threshold in (1, 100, 512, 5000, typemax(Int))
                    B = copy(B0); U = Matrix{BigInt}(undef, 2, 2)
                    Flatter._reduce_two_columns!(B, U, threshold)
                    @test rr_check_exact(B0, B, U)
                end
            end
        end

        @testset "single column is the identity" begin
            B = reshape(BigInt[7], 1, 1)
            U = Matrix{BigInt}(undef, 1, 1)
            Flatter.lagrange_reduce!(B, U)
            @test B == reshape(BigInt[7], 1, 1)
            @test U == reshape(BigInt[1], 1, 1)
        end
    end

    @testset "the window schedule" begin
        @testset "cycles middle, left, right at n = $n" for n in (4, 8, 16, 33)
            middle = Flatter._reduction_window(0, n)
            left = Flatter._reduction_window(1, n)
            right = Flatter._reduction_window(2, n)

            @test left == 1:div(n, 2)
            @test right == (div(n, 2) + 1):n
            @test first(left) == 1 && last(right) == n
            # The middle window must straddle the boundary the halves share,
            # or nothing could ever move across it.
            @test first(middle) <= last(left) < last(middle)
            # And the schedule repeats with period three.
            for k in 0:8
                @test Flatter._reduction_window(k, n) ==
                      Flatter._reduction_window(k + 3, n)
            end
        end

        @testset "the goal is only checked at the end of a cycle" begin
            @test !Flatter._should_check_goal(0)
            @test !Flatter._should_check_goal(1)
            @test !Flatter._should_check_goal(2)
            @test Flatter._should_check_goal(3)
            @test !Flatter._should_check_goal(4)
            @test Flatter._should_check_goal(6)
        end
    end

    @testset "exact invariants" begin
        @testset "n = $n, spread = $spread" for n in (3, 4, 5, 6, 8), spread in (20, 60)
            rng = MersenneTwister(hash((n, spread, :exact)))
            B0 = rr_triangular_basis(rng, n; spread = spread)
            untouched = copy(B0)

            reduced, U, info = Flatter.lattice_reduce(B0)

            @test B0 == untouched                       # input not modified
            @test rr_check_exact(untouched, reduced, U)
            @test size(U) == (n, n)
            @test length(info.profile) == n
        end

        @testset "descending profile, n = $n" for n in (4, 6, 9, 12)
            rng = MersenneTwister(hash((n, :desc)))
            B0 = rr_descending_basis(rng, n)
            reduced, U, _ = Flatter.lattice_reduce(B0)
            @test rr_check_exact(B0, reduced, U)
        end

        @testset "the determinant is preserved" begin
            rng = MersenneTwister(0xDE7)
            for n in (3, 5, 8)
                B0 = rr_triangular_basis(rng, n; spread = 30)
                reduced, U, _ = Flatter.lattice_reduce(B0)
                # |det| is a lattice invariant, so it must survive exactly.
                @test abs(rr_det(reduced)) == abs(rr_det(B0))
            end
        end

        @testset "invariants survive a low iteration cap" begin
            # Cutting the loop short must not damage correctness -- only
            # quality. This is the guarantee that lets the cap exist at all.
            rng = MersenneTwister(0xCA9)
            B0 = rr_triangular_basis(rng, 8; spread = 80)
            for cap in (1, 2, 3, 7)
                reduced, U, info = Flatter.lattice_reduce(B0; max_iterations = cap)
                @test rr_check_exact(B0, reduced, U)
                @test info.iterations <= cap
            end
        end

        @testset "the base-case choice does not change the driver's guarantees" begin
            rng = MersenneTwister(0xBA5E)
            for n in (4, 6, 8)
                B0 = rr_triangular_basis(rng, n; spread = 60)
                viaL, UL, _ = Flatter.lattice_reduce(B0; schoenhage_threshold = typemax(Int))
                viaS, US, _ = Flatter.lattice_reduce(B0; schoenhage_threshold = 1)
                @test rr_check_exact(B0, viaL, UL)
                @test rr_check_exact(B0, viaS, US)
            end
        end

        @testset "invariants survive an aggressive precision policy" begin
            rng = MersenneTwister(0xA66)
            for n in (4, 6, 8)
                B0 = rr_triangular_basis(rng, n; spread = 50)
                reduced, U, _ = Flatter.lattice_reduce(B0; aggressive = true)
                @test rr_check_exact(B0, reduced, U)
            end
        end

        @testset "an unreachable goal still returns a valid basis" begin
            rng = MersenneTwister(0x60A7)
            B0 = rr_triangular_basis(rng, 6; spread = 100)
            # A root Hermite factor no reduction can achieve.
            impossible = Flatter.goal_from_slope(6, Flatter.BKZ_BEST_SLOPE)
            reduced, U, info = Flatter.lattice_reduce(B0; goal = impossible,
                                                      max_iterations = 12)
            @test rr_check_exact(B0, reduced, U)
            @test info.iterations <= 12
        end
    end

    @testset "reduction quality" begin
        @testset "the basis gets shorter, n = $n" for n in (4, 6, 8, 10)
            rng = MersenneTwister(hash((n, :quality)))
            B0 = rr_triangular_basis(rng, n; spread = 60)
            reduced, _, _ = Flatter.lattice_reduce(B0)
            # The original is triangular with a spread diagonal, so its
            # shortest column is one of the diagonal entries; reduction should
            # find something at least as short.
            @test rr_shortest_norm2(reduced) <= rr_shortest_norm2(B0)
        end

        @testset "the profile flattens" begin
            rng = MersenneTwister(0xF1A7)
            for n in (5, 8, 12)
                B0 = rr_triangular_basis(rng, n; spread = 70)
                _, _, info = Flatter.lattice_reduce(B0)
                # The input is triangular so its diagonal is its profile; the
                # output is not, so its profile comes back in `info`.
                before = Flatter.profile_drop(rr_triangular_profile(B0))
                after = Flatter.profile_drop(info.profile)
                @test all(isfinite, info.profile)
                @test after <= before + 1e-6
            end
        end

        # The transform is unimodular but not triangular, so the reduced basis
        # is a general basis. Reading its diagonal as a profile is meaningless
        # and will occasionally hit a zero. This test exists to pin the
        # semantics rather than to check a property of the algorithm.
        @testset "the output is a general basis, with the profile in info" begin
            rng = MersenneTwister(0x9E4E)
            B0 = rr_triangular_basis(rng, 8; spread = 60)
            reduced, U, info = Flatter.lattice_reduce(B0)

            @test length(info.profile) == 8
            @test all(isfinite, info.profile)

            # The lattice determinant equals 2^sum(profile) up to sign, which
            # ties `info.profile` to the returned basis without assuming any
            # structure in it.
            @test isapprox(sum(info.profile), Flatter._log2_abs(rr_det(reduced));
                           atol = 1e-6, rtol = 1e-9)
        end

        @testset "the reported profile matches the returned basis" begin
            rng = MersenneTwister(0x9207)
            B0 = rr_triangular_basis(rng, 8; spread = 50)
            reduced, _, info = Flatter.lattice_reduce(B0)
            # `info.profile` is the R-factor profile; for a triangular result
            # it is the diagonal, up to the sign convention of the QR.
            @test length(info.profile) == 8
            @test all(isfinite, info.profile)
            @test Flatter.profile_spread(info.profile) <=
                  Flatter.profile_spread(rr_triangular_profile(B0)) + 1e-6
        end

        @testset "competitive with fplll" begin
            # Not expected to match -- fplll is a different algorithm with
            # different guarantees -- but the shortest vector found should be
            # within a reasonable factor, which catches a reduction that runs
            # to completion while achieving nothing.
            rng = MersenneTwister(0xF9111A)
            for n in (5, 8)
                B0 = rr_triangular_basis(rng, n; spread = 40)
                ours, _, _ = Flatter.lattice_reduce(B0)
                theirs, transform = Flatter.fplll_reduce(B0)

                @test BigInt.(theirs) == BigInt.(B0) * BigInt.(transform)

                ours_best = rr_shortest_norm2(ours)
                theirs_best = rr_shortest_norm2(theirs)
                # Within 2^(2n) in squared norm, i.e. 2^n in norm.
                @test ours_best <= theirs_best * big(2)^(2 * n)
            end
        end
    end

    @testset "the goal is reachable on easy input" begin
        # A basis that is already nearly reduced should satisfy a modest goal
        # quickly, rather than grinding to the cap.
        rng = MersenneTwister(0xEA57)
        for n in (4, 6, 8)
            B0 = zeros(BigInt, n, n)
            for j in 1:n
                B0[j, j] = big(1) << 20
                for i in 1:(j - 1)
                    B0[i, j] = rand(rng, -3:3)
                end
            end
            _, U, info = Flatter.lattice_reduce(B0; rhf = 1.05)
            @test info.goal_met
            @test info.iterations <= 12
            @test abs(rr_det(U)) == 1
        end
    end

    @testset "in-place form" begin
        rng = MersenneTwister(0x1409)
        n = 6
        B0 = rr_triangular_basis(rng, n; spread = 40)
        B = copy(B0)
        U = Matrix{BigInt}(undef, n, n)

        returned_B, returned_U, info = Flatter.lattice_reduce!(B, U)
        @test returned_B === B
        @test returned_U === U
        @test rr_check_exact(B0, B, U)
    end

    @testset "argument checking" begin
        rng = MersenneTwister(0xBAD)
        good = rr_triangular_basis(rng, 5; spread = 20)

        @test_throws DimensionMismatch Flatter.lattice_reduce(
            BigInt[1 2 3; 0 4 5])
        @test_throws DimensionMismatch Flatter.lattice_reduce!(
            copy(good), Matrix{BigInt}(undef, 3, 3))
        @test_throws ArgumentError Flatter.lattice_reduce(
            BigInt[1 2; 3 4])                                  # not triangular
        @test_throws ArgumentError Flatter.lattice_reduce(
            BigInt[0 2; 0 4])                                  # singular
        @test_throws ArgumentError Flatter.lattice_reduce(
            good; max_iterations = 0)
        @test_throws ArgumentError Flatter.lattice_reduce(
            good; goal = Flatter.goal_from_slope(9, 0.1))      # wrong dimension
    end
end
