# test/test_heuristic2.jl
#
# End-to-end tests for flatter's phase-2 heuristic.  The split-tree mechanics
# themselves are covered in test_sublattice_split.jl; these tests check the
# phase-2 representation, LatRedRelSR-style auxiliary block, and the handoff to
# the already-tested phase-3 Heuristic3.

@testset "heuristic2" begin
    function h2_triangular_basis(n::Int)
        B = zeros(BigInt, n, n)
        for j in 1:n
            # A deliberately steep profile, so the initial goal check cannot
            # make the phase-2 schedule vacuous.
            bits = 18 + 7 * (n - j)
            B[j, j] = big(1) << bits
            for i in 1:(j - 1)
                sign = isodd(i + 2j) ? -1 : 1
                B[i, j] = sign * ((big(1) << max(bits - 2, 1)) + 17i + 11j)
            end
        end
        return B
    end

    @testset "left/right/all cycle is exact and hands the whole to phase 3" begin
        n = 8
        B0 = h2_triangular_basis(n)
        telemetry = Flatter.ReductionTelemetry()

        reduced, U, info = Flatter.heuristic2_reduce(
            B0; base_cutoff = 4, max_iterations = 20,
            telemetry = telemetry, validate = true)

        @test reduced == B0 * U
        @test abs(rr_det(U)) == 1
        @test info.iterations == 3
        @test info.stopped == :phase2_complete
        @test length(info.profile) == n
        @test all(isfinite, info.profile)
        @test isapprox(sum(info.profile), Flatter._log2_abs(abs(rr_det(reduced)));
                       atol = 1e-7, rtol = 1e-10)

        @test telemetry.h2_calls == 1
        @test telemetry.h2_left_steps == 1
        @test telemetry.h2_right_steps == 1
        @test telemetry.h2_all_steps == 1
        @test telemetry.h2_phase3_calls == 1
        @test telemetry.h2_relative_calls == 1
    end


    @testset "structured transform collection matches the full product" begin
        n = 8
        k = 4
        I8 = Matrix{BigInt}(I, n, n)

        # Left-step shape [A B; 0 I].
        L = copy(I8)
        L[1:k, 1:k] .= BigInt[
            1  2  0  0;
            0  1 -1  0;
            0  0  1  3;
            0  0  0  1
        ]
        L[1:k, (k + 1):n] .= reshape(BigInt.(1:16), k, k)

        # Right-step shape [I C; 0 D].
        R = copy(I8)
        R[1:k, (k + 1):n] .= reshape(BigInt.(-8:7), k, k)
        R[(k + 1):n, (k + 1):n] .= BigInt[
            1  0  2  0;
            0  1  0 -1;
            0  0  1  1;
            0  0  0  1
        ]

        # The all-step may affect the whole basis.
        A = copy(I8)
        A[1, 3] = 2
        A[2, 6] = -3
        A[4, 8] = 5
        A[5, 7] = 1

        expected = (L * R) * A
        got = Flatter._h2_collect_steps([L, R, A], [(1, k), (k + 1, n), (1, n)], n)
        @test got == expected
    end

    @testset "LatRedRelSR auxiliary block keeps the exact joint transform" begin
        n = 4
        B10 = h2_triangular_basis(n)
        B20 = BigInt[
             1234567   -777777    314159;
            -3456789    888888   -271828;
             4567891   -999999    161803;
            -5678912   1111111   -141421
        ]
        B1 = copy(B10)
        B2 = copy(B20)
        U1 = Matrix{BigInt}(undef, n, n)
        U2 = Matrix{BigInt}(undef, n, size(B2, 2))
        goal = Flatter.goal_from_rhf(n, Flatter.DEFAULT_REDUCTION_RHF)
        telemetry = Flatter.ReductionTelemetry()

        Flatter._heuristic2_relative_child!(B1, B2, U1, U2;
            goal = goal, split = Flatter.SplitPhase2(n),
            max_iterations = 20, aggressive = false,
            schoenhage_threshold = Flatter.DEFAULT_SCHOENHAGE_THRESHOLD,
            base_cutoff = 4,
            blocksize = Flatter.DEFAULT_FUSED_BLOCKSIZE,
            panelsize = Flatter.DEFAULT_FUSED_PANELSIZE,
            validate = true, telemetry = telemetry, depth = 1)

        @test B1 == B10 * U1
        @test B2 == B20 + B10 * U2
        @test abs(rr_det(U1)) == 1
        @test telemetry.h2_relative_calls == 1
    end

    @testset "three columns use the two-step phase-2 schedule" begin
        n = 3
        B0 = h2_triangular_basis(n)
        telemetry = Flatter.ReductionTelemetry()

        reduced, U, info = Flatter.heuristic2_reduce(
            B0; base_cutoff = 2, max_iterations = 20,
            telemetry = telemetry, validate = true)

        @test reduced == B0 * U
        @test abs(rr_det(U)) == 1
        @test info.iterations == 2
        @test isapprox(sum(info.profile), Flatter._log2_abs(abs(rr_det(reduced)));
                       atol = 1e-7, rtol = 1e-10)
        @test telemetry.h2_left_steps == 1
        @test telemetry.h2_right_steps == 0
        @test telemetry.h2_all_steps == 1
        @test telemetry.h2_phase3_calls == 1
    end

    @testset "an already reduced profile exits before the phase-2 cycle" begin
        n = 8
        B0 = zeros(BigInt, n, n)
        for i in 1:n
            B0[i, i] = big(1) << 30
        end
        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.heuristic2_reduce(
            B0; base_cutoff = 4, telemetry = telemetry)

        @test reduced == B0
        @test U == BigInt[i == j ? 1 : 0 for i in 1:n, j in 1:n]
        @test info.iterations == 0
        @test info.goal_met
        @test info.stopped == :goal
        @test telemetry.h2_calls == 1
        @test telemetry.h2_left_steps == 0
        @test telemetry.h2_right_steps == 0
        @test telemetry.h2_all_steps == 0
    end

    @testset "argument checking" begin
        B = h2_triangular_basis(6)
        U = Matrix{BigInt}(undef, 6, 6)

        bad = copy(B)
        bad[6, 1] = 1
        @test_throws ArgumentError Flatter.heuristic2_reduce!(bad, U; base_cutoff = 3)
        @test_throws DimensionMismatch Flatter.heuristic2_reduce!(
            B, Matrix{BigInt}(undef, 5, 5); base_cutoff = 3)
        @test_throws ArgumentError Flatter.heuristic2_reduce!(
            B, U; base_cutoff = 3, _split = Flatter.SplitPhase3(6))
        @test_throws ArgumentError Flatter.heuristic2_reduce!(B, U; base_cutoff = 1)
    end
end
