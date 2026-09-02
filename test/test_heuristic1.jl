# test/test_heuristic1.jl
#
# Heuristic1 is flatter's known-condition phase-1 route.  These tests exercise
# its rectangular representation, B2/U2 algebra, phase-2 split schedule and the
# public dense dispatcher selection.

@testset "heuristic1" begin
    @testset "fresh phase-1 profile matches flatter Profile(n)" begin
        p = Flatter._h1_initial_profile(4)
        @test length(p) == 4
        @test all(isnan, p)
    end

    function h1_rectangular_basis()
        # Six ambient coordinates, four full-rank columns.  Keeping the entries
        # modest lets a conservative known-condition bound give ample precision.
        return BigInt[
             17   5   2   7;
              3  19   6   1;
              2   4  23   8;
              1   3   5  29;
              4  -2   1   3;
             -1   2   4  -2
        ]
    end

    @testset "rectangular phase-1 cycle preserves the exact transform" begin
        B0 = h1_rectangular_basis()
        telemetry = Flatter.ReductionTelemetry()

        reduced, U, info = Flatter.heuristic1_reduce(
            B0; log_cond = 20, base_cutoff = 2, max_iterations = 30,
            telemetry = telemetry, validate = true)

        @test reduced == B0 * U
        @test abs(rr_det(U)) == 1
        @test info.iterations == 3
        @test info.stopped == :phase1_complete
        @test length(info.profile) == size(B0, 2)
        @test all(isfinite, info.profile)
        @test telemetry.h1_calls >= 1
        @test telemetry.h1_left_steps >= 1
        @test telemetry.h1_right_steps >= 1
        @test telemetry.h1_all_steps >= 1
        @test telemetry.h1_phase3_calls >= 1
    end

    @testset "Heuristic1 has no iteration-zero goal exit" begin
        n = 4
        B0 = zeros(BigInt, n, n)
        for i in 1:n
            B0[i, i] = big(1) << 20
        end
        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.heuristic1_reduce(
            B0; log_cond = 4, base_cutoff = 2,
            max_iterations = 20, telemetry = telemetry, validate = true)

        @test reduced == B0 * U
        @test info.iterations == 3
        @test info.stopped == :phase1_complete
        @test telemetry.h1_left_steps >= 1
        @test telemetry.h1_right_steps >= 1
        @test telemetry.h1_all_steps >= 1
    end

    @testset "external B2 keeps the exact joint transform" begin
        B0 = h1_rectangular_basis()
        B20 = BigInt[
              101   -17;
              -53    91;
              211   -37;
             -127    43;
               71   113;
              -19    29
        ]
        B = copy(B0)
        B2 = copy(B20)
        n = size(B, 2)
        U = Matrix{BigInt}(undef, n, n)
        U2 = Matrix{BigInt}(undef, n, size(B2, 2))
        log_cond = 20.0
        initial = Flatter._h1_initial_profile(n)
        goal = Flatter.goal_from_rhf(n, Flatter.DEFAULT_REDUCTION_RHF)
        telemetry = Flatter.ReductionTelemetry()

        Flatter._heuristic1_reduce!(B, U, B2, U2;
            goal = goal, split = Flatter.SplitPhase2(n), log_cond = log_cond,
            initial_profile = initial, profile_offset = zeros(Float64, n),
            max_iterations = 30, aggressive = false,
            schoenhage_threshold = Flatter.DEFAULT_SCHOENHAGE_THRESHOLD,
            base_cutoff = 2, blocksize = Flatter.DEFAULT_FUSED_BLOCKSIZE,
            panelsize = Flatter.DEFAULT_FUSED_PANELSIZE,
            validate = true, telemetry = telemetry, depth = 0)

        @test B == B0 * U
        @test B2 == B20 + B0 * U2
        @test abs(rr_det(U)) == 1
        @test telemetry.h1_relative_calls >= 1
    end

    @testset "dimension three uses left then all" begin
        B0 = BigInt[
            11  3  2;
             2 13  5;
             1  4 17;
             3 -1  2
        ]
        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.heuristic1_reduce(
            B0; log_cond = 12, base_cutoff = 2,
            max_iterations = 20, telemetry = telemetry, validate = true)

        @test reduced == B0 * U
        @test abs(rr_det(U)) == 1
        @test info.iterations == 2
        # Nested calls may add more left/all counters, but there can be no
        # right step anywhere in the top-level three-column schedule.
        @test telemetry.h1_left_steps >= 1
        @test telemetry.h1_all_steps >= 1
    end

    @testset "known-condition dense dispatcher selects Heuristic1" begin
        # Determinant one, genuinely dense, and comfortably conditioned for the
        # supplied 20-bit log-condition bound.
        B0 = BigInt[
            3 1  2 4;
            4 5  6 1;
            7 8 10 2;
            2 3  1 9
        ]
        @test Flatter.triangular_orientation(B0) === nothing
        telemetry = Flatter.ReductionTelemetry()

        reduced, U, info = Flatter.reduce_basis(
            B0; algorithm = :heuristic, log_cond = 20,
            base_cutoff = 2, max_iterations = 30,
            telemetry = telemetry, validate = true)

        @test rr_check_exact(B0, reduced, U)
        @test info.path === :dense
        @test telemetry.h1_calls >= 1
        @test telemetry.cond_calls == 0
    end

    @testset "zero log_cond retains CondUnknown" begin
        B0 = BigInt[3 1 2; 4 5 6; 7 8 10]
        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.reduce_basis(
            B0; algorithm = :heuristic, log_cond = 0,
            base_cutoff = 2, max_iterations = 20, telemetry = telemetry)

        @test rr_check_exact(B0, reduced, U)
        @test info.path === :dense
        @test telemetry.cond_calls == 1
        @test telemetry.h1_calls == 0
    end

    @testset "argument checking" begin
        B = h1_rectangular_basis()
        U = Matrix{BigInt}(undef, size(B, 2), size(B, 2))
        @test_throws ArgumentError Flatter.heuristic1_reduce!(B, U; log_cond = 0)
        @test_throws ArgumentError Flatter.heuristic1_reduce!(B, U; log_cond = Inf)
        @test_throws DimensionMismatch Flatter.heuristic1_reduce!(
            B, Matrix{BigInt}(undef, 3, 3); log_cond = 20)
        @test_throws ArgumentError Flatter.heuristic1_reduce!(
            B, U; log_cond = 20, _split = Flatter.SplitPhase3(size(B, 2)))
        @test_throws ArgumentError Flatter.reduce_basis(B; log_cond = -1)
    end
end
