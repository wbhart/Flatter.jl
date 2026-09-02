# test_cond_unknown.jl
#
# Focused tests for flatter's CondUnknown phase-1 condition/rank discovery.
# Helpers from test_recursive_reduction.jl are available through runtests.jl.

@testset "CondUnknown" begin
    @testset "extract_similar precision gate" begin
        # After accepting e1, the second column has norm about 2^100 but an
        # orthogonal component of norm one.  At 53 bits flatter's
        # 2*spread+40 rule rejects it; at sufficiently high precision it is
        # resolved as independent.
        B = BigInt[1 big(1) << 100; 0 1]

        B53, P53, rank53, shift53, spread53 = Flatter._cond_extract_similar(B, 53)
        @test size(B53) == size(B)
        @test rank53 == 1
        @test Flatter._cond_permutation_sources(P53) == [1, 2]
        @test shift53 isa Int
        @test spread53 >= 0

        B256, P256, rank256, _, spread256 = Flatter._cond_extract_similar(B, 256)
        @test rank256 == 2
        @test Flatter._cond_permutation_sources(P256) == [1, 2]
        @test all(!iszero(B256[i, i]) for i in 1:2)
        @test spread256 > 90
    end

    @testset "unresolved columns move right while later valid columns move left" begin
        B = BigInt[
            1  big(1) << 100  0;
            0       1          0;
            0       0          1
        ]
        _, P, rank, _, _ = Flatter._cond_extract_similar(B, 53)
        @test rank == 2
        @test Flatter._cond_permutation_sources(P) == [1, 3, 2]
    end

    @testset "exact size sort puts zero columns at the end" begin
        B = BigInt[
            3  0  1  0;
            4  0  0  2;
            0  0  0  0;
            0  0  0  0
        ]
        P = Flatter._cond_sort_permutation(B)
        sources = Flatter._cond_permutation_sources(P)
        sorted = B * P
        norms = [sum(sorted[i, j]^2 for i in axes(sorted, 1)) for j in axes(sorted, 2)]

        @test sources == [3, 4, 1, 2]
        @test norms[1] <= norms[2] <= norms[3]
        @test iszero(norms[end])
    end

    @testset "surrogate dependent block uses Generic orthogonal relative SR" begin
        # CondUnknown does not mark this triangular reference specially in
        # upstream flatter: RelativeSizeReduction therefore takes the Generic
        # path, QR-factorises B1, and then runs the orthogonal kernel.
        B1 = BigInt[8 3; 0 5]
        B20 = BigInt[137 -91; 42 211]
        B2 = copy(B20)
        Urel = zeros(BigInt, 2, 2)

        Flatter._cond_generic_relative_sr!(B1, B2, Urel)

        @test B2 == B20 + B1 * Urel
        @test Flatter._cond_mpz_precision(B1) == Sys.WORD_SIZE
    end

    @testset "precision refinement preserves the exact transform" begin
        # Call CondUnknown directly so triangular orientation does not bypass
        # phase 1.  The first 53-bit pass resolves only one column; after the
        # exact relative reduction and a precision change the second resolves.
        B0 = BigInt[1 big(1) << 100; 0 1]
        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.cond_unknown_reduce(
            B0; base_cutoff = 2, max_iterations = 20,
            telemetry = telemetry, validate = true)

        @test rr_check_exact(B0, reduced, U)
        @test info.rank == 2
        @test telemetry.cond_calls == 1
        @test telemetry.cond_refinements >= 2
        @test telemetry.cond_precision_changes >= 1
        @test telemetry.cond_max_precision >= 53
    end

    @testset "full-rank dense input enters CondUnknown then phase 2" begin
        rng = MersenneTwister(0xC0D0)
        n = 6
        triangular = rr_triangular_basis(rng, n; spread = 35)
        B0 = triangular * ir_random_unimodular(rng, n; operations = 8n)
        @test Flatter.triangular_orientation(B0) === nothing

        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.reduce_basis(
            B0; algorithm = :heuristic, base_cutoff = 3,
            max_iterations = 30, telemetry = telemetry, validate = true)

        @test rr_check_exact(B0, reduced, U)
        @test info.path === :dense
        @test info.rank == n
        @test telemetry.cond_calls == 1
        @test telemetry.cond_refinements >= 1
        @test telemetry.h2_calls >= 1
    end

    @testset "rank-deficient dense input becomes an independent prefix plus zeros" begin
        # c3 = c1 + c2 exactly.  CondUnknown should discover rank two and drive
        # one column to exact zero rather than rejecting the input.
        B0 = BigInt[
            1  1  2;
            1  0  1;
            0  1  1
        ]
        @test Flatter.triangular_orientation(B0) === nothing

        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.reduce_basis(
            B0; algorithm = :heuristic, base_cutoff = 2,
            max_iterations = 20, telemetry = telemetry, validate = true)

        @test rr_check_exact(B0, reduced, U)
        @test info.path === :dense
        @test info.rank == 2
        @test all(iszero, reduced[:, 3])
        @test telemetry.cond_selected_rank == 2
    end

    @testset "rectangular full-column-rank input" begin
        B0 = BigInt[
            2  1  3;
            1  4  2;
            3  0  5;
            0  2  1;
            1  1  0
        ]
        telemetry = Flatter.ReductionTelemetry()
        reduced, U, info = Flatter.reduce_basis(
            B0; algorithm = :heuristic, base_cutoff = 2,
            max_iterations = 20, telemetry = telemetry, validate = true)

        @test rr_check_exact(B0, reduced, U)
        @test size(reduced) == size(B0)
        @test info.path === :dense
        @test info.rank == 3
        @test telemetry.cond_calls == 1
    end

    @testset "zero lattice is rank zero" begin
        B0 = zeros(BigInt, 4, 3)
        reduced, U, info = Flatter.cond_unknown_reduce(B0; base_cutoff = 2)
        @test rr_check_exact(B0, reduced, U)
        @test info.rank == 0
        @test isempty(info.profile)
        @test reduced == B0
    end
end
