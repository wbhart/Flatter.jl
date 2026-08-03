using LinearAlgebra
using Random

Random.seed!(0x5a17a55e)

@testset "Strassen" begin
    @test Flatter.strassen_workspace_length(1; cutoff = 1) == 0
    @test Flatter.strassen_workspace_length(2; cutoff = 1) == 3
    @test Flatter.strassen_workspace_length(3; cutoff = 1) == 15
    @test Flatter.strassen_workspace_length(5; cutoff = 1) == 42
    @test Flatter.strassen_workspace_length(7; cutoff = 1) == 63
    @test Flatter.strassen_workspace_length(8; cutoff = 1) == 63

    @test Flatter.strassen_workspace_length(
        1024;
        cutoff = 32,
    ) == 1024^2 - 32^2

    @testset "Int64, arbitrary square orders" begin
        for n in 1:17
            A = rand(-10:10, n, n)
            B = rand(-10:10, n, n)

            @test Flatter.strassen(A, B; cutoff = 1) == A * B
        end
    end

    @testset "Float64, arbitrary square orders" begin
        for n in 1:17
            A = randn(n, n)
            B = randn(n, n)
            C = Flatter.strassen(A, B; cutoff = 2)

            @test isapprox(C, A * B; rtol = 2e-11, atol = 2e-11)
        end
    end

    @testset "BigInt, including odd orders" begin
        for n in (1, 2, 3, 5, 7, 9, 16, 17, 33)
            A = BigInt.(rand(-100:100, n, n))
            B = BigInt.(rand(-100:100, n, n))

            @test Flatter.strassen(A, B; cutoff = 4) == A * B
        end
    end

    @testset "Preallocated output and workspace" begin
        for n in (15, 16, 17, 31, 32, 33)
            A = rand(-5:5, n, n)
            B = rand(-5:5, n, n)
            C = Matrix{Int}(undef, n, n)

            work = Vector{Int}(
                undef,
                Flatter.strassen_workspace_length(n; cutoff = 4),
            )

            result = Flatter.strassen!(
                C,
                A,
                B;
                cutoff = 4,
                workspace = work,
            )

            @test result === C
            @test C == A * B
        end
    end

    @testset "Rectangular fallback and promotion" begin
        A = rand(-5:5, 7, 5)
        B = rand(-5:5, 5, 9)

        @test Flatter.strassen(A, B; cutoff = 1) == A * B

        Ai = rand(-5:5, 9, 9)
        Bb = BigInt.(rand(-5:5, 9, 9))
        C = Flatter.strassen(Ai, Bb; cutoff = 1)

        @test eltype(C) == BigInt
        @test C == Ai * Bb
    end

    @testset "Validation and alias rejection" begin
        A = rand(4, 4)
        B = rand(4, 4)

        @test_throws ArgumentError Flatter.strassen!(
            A,
            A,
            B;
            cutoff = 1,
        )

        @test_throws ArgumentError Flatter.strassen!(
            B,
            A,
            B;
            cutoff = 1,
        )

        @test_throws ArgumentError Flatter.strassen(
            A,
            B;
            cutoff = 0,
        )

        @test_throws DimensionMismatch Flatter.strassen(
            rand(3, 4),
            rand(5, 3),
        )

        C = Matrix{Float64}(undef, 4, 4)

        short_work = Vector{Float64}(
            undef,
            Flatter.strassen_workspace_length(4; cutoff = 1) - 1,
        )

        @test_throws DimensionMismatch Flatter.strassen!(
            C,
            A,
            B;
            cutoff = 1,
            workspace = short_work,
        )
    end
end
