# test/test_irregular.jl
#
# Assumes `using Test` and `using Random` from runtests.jl, and that
# test_recursive_reduction.jl has been included first — its `rr_` helpers are
# reused here.

# --------------------------------------------------------------------------

"A random unimodular matrix, as a product of integer transvections."
function ir_random_unimodular(rng, n::Int; operations::Int = 4n, magnitude::Int = 3)
    U = zeros(BigInt, n, n)
    for i in 1:n
        U[i, i] = big(1)
    end
    for _ in 1:operations
        a = rand(rng, 1:n)
        b = rand(rng, 1:n)
        a == b && continue
        c = rand(rng, (-magnitude):magnitude)
        iszero(c) && continue
        for k in 1:n
            U[k, a] += c * U[k, b]
        end
    end
    return U
end

@testset "irregular input" begin

    @testset "orientation detection" begin
        upper = BigInt[3 1 2; 0 5 4; 0 0 7]

        @testset "recognises all four corners" begin
            @test Flatter.triangular_orientation(upper) == (false, false)
            @test Flatter.triangular_orientation(reverse(upper; dims = 1)) == (true, false)
            @test Flatter.triangular_orientation(reverse(upper; dims = 2)) == (false, true)
            @test Flatter.triangular_orientation(reverse(reverse(upper; dims = 1); dims = 2)) ==
                  (true, true)
        end

        @testset "rejects a dense matrix" begin
            @test Flatter.triangular_orientation(BigInt[1 2; 3 4]) === nothing
        end

        @testset "rejects a zero on the diagonal" begin
            # Triangular in shape, but singular: the corner walk requires the
            # entry it reaches to be nonzero.
            @test Flatter.triangular_orientation(BigInt[0 1; 0 5]) === nothing
            @test Flatter.triangular_orientation(BigInt[3 1; 0 0]) === nothing
        end

        @testset "rejects a non-square matrix" begin
            @test Flatter.triangular_orientation(BigInt[1 0 0; 0 1 0]) === nothing
        end
    end

    @testset "flipping is its own inverse" begin
        rng = MersenneTwister(0xF11B)
        for n in (3, 6), flip_rows in (false, true), flip_columns in (false, true)
            M = BigInt[rand(rng, -20:20) for _ in 1:n, _ in 1:n]
            original = copy(M)
            Flatter._flip!(M, flip_rows, flip_columns)
            Flatter._flip!(M, flip_rows, flip_columns)
            @test M == original
        end
    end

    @testset "reoriented input" begin
        @testset "every orientation reduces correctly, n = $n" for n in (4, 6, 9)
            rng = MersenneTwister(hash((n, :orient)))
            upper = rr_triangular_basis(rng, n; spread = 40)

            for (flip_rows, flip_columns) in ((false, false), (true, false),
                                              (false, true), (true, true))
                B0 = copy(upper)
                Flatter._flip!(B0, flip_rows, flip_columns)

                reduced, U, info = Flatter.reduce_basis(B0)

                # The exact contract holds in the ORIGINAL orientation, which is
                # what makes flipping transparent to the caller.
                @test rr_check_exact(B0, reduced, U)
                @test info.path in (:triangular, :reoriented)
                @test (flip_rows || flip_columns) == (info.path === :reoriented)
            end
        end

        @testset "a lower triangular basis is accepted" begin
            rng = MersenneTwister(0x10E7)
            n = 6
            lower = permutedims(rr_triangular_basis(rng, n; spread = 30))
            reduced, U, info = Flatter.reduce_basis(lower)
            @test rr_check_exact(lower, reduced, U)
            @test info.path === :reoriented
        end
    end

    @testset "dense input" begin
        @testset "exact contract, n = $n" for n in (4, 6, 8, 12)
            rng = MersenneTwister(hash((n, :dense)))
            triangular = rr_triangular_basis(rng, n; spread = 40)
            scramble = ir_random_unimodular(rng, n)
            B0 = triangular * scramble

            @test Flatter.triangular_orientation(B0) === nothing   # genuinely dense

            reduced, U, info = Flatter.reduce_basis(B0)

            @test rr_check_exact(B0, reduced, U)
            @test info.path === :dense
            @test 1 <= info.rounds <= 4
        end

        @testset "a scrambled basis is brought back to comparable quality" begin
            # The lattice is the same as the triangular one it came from, so
            # reduction should recover something as short. This is the test that
            # the dense path actually works, as opposed to merely terminating.
            rng = MersenneTwister(0x5C4B)
            for n in (5, 8)
                triangular = rr_triangular_basis(rng, n; spread = 30)
                scrambled = triangular * ir_random_unimodular(rng, n)

                direct, _, _ = Flatter.reduce_basis(triangular)
                viadense, _, _ = Flatter.reduce_basis(scrambled)

                # Same lattice, so the determinants agree exactly.
                @test abs(rr_det(viadense)) == abs(rr_det(direct))
                # And the shortest vector found should be within a small factor.
                @test rr_shortest_norm2(viadense) <=
                      rr_shortest_norm2(direct) * big(2)^(2 * n)
            end
        end

        @testset "more rounds never break the contract" begin
            rng = MersenneTwister(0x0D57)
            n = 8
            B0 = rr_triangular_basis(rng, n; spread = 40) * ir_random_unimodular(rng, n)
            for rounds in (1, 2, 6)
                reduced, U, info = Flatter.reduce_basis(B0; max_rounds = rounds)
                @test rr_check_exact(B0, reduced, U)
                @test info.rounds <= rounds
            end
        end
    end

    @testset "to_integer_lattice" begin
        @testset "is upper triangular with no zero diagonal" begin
            setprecision(BigFloat, 200) do
                rng = MersenneTwister(0x1147)
                for n in (4, 8, 12)
                    B = Flatter.bigfloat_matrix(
                        rr_triangular_basis(rng, n; spread = 80), 200)
                    factors, _ = Flatter.householder_block(B)
                    L = Flatter.to_integer_lattice(factors, n)

                    for j in 1:n
                        @test !iszero(L[j, j])
                        for i in (j + 1):n
                            @test iszero(L[i, j])
                        end
                    end
                end
            end
        end

        @testset "preserves the profile shape" begin
            # The scale is uniform, so ratios between diagonal entries survive.
            setprecision(BigFloat, 256) do
                rng = MersenneTwister(0x1148)
                n = 8
                B = Flatter.bigfloat_matrix(rr_triangular_basis(rng, n; spread = 60), 256)
                factors, _ = Flatter.householder_block(B)
                L = Flatter.to_integer_lattice(factors, n)

                for i in 1:(n - 1)
                    before = Flatter._log2_abs(factors[i, i]) -
                             Flatter._log2_abs(factors[i + 1, i + 1])
                    after = Flatter._log2_abs(L[i, i]) - Flatter._log2_abs(L[i + 1, i + 1])
                    @test abs(before - after) < 1.0
                end
            end
        end

        @testset "reports a rank deficient factor" begin
            singular = zeros(BigFloat, 3, 3)
            singular[1, 1] = BigFloat(5)
            singular[2, 2] = BigFloat(0)
            singular[3, 3] = BigFloat(2)
            @test_throws ArgumentError Flatter.to_integer_lattice(singular, 3)
        end
    end

    @testset "argument checking" begin
        rng = MersenneTwister(0xBAD1)
        good = rr_triangular_basis(rng, 5; spread = 20)
        @test_throws DimensionMismatch Flatter.reduce_basis(BigInt[1 2 3; 0 4 5])
        @test_throws DimensionMismatch Flatter.reduce_basis!(
            copy(good), Matrix{BigInt}(undef, 3, 3))
        @test_throws ArgumentError Flatter.reduce_basis(Matrix{BigInt}(undef, 0, 0))
    end
end
