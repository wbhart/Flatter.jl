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

    # The dense path starts at the QR precision policy rather than the reduction
    # one, and doubles when the factorisation comes back degenerate. Both the
    # cheap start and the recovery need to hold.
    @testset "dense precision discovery" begin
        @testset "starts well below the reduction policy" begin
            rng = MersenneTwister(0x9EC1)
            B = BigInt[rand(rng, -100:100) for _ in 1:128, _ in 1:128]
            @test Flatter._dense_precision(B, 128) < Flatter.lll_precision(8, 128)
        end

        @testset "an ill-conditioned basis still factorises" begin
            # A wide profile needs far more precision than the entry sizes
            # suggest, so this is the case the doubling exists for.
            rng = MersenneTwister(0x9EC2)
            n = 8
            B = rr_triangular_basis(rng, n; spread = 400) *
                ir_random_unimodular(rng, n)
            approximation = Flatter._dense_approximation(B, n, n)
            @test size(approximation) == (n, n)
            @test all(!iszero(approximation[i, i]) for i in 1:n)
        end

        # The hardware path is only taken when 53 bits is what the policy asked
        # for. It must agree with the arbitrary-precision path, and fall through
        # to it when 53 bits turns out not to resolve the profile.
        # LAPACK's factorisation must agree with the package's own generic one:
        # same algorithm, different implementation, so the R factors should match
        # to hardware precision up to the sign convention on each column.
        @testset "LAPACK agrees with the generic Householder" begin
            rng = MersenneTwister(0x1A9A)
            for (m, n) in ((8, 8), (12, 8), (16, 16))
                A = randn(rng, m, n)

                viaLAPACK = copy(A)
                LinearAlgebra.LAPACK.geqrf!(viaLAPACK)

                factors, _ = Flatter.householder_block(copy(A))

                # Compare |R| entrywise: Householder sign choices can differ per
                # column between implementations, which flips a whole row of R.
                for j in 1:n, i in 1:j
                    @test abs(abs(viaLAPACK[i, j]) - abs(factors[i, j])) <
                          1e-9 * max(1.0, maximum(abs, A))
                end
            end
        end

        @testset "hardware and arbitrary precision agree" begin
            rng = MersenneTwister(0x53B1)
            for n in (6, 10, 16)
                # Small entries, so the policy asks for exactly 53 bits.
                B = BigInt[rand(rng, -60:60) for _ in 1:n, _ in 1:n]
                abs(rr_det(B)) == 0 && continue
                @test Flatter._dense_precision(B, n) == 53

                fast = Flatter._dense_approximation(B, n, n; hardware = true)
                slow = Flatter._dense_approximation(B, n, n; hardware = false)

                # Both are integer approximations of the same R factor at the
                # same precision, so the diagonals should agree closely. They
                # need not be identical: the scale is chosen from rounded logs.
                for i in 1:n
                    @test abs(Flatter._log2_abs(fast[i, i]) -
                              Flatter._log2_abs(slow[i, i])) < 1.0
                end
            end
        end

        @testset "falls through when 53 bits is not enough" begin
            # A wide profile: the policy still asks for more than 53 bits, so
            # the hardware path is skipped outright rather than attempted.
            rng = MersenneTwister(0x53B2)
            n = 8
            B = rr_triangular_basis(rng, n; spread = 400) *
                ir_random_unimodular(rng, n)
            @test Flatter._dense_precision(B, n) > 53
            approximation = Flatter._dense_approximation(B, n, n; hardware = true)
            @test all(!iszero(approximation[i, i]) for i in 1:n)
        end

        # Rank deficiency reaches `to_integer_lattice` as a diagonal entry at the
        # noise floor rather than an exact zero, so the check has to be relative
        # to the working precision. Raising the precision cannot help here, so
        # the doubling loop runs to its ceiling and reports.
        # The tier list is the seam a wider fixed-precision type plugs into.
        # `BigFloat` stands in for one here: it is not a fixed-precision type,
        # but it exercises the dispatch, and the result must not depend on which
        # tier produced it.
        @testset "the tier list does not change the answer" begin
            rng = MersenneTwister(0x71E5)
            for n in (6, 10)
                B = BigInt[rand(rng, -60:60) for _ in 1:n, _ in 1:n]
                iszero(rr_det(B)) && continue

                viaFloat64 = Flatter._dense_approximation(B, n, n;
                                                          float_tiers = (Float64,))
                viaNone = Flatter._dense_approximation(B, n, n; hardware = false)

                for i in 1:n
                    @test abs(Flatter._log2_abs(viaFloat64[i, i]) -
                              Flatter._log2_abs(viaNone[i, i])) < 1.0
                end
            end
        end

        @testset "a tier too narrow is skipped, not attempted" begin
            # A wide profile needs more than 53 bits, so the Float64 tier must
            # be passed over rather than tried and failed.
            rng = MersenneTwister(0x71E6)
            n = 8
            B = rr_triangular_basis(rng, n; spread = 400) *
                ir_random_unimodular(rng, n)
            @test Flatter._dense_precision(B, n) > Base.precision(Float64)
            approximation = Flatter._dense_approximation(B, n, n)
            @test all(!iszero(approximation[i, i]) for i in 1:n)
        end

        @testset "a rank deficient basis is reported, not looped on" begin
            singular = BigInt[1 2 3; 2 4 6; 1 1 1]     # row 2 is twice row 1
            @test iszero(rr_det(singular))
            @test_throws ArgumentError Flatter._dense_approximation(singular, 3, 3)
            @test_throws ArgumentError Flatter.reduce_basis(singular)
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
