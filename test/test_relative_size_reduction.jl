# test/test_relative_size_reduction.jl
#
# Assumes `using Test` and `using Random` from runtests.jl.
#
# IMPORTANT: this file uses `smsv_gso` from test_smsv.jl as its orthogonality
# oracle, so it must be included AFTER that file in runtests.jl.
#
# Three invariants are checked, in decreasing order of strength:
#
#   1. B2_new == B2_old + B1*U, exactly, over BigInt. True on every path.
#   2. Reduction quality. Exactly 1/2 on the triangular path; roughly the
#      deadband on the orthogonal paths, since refinement stops once the
#      multipliers fall below unit magnitude.
#   3. R2 agrees with the coordinates recomputed from the exact reduced B2.

# --------------------------------------------------------------------------
# Generators and oracles
# --------------------------------------------------------------------------

function rsr_random_bigint(rng, bits::Int)
    x = big(0)
    for _ in 1:cld(bits, 32)
        x = (x << 32) + rand(rng, 0:(2^32 - 1))
    end
    x >>= max(0, 32 * cld(bits, 32) - bits)
    return rand(rng, Bool) ? x : -x
end

rsr_random_matrix(rng, m::Int, n::Int; bits::Int = 40) =
    BigInt[rsr_random_bigint(rng, bits) for _ in 1:m, _ in 1:n]

function rsr_random_triu(rng, n::Int; bits::Int = 40, diagonal_bits::Int = 48)
    B = zeros(BigInt, n, n)
    for j in 1:n
        for i in 1:(j - 1)
            B[i, j] = rsr_random_bigint(rng, bits)
        end
        d = big(0)
        while iszero(d)
            d = rsr_random_bigint(rng, diagonal_bits)
        end
        B[j, j] = d
    end
    return B
end

"B2_new == B2_old + B1*U, checked in BigInt so an Int64 overflow cannot hide."
rsr_identity_holds(B1, B2_old, B2_new, U) =
    BigInt.(B2_new) == BigInt.(B2_old) + BigInt.(B1) * BigInt.(U)

# For an upper triangular B1 the Gram-Schmidt vectors are b*_j = B1[j,j]*e_j,
# so the coefficient of b*_j in a column v is exactly v[j]/B1[j,j] and the
# reduction condition is 2|v[j]| <= |B1[j,j]|. No GSO needed, and none possible:
# [B1 B2] is n1 x (n1+n2) and so rank deficient.
function rsr_triangular_quality(B1, B2)
    n1, n2 = size(B1, 1), size(B2, 2)
    worst = 0 // 1
    for j in 1:n2, row in 1:n1
        worst = max(worst, abs(Rational{BigInt}(B2[row, j]) // B1[row, row]))
    end
    return worst
end

# The coefficients of b*_j, j <= n1, in each column of B2. Only these are
# constrained: relative size reduction never reduces B2's columns against each
# other, so mu[j, n1+c] for j > n1 is unconstrained and must not be asserted on.
#
# Needs m >= n1 + n2 for [B1 B2] to have full column rank.
function rsr_relative_mu(B1, B2)
    n1, n2 = size(B1, 2), size(B2, 2)
    combined = Matrix{Rational{BigInt}}(hcat(BigInt.(B1), BigInt.(B2)))
    _, mu, _ = smsv_gso(combined)
    return [mu[j, n1 + c] for j in 1:n1, c in 1:n2]
end

rsr_worst_mu(B1, B2) =
    isempty(B1) ? 0 // 1 : maximum(abs, rsr_relative_mu(B1, B2))

rsr_maxabs(A) = isempty(A) ? zero(eltype(A)) : maximum(abs, A)

# --------------------------------------------------------------------------

@testset "relative_size_reduction" begin

    @testset "triangular kernel" begin

        @testset "contract, n1 = $n1, n2 = $n2" for n1 in (1, 2, 5, 11), n2 in (1, 3, 7)
            rng = MersenneTwister(hash((n1, n2, :tri)))
            B1 = rsr_random_triu(rng, n1)
            B2 = rsr_random_matrix(rng, n1, n2; bits = 70)
            original = copy(B2)

            reduced, U, R2 = Flatter.relative_size_reduction(B1, B2; triangular = true)

            @test B2 == original                       # input untouched
            @test rsr_identity_holds(B1, original, reduced, U)
            @test rsr_triangular_quality(B1, reduced) <= 1 // 2
            @test R2 == reduced
            @test size(U) == (n1, n2)
        end

        @testset "negative diagonal entries" begin
            B1 = BigInt[-97 400 -1200; 0 61 900; 0 0 -53]
            B2 = BigInt[5000 -8000; -3000 12000; 700 -400]
            reduced, U, _ = Flatter.relative_size_reduction(B1, B2; triangular = true)
            @test rsr_identity_holds(B1, B2, reduced, U)
            @test rsr_triangular_quality(B1, reduced) <= 1 // 2
        end

        @testset "new_shift only rescales the output" begin
            rng = MersenneTwister(0x5417)
            B1 = rsr_random_triu(rng, 8)
            B2 = rsr_random_matrix(rng, 8, 5; bits = 70)

            base, base_U, _ = Flatter.relative_size_reduction(B1, B2; triangular = true)

            for shift in (1, 3, 17)
                shifted, shifted_U, _ =
                    Flatter.relative_size_reduction(B1, B2; triangular = true,
                                                    new_shift = shift)
                # Each entry is rescaled once, after it stops changing, so the
                # reduction itself is unaffected.
                @test shifted == (base .>> shift)
                @test shifted_U == base_U
            end

            for shift in (-1, -9)
                shifted, shifted_U, _ =
                    Flatter.relative_size_reduction(B1, B2; triangular = true,
                                                    new_shift = shift)
                @test shifted == (base .<< (-shift))
                @test shifted_U == base_U
            end
        end

        @testset "Int64 element type" begin
            rng = MersenneTwister(0x6411)
            B1 = Int64[101 30 -12; 0 -87 44; 0 0 73]
            B2 = Int64[rand(rng, -900:900) for _ in 1:3, _ in 1:4]
            reduced, U, R2 = Flatter.relative_size_reduction(B1, B2; triangular = true)
            @test eltype(reduced) === Int64 && eltype(U) === Int64
            @test rsr_identity_holds(B1, B2, reduced, U)
            @test rsr_triangular_quality(B1, reduced) <= 1 // 2
            @test R2 == reduced
        end
    end

    @testset "orthogonal kernel, supplied BigFloat factors" begin
        bits = 256

        @testset "contract, m = $m, n1 = $n1, n2 = $n2" for (m, n1, n2) in
                ((14, 5, 4), (20, 8, 6), (9, 3, 3), (12, 6, 1))
            rng = MersenneTwister(hash((m, n1, n2, :orth)))
            B1 = rsr_random_matrix(rng, m, n1; bits = 30)
            B2 = rsr_random_matrix(rng, m, n2; bits = 60)
            original = copy(B2)

            factors, wy = setprecision(BigFloat, bits) do
                Flatter.householder_block(Flatter.bigfloat_matrix(B1, bits))
            end
            @test Flatter.uniform_precision(factors) == bits

            reduced, U, R2 = Flatter.relative_size_reduction(
                B1, B2; factors = factors, compact_T = wy)

            @test rsr_identity_holds(B1, original, reduced, U)
            @test rsr_worst_mu(B1, reduced) <= 51 // 100 + 1 // 1000
            @test Flatter.uniform_precision(R2) == bits
        end

        @testset "R2 matches coordinates recomputed from the exact result" begin
            bits = 256
            rng = MersenneTwister(0x8207)
            m, n1, n2 = 16, 6, 5
            B1 = rsr_random_matrix(rng, m, n1; bits = 30)
            B2 = rsr_random_matrix(rng, m, n2; bits = 60)

            factors, wy = setprecision(BigFloat, bits) do
                Flatter.householder_block(Flatter.bigfloat_matrix(B1, bits))
            end
            reduced, _, R2 = Flatter.relative_size_reduction(
                B1, B2; factors = factors, compact_T = wy)

            setprecision(BigFloat, bits) do
                fresh = Flatter.bigfloat_matrix(reduced, bits)
                Flatter.apply_Qt!(fresh, factors, wy; reflectors = n1)
                scale = max(one(BigFloat), rsr_maxabs(fresh))
                # R2 comes from incremental updates during the final pass;
                # `fresh` is recomputed from scratch. They differ only by the
                # float error of one pass.
                @test rsr_maxabs(R2 - fresh) <= scale * BigFloat(2)^(-bits + 40)
            end
        end

        # Refinement can only reduce as far as the coordinates can be known.
        # Converting B2 to a p-bit float has absolute error about |B2|*2^-p,
        # apply_Qt! preserves it, so a quotient is known only to
        # |B2| * 2^-p / |R_jj|. Reduction therefore needs
        #
        #     p > log2( max|B2| / min|R_jj| )
        #
        # and crucially |B2| here is the magnitude AFTER reduction, since only
        # the component in B1's span can shrink.
        @testset "refinement converges when B2 is nearly in B1's span" begin
            # B2 = B1*X + E with X huge and E small, so reduction really does
            # shrink B2 and each pass sees more accurate coordinates than the
            # last. This is the regime the refinement loop exists for.
            bits = 128
            rng = MersenneTwister(0x9E11)
            m, n1, n2 = 15, 6, 4
            B1 = rsr_random_matrix(rng, m, n1; bits = 25)
            X = rsr_random_matrix(rng, n1, n2; bits = 300)
            E = rsr_random_matrix(rng, m, n2; bits = 25)
            B2 = B1 * X + E

            factors, wy = setprecision(BigFloat, bits) do
                Flatter.householder_block(Flatter.bigfloat_matrix(B1, bits))
            end
            reduced, U, _ = Flatter.relative_size_reduction(
                B1, B2; factors = factors, compact_T = wy)

            @test rsr_identity_holds(B1, B2, reduced, U)
            @test rsr_worst_mu(B1, reduced) <= 51 // 100 + 1 // 1000
            # From ~2^328 down to the size of E. Nothing like this is reachable
            # in a single pass at 128 bits.
            @test rsr_maxabs(reduced) < rsr_maxabs(B2) >> 200

            # And prove that more than one pass was in fact needed.
            @test_throws ErrorException Flatter.relative_size_reduction(
                B1, B2; factors = factors, compact_T = wy, max_passes = 1)
        end

        @testset "insufficient precision stops instead of looping" begin
            # B2 has a large component ORTHOGONAL to B1, which reduction cannot
            # shrink, so the coordinate error floor stays at |B2|*2^-p however
            # many passes are taken. The convergence guard must notice that the
            # multipliers have stopped shrinking and give up; without it this
            # would run to the pass cap. Raising the precision above the ratio
            # of the two blocks fixes it.
            rng = MersenneTwister(0x9E12)
            m, n1, n2 = 15, 6, 4
            B1 = rsr_random_matrix(rng, m, n1; bits = 20)
            B2 = rsr_random_matrix(rng, m, n2; bits = 400)

            function reduce_at(bits)
                factors, wy = setprecision(BigFloat, bits) do
                    Flatter.householder_block(Flatter.bigfloat_matrix(B1, bits))
                end
                return Flatter.relative_size_reduction(
                    B1, B2; factors = factors, compact_T = wy)
            end

            # 400 - 20 = 380 bits are needed; 300 is not enough.
            starved, starved_U, _ = reduce_at(300)
            ample, ample_U, _ = reduce_at(600)

            # The transform is exact on both, whatever the reduction quality.
            @test rsr_identity_holds(B1, B2, starved, starved_U)
            @test rsr_identity_holds(B1, B2, ample, ample_U)

            @test rsr_worst_mu(B1, ample) <= 51 // 100 + 1 // 1000
            @test rsr_worst_mu(B1, starved) > rsr_worst_mu(B1, ample)
        end
    end

    @testset "orthogonal kernel, supplied Float64 factors" begin
        @testset "contract" begin
            rng = MersenneTwister(0xF64)
            m, n1, n2 = 14, 5, 4
            B1 = BigInt[rand(rng, -60:60) for _ in 1:m, _ in 1:n1]
            B2 = BigInt[rand(rng, -5000:5000) for _ in 1:m, _ in 1:n2]

            factors, wy = Flatter.householder_block(Float64.(B1))
            reduced, U, R2 = Flatter.relative_size_reduction(
                B1, B2; factors = factors, compact_T = wy)

            @test eltype(R2) === Float64
            @test rsr_identity_holds(B1, B2, reduced, U)
            # Looser than the BigFloat case: 53 bits, and the deadband.
            @test rsr_worst_mu(B1, reduced) <= 55 // 100
        end

        @testset "entries too large for Float64 are rejected, not silently Inf" begin
            rng = MersenneTwister(0xF65)
            m, n1 = 10, 4
            B1 = BigInt[rand(rng, -60:60) for _ in 1:m, _ in 1:n1]
            B2 = fill(big(2)^2000, m, 2)

            factors, wy = Flatter.householder_block(Float64.(B1))
            @test_throws ArgumentError Flatter.relative_size_reduction(
                B1, B2; factors = factors, compact_T = wy)
        end
    end

    @testset "generic kernel" begin
        @testset "contract" begin
            rng = MersenneTwister(0x9E1C)
            m, n1, n2 = 18, 7, 6
            B1 = rsr_random_matrix(rng, m, n1; bits = 30)
            B2 = rsr_random_matrix(rng, m, n2; bits = 60)

            reduced, U, R2 = Flatter.relative_size_reduction(B1, B2)

            @test rsr_identity_holds(B1, B2, reduced, U)
            @test rsr_worst_mu(B1, reduced) <= 51 // 100 + 1 // 1000
            @test R2 isa Matrix{BigFloat}
        end

        @testset "agrees with supplying the same factorization explicitly" begin
            rng = MersenneTwister(0x9E1D)
            m, n1, n2 = 15, 6, 5
            B1 = rsr_random_matrix(rng, m, n1; bits = 30)
            B2 = rsr_random_matrix(rng, m, n2; bits = 60)
            bits = 200

            implicit, U_implicit, _ =
                Flatter.relative_size_reduction(B1, B2; precision = bits)

            factors, wy = setprecision(BigFloat, bits) do
                Flatter.householder_block(Flatter.bigfloat_matrix(B1, bits))
            end
            explicit, U_explicit, _ = Flatter.relative_size_reduction(
                B1, B2; factors = factors, compact_T = wy)

            @test implicit == explicit
            @test U_implicit == U_explicit
        end

        @testset "coordinates = false skips R2" begin
            rng = MersenneTwister(0x9E1E)
            B1 = rsr_random_matrix(rng, 12, 4; bits = 30)
            B2 = rsr_random_matrix(rng, 12, 3; bits = 50)
            reduced, U, R2 = Flatter.relative_size_reduction(B1, B2;
                                                             coordinates = false)
            @test R2 === nothing
            @test rsr_identity_holds(B1, B2, reduced, U)
        end
    end

    @testset "in-place form and edge cases" begin
        rng = MersenneTwister(0xEDDA)

        @testset "mutates its arguments and returns them" begin
            B1 = rsr_random_triu(rng, 6)
            B2 = rsr_random_matrix(rng, 6, 4; bits = 60)
            original = copy(B2)
            U = Matrix{BigInt}(undef, 6, 4)

            returned, returned_U, _ =
                Flatter.relative_size_reduction!(B1, B2, U; triangular = true)
            @test returned === B2
            @test returned_U === U
            @test rsr_identity_holds(B1, original, B2, U)
        end

        @testset "empty B2" begin
            B1 = rsr_random_matrix(rng, 8, 3; bits = 30)
            B2 = Matrix{BigInt}(undef, 8, 0)
            reduced, U, R2 = Flatter.relative_size_reduction(B1, B2)
            @test size(reduced) == (8, 0)
            @test size(U) == (3, 0)
            @test size(R2) == (8, 0)
        end

        @testset "an already reduced B2 is a fixed point" begin
            B1 = BigInt[1000 0 0; 0 1000 0; 0 0 1000]
            B2 = BigInt[7 -3; 11 4; -9 2]
            reduced, U, _ = Flatter.relative_size_reduction(B1, B2; triangular = true)
            @test reduced == B2
            @test all(iszero, U)
        end
    end

    @testset "argument checking" begin
        rng = MersenneTwister(0xBAD)
        B1 = rsr_random_matrix(rng, 10, 4; bits = 30)
        B2 = rsr_random_matrix(rng, 10, 3; bits = 30)
        factors, wy = Flatter.householder_block(Float64.(B1))

        @test_throws DimensionMismatch Flatter.relative_size_reduction(
            B1, rsr_random_matrix(rng, 9, 3; bits = 30))
        @test_throws DimensionMismatch Flatter.relative_size_reduction!(
            B1, copy(B2), Matrix{BigInt}(undef, 3, 3))
        @test_throws ArgumentError Flatter.relative_size_reduction(
            B1, B2; factors = factors)
        @test_throws ArgumentError Flatter.relative_size_reduction(
            B1, B2; factors = factors, compact_T = wy, new_shift = 3)
        @test_throws ArgumentError Flatter.relative_size_reduction(
            B1, B2; algorithm = :nope)
        @test_throws ArgumentError Flatter.relative_size_reduction(
            B1, B2; deadband = 0.0)

        # Not square, and not triangular.
        @test_throws DimensionMismatch Flatter.relative_size_reduction(
            B1, B2; triangular = true)
        square = rsr_random_matrix(rng, 5, 5; bits = 30)
        @test_throws ArgumentError Flatter.relative_size_reduction(
            square, rsr_random_matrix(rng, 5, 2; bits = 30); triangular = true)
        singular = BigInt[3 1; 0 0]
        @test_throws ArgumentError Flatter.relative_size_reduction(
            singular, reshape(BigInt[5, 7], 2, 1); triangular = true)
    end
end
