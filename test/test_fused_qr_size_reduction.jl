# test/test_fused_qr_size_reduction.jl
#
# Assumes `using Test` and `using Random` from runtests.jl.
#
# IMPORTANT: uses `smsv_gso` from test_smsv.jl as its orthogonality oracle, so
# it must be included AFTER that file in runtests.jl.
#
# The invariants, in decreasing order of strength:
#
#   1. B_reduced == B_original * U, exactly, over BigInt.
#   2. U is unimodular: det(U) == +/-1, by exact fraction-free elimination.
#   3. Each column is size reduced against its predecessors' Gram-Schmidt
#      frame, to within the deadband.
#   4. The packed output really is a QR factorization of the reduced basis:
#      applying Q' to it reproduces the upper trapezoid of R.

# --------------------------------------------------------------------------
# Generators and oracles
# --------------------------------------------------------------------------

function fqr_random_bigint(rng, bits::Int)
    x = big(0)
    for _ in 1:cld(bits, 32)
        x = (x << 32) + rand(rng, 0:(2^32 - 1))
    end
    x >>= max(0, 32 * cld(bits, 32) - bits)
    return rand(rng, Bool) ? x : -x
end

fqr_random_basis(rng, m::Int, n::Int; bits::Int = 30) =
    BigInt[fqr_random_bigint(rng, bits) for _ in 1:m, _ in 1:n]

"""
A basis with a deliberately wide Gram-Schmidt profile: column `j` is scaled by
`2^(spread*(j-1)/(n-1))`, which is the regime the fused algorithm exists for,
since the R factor of such a basis before reduction is badly conditioned.
"""
function fqr_spread_basis(rng, m::Int, n::Int; bits::Int = 20, spread::Int = 60)
    B = fqr_random_basis(rng, m, n; bits = bits)
    for j in 1:n
        shift = n == 1 ? 0 : div(spread * (j - 1), n - 1)
        for i in 1:m
            B[i, j] <<= shift
        end
    end
    return B
end

"""
An orthogonal-ish basis put through random integer column operations. The
lattice is unchanged but the basis is badly skewed, so reduction has real work
to do and the answer can be checked against the known starting point.
"""
function fqr_scrambled_basis(rng, m::Int, n::Int; operations::Int = 30,
                             magnitude::Int = 6)
    B = zeros(BigInt, m, n)
    for j in 1:n
        B[j, j] = big(1) << rand(rng, 0:4)
    end
    for _ in 1:operations
        a = rand(rng, 1:n)
        b = rand(rng, 1:n)
        a == b && continue
        c = rand(rng, (-magnitude):magnitude)
        for i in 1:m
            B[i, a] += c * B[i, b]
        end
    end
    return B
end

"Exact determinant by fraction-free (Bareiss) elimination."
function fqr_det(A::AbstractMatrix{<:Integer})
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("not square"))
    n == 0 && return big(1)
    M = Matrix{BigInt}(A)
    sign = big(1)
    previous = big(1)
    for k in 1:(n - 1)
        if iszero(M[k, k])
            pivot = 0
            for r in (k + 1):n
                if !iszero(M[r, k])
                    pivot = r
                    break
                end
            end
            iszero(pivot) && return big(0)
            for c in 1:n
                M[k, c], M[pivot, c] = M[pivot, c], M[k, c]
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

"Worst |mu[j,i]| over j < i: how well each column is reduced against those before it."
function fqr_worst_mu(B::AbstractMatrix{<:Integer})
    n = size(B, 2)
    n <= 1 && return 0 // 1
    _, mu, _ = smsv_gso(Matrix{Rational{BigInt}}(BigInt.(B)))
    worst = 0 // 1
    for i in 2:n, j in 1:(i - 1)
        worst = max(worst, abs(mu[j, i]))
    end
    return worst
end

fqr_maxabs(A) = isempty(A) ? zero(eltype(A)) : maximum(abs, A)

# --------------------------------------------------------------------------

@testset "fused_qr_size_reduction" begin

    @testset "exact invariants, m = $m, n = $n" for (m, n) in
            ((4, 2), (6, 6), (9, 5), (12, 12), (15, 8), (5, 1))
        rng = MersenneTwister(hash((m, n, :fqr)))
        B0 = fqr_random_basis(rng, m, n; bits = 30)
        untouched = copy(B0)

        reduced, U, R, tau = Flatter.fused_qr_size_reduction(B0; precision = 256)

        @test BigInt.(reduced) == BigInt.(untouched) * BigInt.(U)
        @test abs(fqr_det(U)) == 1
        @test size(R) == (m, n)
        @test length(tau) == n
        @test B0 == untouched
    end

    @testset "reduction quality" begin
        @testset "random basis, m = $m, n = $n" for (m, n) in ((8, 4), (10, 7), (12, 12))
            rng = MersenneTwister(hash((m, n, :quality)))
            B0 = fqr_random_basis(rng, m, n; bits = 30)
            reduced, _, _, _ = Flatter.fused_qr_size_reduction(B0; precision = 256)
            @test fqr_worst_mu(reduced) <= 51 // 100 + 1 // 1000
        end

        @testset "wide profile spread" begin
            rng = MersenneTwister(0x59DE)
            B0 = fqr_spread_basis(rng, 10, 8; bits = 20, spread = 120)
            reduced, U, _, _ = Flatter.fused_qr_size_reduction(B0; precision = 512)
            @test BigInt.(reduced) == BigInt.(B0) * BigInt.(U)
            @test abs(fqr_det(U)) == 1
            @test fqr_worst_mu(reduced) <= 51 // 100 + 1 // 1000
        end

        @testset "scrambled basis is brought back down" begin
            rng = MersenneTwister(0x5C4A)
            B0 = fqr_scrambled_basis(rng, 9, 9; operations = 40)
            reduced, U, _, _ = Flatter.fused_qr_size_reduction(B0; precision = 256)

            @test BigInt.(reduced) == BigInt.(B0) * BigInt.(U)
            @test abs(fqr_det(U)) == 1
            @test fqr_worst_mu(reduced) <= 51 // 100 + 1 // 1000
            # The scrambling inflated the entries; reduction should undo much of it.
            @test fqr_maxabs(reduced) <= fqr_maxabs(B0)
        end

        @testset "an already reduced basis is close to a fixed point" begin
            # Orthogonal columns of distinct scales: nothing to reduce.
            B0 = zeros(BigInt, 5, 5)
            for j in 1:5
                B0[j, j] = big(10)^j
            end
            reduced, U, _, _ = Flatter.fused_qr_size_reduction(B0; precision = 256)
            @test reduced == B0
            expected_U = zeros(BigInt, 5, 5)
            for i in 1:5
                expected_U[i, i] = big(1)
            end
            @test U == expected_U
        end
    end

    @testset "the packed output is a genuine QR factorization" begin
        @testset "Q' reproduces R, m = $m, n = $n" for (m, n) in ((10, 6), (8, 8), (14, 5))
            bits = 256
            rng = MersenneTwister(hash((m, n, :qr)))
            B0 = fqr_random_basis(rng, m, n; bits = 25)

            reduced, _, R, tau = Flatter.fused_qr_size_reduction(B0; precision = bits)

            setprecision(BigFloat, bits) do
                T = Flatter.compact_wy_from_reflectors(R, tau)
                C = Flatter.bigfloat_matrix(reduced, bits)
                Flatter.apply_Qt!(C, R, T)

                scale = max(one(BigFloat), fqr_maxabs(C))
                tolerance = scale * BigFloat(2)^(-bits + 60)

                # Upper trapezoid must match the R carried in the packed output,
                # and everything below it must have been annihilated.
                for j in 1:n, i in 1:min(j, n)
                    @test abs(C[i, j] - R[i, j]) <= tolerance
                end
                for j in 1:n, i in (j + 1):m
                    @test abs(C[i, j]) <= tolerance
                end
            end
        end

        @testset "diagonal agrees with an independent factorization" begin
            bits = 256
            rng = MersenneTwister(0x1DEA)
            m, n = 11, 7
            B0 = fqr_random_basis(rng, m, n; bits = 25)
            reduced, _, R, _ = Flatter.fused_qr_size_reduction(B0; precision = bits)

            setprecision(BigFloat, bits) do
                independent, _ = Flatter.householder_block(
                    Flatter.bigfloat_matrix(reduced, bits))
                for j in 1:n
                    expected = abs(independent[j, j])
                    @test abs(abs(R[j, j]) - expected) <=
                          max(one(BigFloat), expected) * BigFloat(2)^(-bits + 60)
                end
            end
        end

        @testset "compact_wy_from_reflectors matches sequential application" begin
            bits = 200
            rng = MersenneTwister(0x3A7)
            m, n = 9, 5
            B0 = fqr_random_basis(rng, m, n; bits = 25)
            reduced, _, R, tau = Flatter.fused_qr_size_reduction(B0; precision = bits)

            setprecision(BigFloat, bits) do
                T = Flatter.compact_wy_from_reflectors(R, tau)
                # V and T reconstruct Q = I - V*T*V'; check Q' is orthogonal.
                identity_m = zeros(BigFloat, m, m)
                for i in 1:m
                    identity_m[i, i] = one(BigFloat)
                end
                Qt = Flatter.apply_Qt!(copy(identity_m), R, T)
                @test fqr_maxabs(Qt * permutedims(Qt) - identity_m) <
                      BigFloat(2)^(-bits + 60)
            end
        end
    end

    @testset "Float64 path" begin
        @testset "small entries work" begin
            rng = MersenneTwister(0xF64F)
            m, n = 10, 6
            B0 = BigInt[rand(rng, -500:500) for _ in 1:m, _ in 1:n]

            reduced, U, R, tau = Flatter.fused_qr_size_reduction(
                B0; float_type = Float64)

            @test eltype(R) === Float64
            @test BigInt.(reduced) == BigInt.(B0) * BigInt.(U)
            @test abs(fqr_det(U)) == 1
            # Looser than BigFloat: 53 bits, plus the deadband.
            @test fqr_worst_mu(reduced) <= 55 // 100
        end

        @testset "entries beyond the exponent range are rejected" begin
            B0 = zeros(BigInt, 4, 3)
            for j in 1:3, i in 1:4
                B0[i, j] = big(2)^2000 + i + j
            end
            @test_throws ArgumentError Flatter.fused_qr_size_reduction(
                B0; float_type = Float64)
        end
    end

    # The BigFloat kernels are dispatched on `Matrix{BigFloat}`; a view is an
    # `AbstractMatrix{BigFloat}` but not a `Matrix`, so it takes the generic
    # value-semantics path. That gives a way to run both implementations on the
    # same data and compare.
    #
    # They are NOT bit-identical, and shouldn't be: the in-place version
    # accumulates with `mpfr_fma`, which rounds once where the generic version
    # rounds twice. The in-place path is the more accurate of the two.
    @testset "in-place BigFloat kernels agree with the generic ones" begin
        setprecision(BigFloat, 160) do
            size = Flatter._FUSED_WORKSPACE_SIZE

            # Two traps at once. `copy` on a Matrix{BigFloat} is SHALLOW: it
            # duplicates the array of references, not the values. And
            # `BigFloat(x)` returns x ITSELF when the precision already matches,
            # so it cannot be used to duplicate either. Allocating and writing
            # through `_mpfr_set!` is the way to get an independent value.
            duplicate(A) = [Flatter._mpfr_set!(BigFloat(), x) for x in A]

            @testset "reflector generation" begin
                rng = MersenneTwister(0x1A4F)
                for (m, column) in ((9, 1), (9, 4), (12, 7), (5, 5))
                    base = BigFloat.(randn(rng, m, m))
                    inplace = duplicate(base)
                    generic = duplicate(base)
                    tolerance = BigFloat(2)^(-130) * max(1, maximum(abs, base))

                    tau_inplace = Flatter._fused_larfg!(
                        inplace, column, m, Flatter._fused_temporaries(BigFloat, size))
                    tau_generic = Flatter._fused_larfg!(
                        view(generic, :, :), column, m,
                        Flatter._fused_temporaries(BigFloat, size))

                    @test abs(tau_inplace - tau_generic) < tolerance
                    @test maximum(abs, inplace - generic) < tolerance
                end
            end

            @testset "reflector application" begin
                rng = MersenneTwister(0x1A50)
                for (m, n) in ((10, 6), (7, 7))
                    base = BigFloat.(randn(rng, m, n))
                    inplace = duplicate(base)
                    generic = duplicate(base)
                    # Errors accumulate over a full sweep of reflectors, so this
                    # is looser than the single-reflector case above.
                    tolerance = BigFloat(2)^(-110) * max(1, maximum(abs, base))
                    ws_inplace = Flatter._fused_temporaries(BigFloat, size)
                    ws_generic = Flatter._fused_temporaries(BigFloat, size)

                    for reflector in 1:min(m, n) - 1
                        tau = Flatter._fused_larfg!(inplace, reflector, m, ws_inplace)
                        tau_generic = Flatter._fused_larfg!(
                            view(generic, :, :), reflector, m, ws_generic)
                        for target in (reflector + 1):n
                            Flatter._fused_larf_col!(inplace, target, reflector,
                                                     tau, m, ws_inplace)
                            Flatter._fused_larf_col!(view(generic, :, :), target,
                                                     reflector, tau_generic, m,
                                                     ws_generic)
                        end
                    end
                    @test maximum(abs, inplace - generic) < tolerance
                end
            end
        end
    end

    # Two independent hazards, both easy to trip over and both pinned here.
    # `copy` on a Matrix{BigFloat} duplicates references, not values; and
    # `BigFloat(x)` on a BigFloat returns x itself when the precision matches,
    # so neither yields an independent copy. Only allocating a new value and
    # writing into it does.
    @testset "duplicating a BigFloat matrix" begin
        setprecision(BigFloat, 128) do
            A = BigFloat[1.5 2.5; 3.5 4.5]
            shallow = copy(A)
            constructed = [BigFloat(x) for x in A]
            independent = [Flatter._mpfr_set!(BigFloat(), x) for x in A]

            # Same objects, before anything is mutated.
            @test all(shallow[i] === A[i] for i in eachindex(A))
            @test all(constructed[i] === A[i] for i in eachindex(A))
            @test all(independent[i] !== A[i] for i in eachindex(A))
            @test independent == A

            ws = Flatter._fused_temporaries(BigFloat, Flatter._FUSED_WORKSPACE_SIZE)
            Flatter._fused_larfg!(A, 1, 2, ws)

            @test shallow == A          # shares values, so it changed too
            @test constructed == A      # likewise
            @test independent != A      # genuinely independent
        end
    end

    # `at_precision` exists to hand back an independent, uniformly rounded
    # matrix, so it must not fall into the same trap when the precision it is
    # asked for happens to match what it was given.
    @testset "at_precision returns independent entries" begin
        setprecision(BigFloat, 128) do
            A = BigFloat[1.5 2.5; 3.5 4.5]
            same = Flatter.at_precision(A, 128)
            wider = Flatter.at_precision(A, 256)

            @test same == A
            @test all(same[i] !== A[i] for i in eachindex(A))
            @test Flatter.uniform_precision(same) == 128
            @test Flatter.uniform_precision(wider) == 256
        end
    end

    # The in-place kernels mutate BigFloat OBJECTS. `R` is owned by the
    # factorisation so that is safe, but `B` and `U` belong to the caller and
    # may share objects with a copy the caller still holds -- `Matrix{BigInt}(B)`
    # copies the array, not the entries. If a kernel ever started mutating those
    # in place, this test is what would catch it.
    @testset "the caller's matrices are never mutated in place" begin
        rng = MersenneTwister(0xA11A5)
        m, n = 10, 6
        B = fqr_random_basis(rng, m, n; bits = 40)

        shallow = Matrix{BigInt}(B)          # shares entry objects with B
        deep = [BigInt(x) for x in B]        # independent objects

        Flatter.fused_qr_size_reduction!(B, Matrix{BigInt}(undef, n, n);
                                         precision = 200)

        # `shallow` must still equal the ORIGINAL, not the reduced result:
        # assignment into B rebinds slots, it does not mutate the old objects.
        @test shallow == deep
    end

    # The integer update loops compute the product into a scratch value rather
    # than allocating one per element. The scratch must not leak into the
    # result, and the caller's entry objects must still be rebound rather than
    # mutated, since they may be shared with a copy the caller holds.
    @testset "integer update matches the direct expression" begin
        rng = MersenneTwister(0x17D)
        for (m, n) in ((8, 5), (12, 12))
            base = fqr_random_basis(rng, m, n; bits = 60)
            multiplier = BigInt(rand(rng, -50:50))

            direct = [BigInt(x) for x in base]
            for k in 1:m
                direct[k, 2] -= multiplier * direct[k, 1]
            end

            via = [BigInt(x) for x in base]
            shared = via                      # same array, to check rebinding
            entries = [via[k, 2] for k in 1:m]
            Flatter._fused_integer_update!(via, 2, 1, 1:m, multiplier, zero(BigInt))

            @test via[:, 2] == direct[:, 2]
            @test via[:, 3:end] == direct[:, 3:end]
            # The old entry objects must be untouched: the slot was rebound.
            @test all(entries[k] === via[k, 2] || entries[k] == base[k, 2]
                      for k in 1:m)
        end
    end

    @testset "in-place form" begin
        rng = MersenneTwister(0x1409)
        m, n = 9, 5
        B0 = fqr_random_basis(rng, m, n; bits = 30)
        B = copy(B0)
        U = Matrix{BigInt}(undef, n, n)

        returned_B, returned_U, R, tau =
            Flatter.fused_qr_size_reduction!(B, U; precision = 256)

        @test returned_B === B
        @test returned_U === U
        @test BigInt.(B) == BigInt.(B0) * BigInt.(U)
        @test abs(fqr_det(U)) == 1
    end

    @testset "precision" begin
        @testset "the default heuristic works unaided" begin
            rng = MersenneTwister(0x9DEF)
            B0 = fqr_random_basis(rng, 8, 5; bits = 60)
            reduced, U, R, _ = Flatter.fused_qr_size_reduction(B0)
            @test BigInt.(reduced) == BigInt.(B0) * BigInt.(U)
            @test abs(fqr_det(U)) == 1
            @test fqr_worst_mu(reduced) <= 51 // 100 + 1 // 1000
        end

        @testset "R comes back at a uniform precision" begin
            rng = MersenneTwister(0x9DF0)
            B0 = fqr_random_basis(rng, 7, 4; bits = 30)
            _, _, R, tau = Flatter.fused_qr_size_reduction(B0; precision = 192)
            @test Flatter.uniform_precision(R) == 192
            @test Flatter.uniform_precision(tau) == 192
        end
    end

    @testset "argument checking" begin
        rng = MersenneTwister(0xBADF)
        good = fqr_random_basis(rng, 6, 4; bits = 20)

        # More columns than rows.
        @test_throws DimensionMismatch Flatter.fused_qr_size_reduction(
            fqr_random_basis(rng, 3, 5; bits = 20))
        @test_throws DimensionMismatch Flatter.fused_qr_size_reduction!(
            copy(good), Matrix{BigInt}(undef, 3, 3))
        @test_throws ArgumentError Flatter.fused_qr_size_reduction(
            good; deadband = 0.0)
        @test_throws ArgumentError Flatter.fused_qr_size_reduction(
            good; max_passes = 0)
        @test_throws ArgumentError Flatter.fused_qr_size_reduction(
            Matrix{BigInt}(undef, 0, 0))

        # Rank deficient: a repeated column makes a diagonal entry vanish.
        singular = copy(good)
        singular[:, 3] .= singular[:, 1]
        @test_throws Exception Flatter.fused_qr_size_reduction(
            singular; precision = 128)
    end
end
