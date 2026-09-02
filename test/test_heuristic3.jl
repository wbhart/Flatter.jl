# test/test_heuristic3.jl
#
# Assumes `using Test` and `using Random` from runtests.jl.

@testset "heuristic3 tiling" begin

    @testset "tile_partition" begin
        @testset "a single central window leaves a gap either side" begin
            tiles, reducible = Flatter.tile_partition([3:6], 10)
            @test [t.range for t in tiles] == [1:2, 3:6, 7:10]
            @test [t.reduce for t in tiles] == [false, true, false]
            @test reducible == [2]
        end

        @testset "a window at each end leaves no outer gaps" begin
            tiles, reducible = Flatter.tile_partition([1:4, 7:10], 10)
            @test [t.range for t in tiles] == [1:4, 5:6, 7:10]
            @test [t.reduce for t in tiles] == [true, false, true]
            @test reducible == [1, 3]
        end

        @testset "adjacent windows leave no gap between them" begin
            tiles, reducible = Flatter.tile_partition([1:5, 6:10], 10)
            @test [t.range for t in tiles] == [1:5, 6:10]
            @test reducible == [1, 2]
        end

        @testset "no windows gives one untouched tile" begin
            tiles, reducible = Flatter.tile_partition(UnitRange{Int}[], 8)
            @test [t.range for t in tiles] == [1:8]
            @test [t.reduce for t in tiles] == [false]
            @test isempty(reducible)
        end

        @testset "the tiles always cover 1:n exactly once" begin
            for (windows, n) in (([2:3], 7), ([1:2, 5:6], 8), ([4:9], 9),
                                 ([1:1, 3:3, 5:5], 6))
                tiles, _ = Flatter.tile_partition(windows, n)
                covered = reduce(vcat, [collect(t.range) for t in tiles])
                @test covered == collect(1:n)
            end
        end

        @testset "overlapping or unsorted windows are rejected" begin
            @test_throws ArgumentError Flatter.tile_partition([3:6, 5:8], 10)
            @test_throws ArgumentError Flatter.tile_partition([5:8, 1:3], 10)
            @test_throws ArgumentError Flatter.tile_partition([3:12], 10)
        end
    end

    @testset "embed_transforms!" begin
        @testset "produces a block diagonal matrix" begin
            tiles, reducible = Flatter.tile_partition([2:3, 5:6], 7)
            U = Matrix{BigInt}(undef, 7, 7)
            blocks = [BigInt[1 2; 0 1], BigInt[1 0; -3 1]]
            Flatter.embed_transforms!(U, tiles, reducible, blocks)

            @test U[2:3, 2:3] == blocks[1]
            @test U[5:6, 5:6] == blocks[2]
            # Everything outside the reduced tiles is the identity.
            for j in 1:7, i in 1:7
                (i in 2:3 && j in 2:3) && continue
                (i in 5:6 && j in 5:6) && continue
                @test U[i, j] == (i == j ? 1 : 0)
            end
            # Block diagonal with unimodular blocks, so unimodular overall.
            @test abs(rr_det(U)) == 1
        end

        @testset "a mismatched transform is rejected" begin
            tiles, reducible = Flatter.tile_partition([2:3], 5)
            U = Matrix{BigInt}(undef, 5, 5)
            @test_throws DimensionMismatch Flatter.embed_transforms!(
                U, tiles, reducible, [BigInt[1 0 0; 0 1 0; 0 0 1]])
            @test_throws DimensionMismatch Flatter.embed_transforms!(
                U, tiles, reducible, Matrix{BigInt}[])
        end
    end

    # The tiled update must agree with the plain product it stands in for. That
    # is the whole contract: exploiting the block structure is an optimisation,
    # not a different computation.
    @testset "tiled_basis_update! matches the full product" begin
        @testset "windows = $windows, n = $n" for (windows, n) in
                (([3:6], 10), ([1:4, 7:10], 10), ([2:5], 8), ([1:8], 8))
            rng = MersenneTwister(hash((windows, n, :update)))
            tiles, reducible = Flatter.tile_partition(windows, n)

            # Upper triangular by tile, as the driver's basis is.
            B = zeros(BigInt, n, n)
            for j in 1:n, i in 1:j
                B[i, j] = rand(rng, -40:40)
                i == j && iszero(B[i, j]) && (B[i, j] = big(1))
            end

            blocks = [BigInt[i == j ? 1 : rand(rng, -3:3)
                             for i in 1:length(tiles[k].range),
                                 j in 1:length(tiles[k].range)]
                      for k in reducible]
            # Make each block unimodular by forcing it triangular.
            for block in blocks, j in 1:size(block, 2), i in (j + 1):size(block, 1)
                block[i, j] = 0
            end

            U = Matrix{BigInt}(undef, n, n)
            Flatter.embed_transforms!(U, tiles, reducible, blocks)

            expected = B * U
            actual = zeros(BigInt, n, n)
            Flatter.tiled_basis_update!(actual, B, U, tiles)

            # Only the blocks at or above the tile diagonal are written; the rest
            # is structurally zero in both.
            for (c, column_tile) in enumerate(tiles), r in 1:c
                @test actual[tiles[r].range, column_tile.range] ==
                      expected[tiles[r].range, column_tile.range]
            end
        end
    end

    @testset "tiled_diagonal_qr! factorises each reduced tile" begin
        setprecision(BigFloat, 200) do
            rng = MersenneTwister(0x4173)
            n = 12
            tiles, _ = Flatter.tile_partition([1:4, 7:10], n)

            B = zeros(BigInt, n, n)
            for j in 1:n, i in 1:j
                B[i, j] = rand(rng, -30:30)
                i == j && iszero(B[i, j]) && (B[i, j] = big(1))
            end

            R = zeros(BigFloat, n, n)
            tau = zeros(BigFloat, n)
            Flatter.tiled_diagonal_qr!(R, tau, B, tiles, 200)

            for tile in tiles
                range = tile.range
                if tile.reduce
                    # The block's own factorisation, independent of the rest.
                    reference, reference_T = Flatter.householder_block(
                        Flatter.bigfloat_matrix(B[range, range], 200))
                    @test maximum(abs, triu(R[range, range]) - triu(reference)) <
                          BigFloat(2)^(-180) * max(1, maximum(abs, reference))

                    # `tau` is the diagonal of the compact-WY T factor.  This
                    # catches the easy-to-miss mistake of treating the second
                    # return value of `householder_block` as a vector and
                    # linearly indexing its first column.
                    for (k, row) in enumerate(range)
                        @test abs(tau[row] - reference_T[k, k]) <
                              BigFloat(2)^(-180) * max(1, abs(reference_T[k, k]))
                    end
                else
                    # A gap tile is still upper triangular, so it is its own R
                    # factor and is copied rather than factorised.
                    for j in range, i in range
                        expected = i <= j ? BigFloat(B[i, j]) : zero(BigFloat)
                        @test R[i, j] == expected
                    end
                end
            end
        end
    end
end

@testset "heuristic3 tiled size reduction" begin

    "A basis that is upper triangular by tile, as the driver's is."
    function h3_tiled_basis(rng, n::Int; bits::Int = 30)
        B = zeros(BigInt, n, n)
        for j in 1:n, i in 1:j
            B[i, j] = rand(rng, -(1 << bits):(1 << bits))
        end
        for i in 1:n
            iszero(B[i, i]) && (B[i, i] = big(1) << rand(rng, 1:bits))
        end
        return B
    end

    # The exact contract, which holds however the tiles are cut: the reduced
    # basis is the original times the accumulated transform, and that transform
    # is unimodular. Quality depends on the decomposition; correctness does not.
    @testset "exact contract, windows = $windows, n = $n" for (windows, n) in
            (([3:6], 10), ([1:4, 7:10], 10), ([2:5], 8), ([1:6], 6))
        rng = MersenneTwister(hash((windows, n, :sr)))
        tiles, reducible = Flatter.tile_partition(windows, n)

        B0 = h3_tiled_basis(rng, n)
        B = copy(B0)
        B_next = copy(B0)
        U_i = Matrix{BigInt}(undef, n, n)
        Flatter._set_identity!(U_i)
        U_sr = Matrix{BigInt}(undef, n, n)

        R = zeros(BigFloat, n, n)
        tau = zeros(BigFloat, n)
        setprecision(BigFloat, 256) do
            Flatter.tiled_diagonal_qr!(R, tau, B_next, tiles, 256)
            Flatter.tiled_size_reduction!(B, B_next, U_i, U_sr, R, tau, tiles;
                                          precision = 256)
        end

        # `U_i` accumulates the PRODUCT of the per-pair transforms; `U_sr` holds
        # each pair's block on its own and never sees the cross terms. With one
        # or two tiles they coincide, which is exactly why a test written
        # against `U_sr` passes until a third tile appears.
        @test abs(rr_det(U_i)) == 1
        @test B_next == B0 * U_i
    end

    @testset "a single tile leaves everything alone" begin
        # With no tile above any other there is nothing to reduce against, so
        # the transform must come back the identity.
        rng = MersenneTwister(0x51E1)
        n = 6
        tiles, _ = Flatter.tile_partition([1:6], n)
        B0 = h3_tiled_basis(rng, n)
        B, B_next = copy(B0), copy(B0)
        U_i = Matrix{BigInt}(undef, n, n); Flatter._set_identity!(U_i)
        U_sr = Matrix{BigInt}(undef, n, n)
        R = zeros(BigFloat, n, n); tau = zeros(BigFloat, n)

        setprecision(BigFloat, 200) do
            Flatter.tiled_diagonal_qr!(R, tau, B_next, tiles, 200)
            Flatter.tiled_size_reduction!(B, B_next, U_i, U_sr, R, tau, tiles;
                                          precision = 200)
        end

        @test U_i == BigInt[i == j ? 1 : 0 for i in 1:n, j in 1:n]
        @test U_sr == BigInt[i == j ? 1 : 0 for i in 1:n, j in 1:n]
        @test B_next == B0
    end

    @testset "reduction actually shortens the off-diagonal blocks" begin
        # Terminating with a valid transform is not enough: the point is that
        # the upper blocks get smaller.
        rng = MersenneTwister(0x51E2)
        n = 12
        tiles, _ = Flatter.tile_partition([1:6, 7:12], n)

        B0 = zeros(BigInt, n, n)
        for j in 1:n, i in 1:j
            B0[i, j] = i == j ? big(1) << 20 : rand(rng, -(1 << 40):(1 << 40))
        end

        B, B_next = copy(B0), copy(B0)
        U_i = Matrix{BigInt}(undef, n, n); Flatter._set_identity!(U_i)
        U_sr = Matrix{BigInt}(undef, n, n)
        R = zeros(BigFloat, n, n); tau = zeros(BigFloat, n)

        setprecision(BigFloat, 400) do
            Flatter.tiled_diagonal_qr!(R, tau, B_next, tiles, 400)
            Flatter.tiled_size_reduction!(B, B_next, U_i, U_sr, R, tau, tiles;
                                          precision = 400)
        end

        @test B_next == B0 * U_i
        before = maximum(abs, B0[1:6, 7:12])
        after = maximum(abs, B_next[1:6, 7:12])
        @test after < before
    end
end

@testset "heuristic3 fpLLL precision cutoff" begin
    n = 4

    low = zeros(BigInt, n, n)
    for i in 1:n
        low[i, i] = big(1) << 40
    end
    low_U = Matrix{BigInt}(undef, n, n)
    low_telemetry = Flatter.ReductionTelemetry()
    low_reduced, low_U, _ = Flatter.lattice_reduce!(
        copy(low), low_U; base_cutoff = 4, max_iterations = 4,
        want_profile = false, telemetry = low_telemetry, _heuristic_phase = 3)
    @test low_reduced == low * low_U
    @test low_telemetry.fplll_calls == 1

    high = zeros(BigInt, n, n)
    for i in 1:n
        high[i, i] = big(1) << 160
    end
    high_U = Matrix{BigInt}(undef, n, n)
    high_telemetry = Flatter.ReductionTelemetry()
    high_reduced, high_U, info = Flatter.lattice_reduce!(
        copy(high), high_U; base_cutoff = 4, max_iterations = 4,
        want_profile = false, telemetry = high_telemetry, _heuristic_phase = 3)
    @test high_reduced == high * high_U
    @test info.goal_met
    @test high_telemetry.fplll_calls == 0
end

@testset "heuristic3 update_representation" begin

    # Called with no surrounding precision block, as in the driver. Every
    # BigFloat allocated by the update must use the requested working precision.
    @testset "works outside a setprecision block" begin
        rng = MersenneTwister(0x0303)
        n, window = 10, 4:9
        working = zeros(BigInt, n, n)
        for j in 1:n
            working[j, j] = big(1) << 25 + rand(rng, 1:500)
            for i in 1:(j - 1)
                working[i, j] = rand(rng, -(big(1) << 40):(big(1) << 40))
            end
        end
        sub = BigInt[i == j ? 1 : 0 for i in 1:length(window), j in 1:length(window)]

        basis, transform, R, tau = Flatter.heuristic3_update!(
            working, window, sub, n, 512)

        @test basis == working * transform
        @test abs(rr_det(transform)) == 1
        @test Flatter.uniform_precision(R) == 512
        @test all(Base.precision(t) == 512 for t in tau)
    end


    "A triangular basis with a controlled diagonal spread."
    function h3_basis(rng, n::Int; spread::Int = 60)
        B = zeros(BigInt, n, n)
        for j in 1:n
            shift = n == 1 ? 0 : div(spread * (n - j), n - 1)
            B[j, j] = (big(rand(rng, 1:100)) + 1) << shift
            for i in 1:(j - 1)
                B[i, j] = rand(rng, -1000:1000)
            end
        end
        return B
    end

    "A random unimodular matrix of the given order."
    function h3_unimodular(rng, k::Int)
        U = BigInt[i == j ? 1 : 0 for i in 1:k, j in 1:k]
        for _ in 1:(4k)
            a, b = rand(rng, 1:k), rand(rng, 1:k)
            a == b && continue
            c = rand(rng, -2:2)
            iszero(c) && continue
            for r in 1:k
                U[r, a] += c * U[r, b]
            end
        end
        return U
    end

    # The invariant that must hold whatever the tiling: the returned basis is
    # the old one times the returned transform, and that transform is
    # unimodular. Quality depends on the tile decomposition; correctness does
    # not.
    @testset "exact contract, n = $n, window = $window" for (n, window) in
            ((10, 3:6), (10, 1:5), (12, 7:12), (8, 2:7), (6, 1:6))
        rng = MersenneTwister(hash((n, window, :update)))
        working = h3_basis(rng, n)
        sub = h3_unimodular(rng, length(window))

        basis, transform, R, tau = setprecision(BigFloat, 300) do
            Flatter.heuristic3_update!(working, window, sub, n, 300)
        end

        @test basis == working * transform
        @test abs(rr_det(transform)) == 1
        @test size(R) == (n, n)
        @test length(tau) == n
    end

    # The profile is read straight off R's diagonal, so every entry of it has to
    # be finite -- including the tiles no reduction touched. Leaving those blocks
    # zero puts a -Inf in the profile and the compression refuses it.
    @testset "every diagonal entry of R is finite and nonzero" begin
        @testset "window = $window, n = $n" for (window, n) in
                ((3:6, 10), (1:4, 10), (7:12, 12), (2:7, 8))
            rng = MersenneTwister(hash((window, n, :finite)))
            working = h3_basis(rng, n; spread = 50)
            sub = h3_unimodular(rng, length(window))

            _, _, R, _ = Flatter.heuristic3_update!(working, window, sub, n, 300)

            for i in 1:n
                @test isfinite(Float64(log2(abs(R[i, i]))))
                @test !iszero(R[i, i])
            end
        end
    end

    # The blocks ABOVE the diagonal carry the coordinates of one tile in
    # another's frame. Discarding them leaves a block-diagonal R, which still
    # has a valid profile but describes a different lattice to the compression.
    # A tile pair whose entries are far wider than the driver's precision needs
    # more precision than the driver would supply. The reduction must get it --
    # under-provisioned, it does not fail, it just stops converging and runs
    # every refinement pass on every column -- and `R` must still come back
    # uniform, since the reduction returns its coordinates at its own figure.
    @testset "a tile pair needing more precision than the driver still works" begin
        rng = MersenneTwister(0x0305)
        n, window = 16, 9:16

        working = zeros(BigInt, n, n)
        for j in 1:n
            working[j, j] = big(1) << 10 + rand(rng, 1:100)
            for i in 1:(j - 1)
                working[i, j] = rand(rng, -1000:1000)
            end
        end
        # Far wider than a modest driver precision would cover.
        for j in 9:16, i in 1:8
            working[i, j] = rand(rng, -(big(1) << 300):(big(1) << 300))
        end
        sub = h3_unimodular(rng, length(window))

        basis, transform, R, _ = Flatter.heuristic3_update!(
            working, window, sub, n, 128)

        @test basis == working * transform
        @test abs(rr_det(transform)) == 1
        @test Flatter.uniform_precision(R) == 128
        @test maximum(abs, basis[1:8, 9:16]) < maximum(abs, working[1:8, 9:16])
    end

    @testset "R has nonzero blocks above the tile diagonal" begin
        rng = MersenneTwister(0x0304)
        n, window = 12, 7:12
        working = h3_basis(rng, n; spread = 70)
        sub = h3_unimodular(rng, length(window))

        _, _, R, _ = Flatter.heuristic3_update!(working, window, sub, n, 400)

        @test any(!iszero, R[1:6, 7:12])
    end

    @testset "the R factor matches the basis it came from" begin
        # Each reduced tile's diagonal block of R is that block's own QR, so its
        # absolute diagonal gives the tile's profile.
        rng = MersenneTwister(0x0301)
        n, window = 12, 5:12
        working = h3_basis(rng, n; spread = 90)
        sub = h3_unimodular(rng, length(window))

        basis, transform, R, _ = setprecision(BigFloat, 400) do
            Flatter.heuristic3_update!(working, window, sub, n, 400)
        end

        @test basis == working * transform
        for i in window
            @test !iszero(R[i, i])
        end
    end

    @testset "an identity sublattice transform still reduces" begin
        # With nothing to fold in, the update is pure size reduction across the
        # tile boundary -- and must still return a valid, non-trivial transform.
        #
        # The off-diagonal entries have to be large relative to the diagonal of
        # the block they are reduced against, otherwise the identity transform is
        # already size reduced.
        rng = MersenneTwister(0x0302)
        n, window = 10, 6:10

        working = zeros(BigInt, n, n)
        for j in 1:n
            working[j, j] = big(1) << 20 + rand(rng, 1:1000)
            for i in 1:(j - 1)
                working[i, j] = rand(rng, -1000:1000)
            end
        end
        # Far larger than any diagonal, so every row genuinely needs reducing.
        for j in 6:10, i in 1:5
            working[i, j] = rand(rng, -(big(1) << 50):(big(1) << 50))
        end
        identity_sub = BigInt[i == j ? 1 : 0 for i in 1:5, j in 1:5]

        basis, transform, _, _ = setprecision(BigFloat, 300) do
            Flatter.heuristic3_update!(working, window, identity_sub, n, 300)
        end

        @test basis == working * transform
        @test abs(rr_det(transform)) == 1
        # Every entry should come down to the scale of the diagonal it was
        # reduced against, which is around 2^20 -- a fall of some 30 bits.
        @test maximum(abs, basis[1:5, 6:10]) <
              maximum(abs, working[1:5, 6:10]) >> 20
        @test transform != BigInt[i == j ? 1 : 0 for i in 1:n, j in 1:n]
    end
end

@testset "heuristic3 on a badly scaled tile" begin
    # A knapsack basis is mostly unit diagonal entries beside one enormous one.
    # Factorising such a tile at the driver's precision -- chosen for the whole
    # matrix rather than for the tile -- cancels a diagonal to zero on the first
    # iteration, which is what broke the driver at dimension 48.
    @testset "a tile with a huge entry keeps its diagonal, n = $n" for n in (12, 16, 24)
        rng = MersenneTwister(hash((n, :scaled)))
        window = div(n, 2):n

        working = zeros(BigInt, n, n)
        for j in 1:n
            working[j, j] = big(1)
            for i in 1:(j - 1)
                working[i, j] = rand(rng, -5:5)
            end
        end
        # One column far larger than everything else, as a knapsack weight is.
        working[1, n] = big(1) << 200
        working[n, n] = big(1) << 180

        sub = BigInt[i == j ? 1 : 0
                     for i in 1:length(window), j in 1:length(window)]

        # A deliberately modest precision: the tile has to ask for more itself.
        basis, transform, R, _ = Flatter.heuristic3_update!(
            working, [window], [sub], n, 64)

        @test basis == working * transform
        for i in 1:n
            @test !iszero(R[i, i])
        end
        @test Flatter.uniform_precision(R) == 64
    end
end

@testset "heuristic3 R against a true factorisation" begin

    "A triangular basis with a controlled diagonal spread."
    function h3r_basis(rng, n::Int; spread::Int = 60)
        B = zeros(BigInt, n, n)
        for j in 1:n
            shift = n == 1 ? 0 : div(spread * (n - j), n - 1)
            B[j, j] = (big(rand(rng, 1:100)) + 1) << shift
            for i in 1:(j - 1)
                B[i, j] = rand(rng, -1000:1000)
            end
        end
        return B
    end

    function h3r_unimodular(rng, k::Int)
        U = BigInt[i == j ? 1 : 0 for i in 1:k, j in 1:k]
        for _ in 1:(4k)
            a, b = rand(rng, 1:k), rand(rng, 1:k)
            a == b && continue
            c = rand(rng, -2:2)
            iszero(c) && continue
            for r in 1:k
                U[r, a] += c * U[r, b]
            end
        end
        return U
    end

    # The driver takes its working precision from this diagonal. If it disagrees
    # with a genuine factorisation of the same basis, every precision decision
    # downstream is made on a wrong number -- which is the difference between a
    # design compromise and a bug.
    @testset "the diagonal matches a global QR, n = $n, window = $window" for
            (n, window) in ((12, 5:12), (16, 1:8), (16, 9:16), (20, 6:15))
        rng = MersenneTwister(hash((n, window, :truth)))
        working = h3r_basis(rng, n; spread = 80)
        sub = h3r_unimodular(rng, length(window))

        basis, transform, R, _ = Flatter.heuristic3_update!(
            working, [window], [sub], n, 512)

        @test basis == working * transform

        # A genuine factorisation of the basis the tiled update produced.
        truth = setprecision(BigFloat, 512) do
            factors, _ = Flatter.householder_block(
                Flatter.bigfloat_matrix(basis, 512))
            factors
        end

        for i in 1:n
            @test abs(Float64(log2(abs(R[i, i]))) -
                      Float64(log2(abs(truth[i, i])))) < 1e-6
        end
    end

    # The basis the tiled update returns must be reducible at the precision its
    # own profile implies. A basis whose off-diagonal entries are far larger than
    # the profile suggests will exhaust the size reduction's passes downstream.
    @testset "the result is size reducible at its own precision" begin
        rng = MersenneTwister(0x7121)
        for (n, window) in ((16, 9:16), (20, 1:10))
            working = h3r_basis(rng, n; spread = 60)
            sub = h3r_unimodular(rng, length(window))

            basis, _, R, _ = Flatter.heuristic3_update!(
                working, [window], [sub], n, 400)

            profile = [Float64(log2(abs(R[i, i]))) for i in 1:n]
            spread = maximum(profile) - minimum(profile)
            bits = Flatter.lll_precision(spread, n)

            U = Matrix{BigInt}(undef, n, n)
            reduced, _, _, _ = Flatter.fused_qr_size_reduction!(
                Matrix{BigInt}(basis), U; precision = bits)
            @test reduced == basis * U
        end
    end
end

@testset "heuristic3 over repeated updates" begin
    # A single update keeps R's diagonal exact -- the tests above check that.
    # The driver fails only after seven to sixteen iterations, so whatever goes
    # wrong needs the output of one update fed into the next. This reproduces
    # that loop without the driver around it.

    function h3rep_basis(rng, n::Int; spread::Int = 80)
        B = zeros(BigInt, n, n)
        for j in 1:n
            shift = n == 1 ? 0 : div(spread * (n - j), n - 1)
            B[j, j] = (big(rand(rng, 1:100)) + 1) << shift
            for i in 1:(j - 1)
                B[i, j] = rand(rng, -1000:1000)
            end
        end
        return B
    end

    function h3rep_unimodular(rng, k::Int)
        U = BigInt[i == j ? 1 : 0 for i in 1:k, j in 1:k]
        for _ in 1:(4k)
            a, b = rand(rng, 1:k), rand(rng, 1:k)
            a == b && continue
            c = rand(rng, -2:2)
            iszero(c) && continue
            for r in 1:k
                U[r, a] += c * U[r, b]
            end
        end
        return U
    end

    "The true profile of a basis, from a global factorisation."
    function h3rep_truth(B, n, bits)
        Flatter.with_precision(bits) do
            factors, _ = Flatter.householder_block(Flatter.bigfloat_matrix(B, bits))
            [Flatter._log2_abs(factors[i, i]) for i in 1:n]
        end
    end

    # `heuristic3_update!` requires a fully upper-triangular working
    # representation.  A raw tile update is only block upper triangular; the
    # driver restores the stronger invariant by compressing its R factor between
    # updates.
    @testset "the update rejects a basis that is not upper triangular" begin
        rng = MersenneTwister(0x7A11)
        n = 12
        working = h3rep_basis(rng, n)
        working[3, 1] = big(7)          # one entry below the diagonal
        window = 1:div(n, 2)
        sub = h3rep_unimodular(rng, length(window))

        @test_throws ArgumentError Flatter.heuristic3_update!(
            working, [window], [sub], n, 400)
    end

    # The chained test above passes, so the update is sound when its own output
    # is fed straight back in. The driver does one thing in between: it
    # compresses `R` into the next iteration's integer basis. This adds exactly
    # that step and nothing else.
    @testset "R stays exact with compression between updates, n = $n" for n in (12, 16, 24)
        rng = MersenneTwister(hash((n, :compressed)))
        working = h3rep_basis(rng, n)
        bits = 400

        profile = [Flatter._log2_abs(working[i, i]) for i in 1:n]
        offsets = zeros(Float64, n)
        working, _, precision = Flatter._compress_integer(working, profile, offsets,
                                                          n, false)

        cycle = [div(n, 4):div(3n, 4), 1:div(n, 2), (div(n, 2) + 1):n]

        for round in 1:10
            window = cycle[mod1(round, length(cycle))]
            sub = h3rep_unimodular(rng, length(window))

            basis, transform, R, _ = Flatter.heuristic3_update!(
                working, [window], [sub], n, precision)

            @test basis == working * transform

            truth = h3rep_truth(basis, n, max(precision, 400))
            for i in 1:n
                reported = Flatter._log2_abs(R[i, i])
                @test abs(reported - truth[i]) < 1.0
            end

            # As the driver does it: the raw diagonal, with `_compress_factor`
            # owning the offset bookkeeping. Adding `offsets` here as well would
            # count the accumulated shift twice.
            for i in 1:n
                profile[i] = Flatter._log2_abs(R[i, i])
            end
            working, _, precision = Flatter._compress_factor(R, profile, offsets,
                                                             n, BigInt, false)
        end
    end

    # Update and compression chained both hold. The last difference from the
    # driver is where the sub-transform comes from: these tests use a product of
    # small transvections, the driver uses the output of an actual reduction.
    # Those are not alike -- a real transform has large, strongly structured
    # entries, and the tile it produces is conditioned quite differently.
    @testset "R stays exact with a real sub-transform, n = $n" for n in (12, 16, 24)
        rng = MersenneTwister(hash((n, :realsub)))
        working = h3rep_basis(rng, n)

        profile = [Flatter._log2_abs(working[i, i]) for i in 1:n]
        offsets = zeros(Float64, n)
        working, _, precision = Flatter._compress_integer(working, profile, offsets,
                                                          n, false)

        cycle = [div(n, 4):div(3n, 4), 1:div(n, 2), (div(n, 2) + 1):n]

        for round in 1:10
            window = cycle[mod1(round, length(cycle))]
            width = length(window)

            # The transform an actual reduction of the window produces.
            sub_basis = Matrix{BigInt}(working[window, window])
            sub = Matrix{BigInt}(undef, width, width)
            Flatter.lattice_reduce!(sub_basis, sub; base_cutoff = 4,
                                    max_iterations = 20, want_profile = false)

            basis, transform, R, _ = Flatter.heuristic3_update!(
                working, [window], [sub], n, precision)

            @test basis == working * transform

            truth = h3rep_truth(basis, n, max(precision, 400))
            for i in 1:n
                reported = Flatter._log2_abs(R[i, i])
                @test abs(reported - truth[i]) < 1.0
            end

            for i in 1:n
                profile[i] = Flatter._log2_abs(R[i, i])
            end
            working, _, precision = Flatter._compress_factor(R, profile, offsets,
                                                             n, BigInt, false)
        end
    end
end

@testset "heuristic3 which stage moves the profile" begin
    # The update has two stages: apply the block-diagonal transform, then size
    # reduce across tiles. Size reduction adds multiples of earlier columns to
    # later ones, which leaves every leading sublattice determinant alone -- so
    # it CANNOT change the profile. If the profile moves, it moved in the basis
    # update. This measures both rather than assuming.

    function h3s_profile(B, n, bits)
        Flatter.with_precision(bits) do
            factors, _ = Flatter.householder_block(Flatter.bigfloat_matrix(B, bits))
            [Flatter._log2_abs(factors[i, i]) for i in 1:n]
        end
    end

    @testset "size reduction leaves the profile alone, n = $n" for n in (12, 16, 20)
        rng = MersenneTwister(hash((n, :stages)))
        window = div(n, 3):n
        bits = 512

        working = zeros(BigInt, n, n)
        for j in 1:n
            working[j, j] = big(1) << rand(rng, 5:40)
            for i in 1:(j - 1)
                working[i, j] = rand(rng, -(big(1) << 30):(big(1) << 30))
            end
        end

        # Reduce the window for real, as the driver does.
        width = length(window)
        sub_basis = Matrix{BigInt}(working[window, window])
        sub = Matrix{BigInt}(undef, width, width)
        Flatter.lattice_reduce!(sub_basis, sub; base_cutoff = 4,
                                max_iterations = 20, want_profile = false)

        before = h3s_profile(working, n, bits)

        # Stage one on its own: the block-diagonal product.
        tiles, reducible = Flatter.tile_partition([window], n)
        embedded = Matrix{BigInt}(undef, n, n)
        Flatter.embed_transforms!(embedded, tiles, reducible, [sub])
        updated = zeros(BigInt, n, n)
        Flatter.tiled_basis_update!(updated, working, embedded, tiles)
        @test updated == working * embedded
        after_update = h3s_profile(updated, n, bits)

        # Stage two: the whole update, which adds the size reduction.
        final, transform, _, _ = Flatter.heuristic3_update!(
            working, [window], [sub], n, bits)
        @test final == working * transform
        after_all = h3s_profile(final, n, bits)

        spread(p) = maximum(p) - minimum(p)

        # Size reduction must not move the profile at all.
        for i in 1:n
            @test abs(after_all[i] - after_update[i]) < 1.0
        end

        # And the whole update must not make the basis worse. If this fails
        # while the check above passes, the fault is in the basis update.
        @test spread(after_all) <= spread(before) + 64.0
    end
end
