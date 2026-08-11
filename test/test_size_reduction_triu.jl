# test/test_Flatter.size_reduction_triu.jl

# ---------------------------------------------------------------------------
# Predicates and generators
# ---------------------------------------------------------------------------

"2*|R[i,j]| <= |R[i,i]| for every strictly upper entry."
function is_size_reduced(R::AbstractMatrix)
    n = size(R, 1)
    for j in 1:n, i in 1:(j - 1)
        2 * abs(R[i, j]) <= abs(R[i, i]) || return false
    end
    return true
end

"Unit upper triangular, which for an integer matrix certifies determinant one."
function is_unit_upper_triangular(U::AbstractMatrix)
    n = size(U, 1)
    for j in 1:n, i in 1:n
        if i == j
            isone(U[i, i]) || return false
        elseif i > j
            iszero(U[i, j]) || return false
        end
    end
    return true
end

"""
Check the full contract in `BigInt`, so that a silent `Int64` overflow inside
the kernel shows up as a failed identity rather than a plausible wrong answer.
"""
function check_reduction(R0::AbstractMatrix, Rred::AbstractMatrix, U::AbstractMatrix)
    B0 = BigInt.(R0)
    Br = BigInt.(Rred)
    Bu = BigInt.(U)
    return is_size_reduced(Br) && is_unit_upper_triangular(Bu) && Br == B0 * Bu
end

"Small entries: safe for `Int64` at the sizes used here."
function random_upper_triangular(rng, ::Type{T}, n::Integer;
                                 magnitude::Int = 100,
                                 diagonal_magnitude::Int = 200) where {T<:Integer}
    R = zeros(T, n, n)
    for j in 1:n
        for i in 1:(j - 1)
            R[i, j] = T(rand(rng, (-magnitude):magnitude))
        end
        d = 0
        while d == 0
            d = rand(rng, (-diagonal_magnitude):diagonal_magnitude)
        end
        R[j, j] = T(d)
    end
    return R
end

"A random signed `BigInt` of roughly `bits` bits, built without assuming a
`BigInt` range sampler."
function random_bigint(rng, bits::Int)
    x = big(0)
    for _ in 1:cld(bits, 32)
        x = (x << 32) + rand(rng, 0:(2^32 - 1))
    end
    x >>= max(0, 32 * cld(bits, 32) - bits)
    return rand(rng, Bool) ? x : -x
end

function random_upper_triangular_big(rng, n::Integer; bits::Int = 128)
    R = zeros(BigInt, n, n)
    for j in 1:n
        for i in 1:(j - 1)
            R[i, j] = random_bigint(rng, bits)
        end
        d = big(0)
        while d == 0
            d = random_bigint(rng, bits)
        end
        R[j, j] = d
    end
    return R
end

# ---------------------------------------------------------------------------

@testset "size_reduction_triu" begin

    @testset "elementary, exact rounding (:zz)" begin
        rng = MersenneTwister(0xBEEF)

        @testset "Int64, n = $n" for n in 1:12
            R0 = random_upper_triangular(rng, Int64, n)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test check_reduction(R0, Rred, U)
            @test eltype(Rred) === Int64 && eltype(U) === Int64
        end

        @testset "BigInt, n = $n" for n in (1, 2, 5, 13, 20)
            R0 = random_upper_triangular_big(rng, n; bits = 160)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test check_reduction(R0, Rred, U)
        end

        @testset "Int32 and Int128" begin
            R0 = random_upper_triangular(rng, Int32, 9)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test check_reduction(R0, Rred, U)
            @test eltype(U) === Int32

            R0 = random_upper_triangular(rng, Int128, 9)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test check_reduction(R0, Rred, U)
        end

        @testset "negative diagonals are handled" begin
            R0 = BigInt[-7 100 -250; 0 -11 400; 0 0 -5]
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test check_reduction(R0, Rred, U)
        end

        @testset "the input is not modified" begin
            R0 = random_upper_triangular(rng, Int64, 10)
            keep = copy(R0)
            Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test R0 == keep
        end

        @testset "an already reduced matrix is a fixed point" begin
            # Strictly reduced, so every multiplier rounds to zero.
            R0 = BigInt[100 3 -4; 0 200 7; 0 0 50]
            @test is_size_reduced(R0)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test Rred == R0
            @test U == BigInt[1 0 0; 0 1 0; 0 0 1]
        end
    end

    @testset "elementary, floating point rounding (:ll)" begin
        rng = MersenneTwister(0xF10A7)

        @testset "Int64, n = $n" for n in 1:12
            R0 = random_upper_triangular(rng, Int64, n)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :ll)
            @test check_reduction(R0, Rred, U)
        end

        # Entries well inside the Float64 mantissa, where the quotient is
        # computed without loss, so :ll must agree with :zz except at exact
        # halves, where round-to-even and round-half-away can differ.
        @testset "both roundings give valid reductions" begin
            for trial in 1:20
                R0 = random_upper_triangular(rng, Int64, 10;
                                             magnitude = 97, diagonal_magnitude = 1001)
                Rll, Ull = Flatter.size_reduction_triu(R0; algorithm = :ll)
                Rzz, Uzz = Flatter.size_reduction_triu(R0; algorithm = :zz)
                @test check_reduction(R0, Rll, Ull)
                # Both are valid reductions even where they disagree.
                @test is_size_reduced(Rll) && is_size_reduced(Rzz)
            end
        end

        @testset "BigInt promotes to BigFloat rather than overflowing" begin
            R0 = random_upper_triangular_big(rng, 8; bits = 300)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :ll)
            @test check_reduction(R0, Rred, U)
        end
    end

    @testset "blocked" begin
        rng = MersenneTwister(0x8107CED)

        # The blocked kernel is not a reordering of the elementary one: it
        # applies U(j,j) to the original tile rather than to the already
        # reduced one, so it lands on a different reduced representative for
        # many inputs.  What it must satisfy is the contract, not equality.
        @testset "contract holds, n = $n, blocksize = $bs" for n in (1, 2, 3, 7, 16, 33),
                                                               bs in (1, 2, 3, 5, 8)
            R0 = random_upper_triangular_big(rng, n; bits = 96)
            Rred, U = Flatter.size_reduction_triu(R0; algorithm = :blocked, blocksize = bs)
            @test check_reduction(R0, Rred, U)
        end

        @testset "Int64 entries" begin
            for n in (5, 11, 20), bs in (2, 4, 7)
                R0 = random_upper_triangular(rng, Int64, n)
                Rred, U = Flatter.size_reduction_triu(R0; algorithm = :blocked, blocksize = bs)
                @test check_reduction(R0, Rred, U)
            end
        end

        @testset "delegates to :zz when n <= blocksize" begin
            R0 = random_upper_triangular_big(rng, 12; bits = 80)
            Rb, Ub = Flatter.size_reduction_triu(R0; algorithm = :blocked, blocksize = 12)
            Rz, Uz = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test Rb == Rz
            @test Ub == Uz
        end

        # Strassen over the integers is exact, so forcing the recursion on
        # must reproduce the plain multiplication exactly.  A mismatch here
        # would point at the tile products, not at the reduction logic.
        @testset "the result does not depend on strassen_cutoff" begin
            R0 = random_upper_triangular_big(rng, 40; bits = 80)
            Rlow, Ulow = Flatter.size_reduction_triu(R0; algorithm = :blocked,
                                             blocksize = 10, strassen_cutoff = 2)
            Rhigh, Uhigh = Flatter.size_reduction_triu(R0; algorithm = :blocked,
                                               blocksize = 10, strassen_cutoff = 4096)
            @test Rlow == Rhigh && Ulow == Uhigh
            @test check_reduction(R0, Rlow, Ulow)
        end

        @testset "diagonal is preserved up to sign structure" begin
            # Column operations never touch the diagonal of an upper
            # triangular matrix, since they only add earlier columns.
            R0 = random_upper_triangular_big(rng, 21; bits = 64)
            Rred, _ = Flatter.size_reduction_triu(R0; algorithm = :blocked, blocksize = 5)
            @test [Rred[i, i] for i in 1:21] == [R0[i, i] for i in 1:21]
        end
    end

    @testset "dispatcher and argument checking" begin
        rng = MersenneTwister(0x0FF1CE)

        @testset ":auto picks blocked above the threshold" begin
            R0 = random_upper_triangular_big(rng, 30; bits = 64)
            Ra, Ua = Flatter.size_reduction_triu(R0; blocksize = 6)                     # :auto
            Rb, Ub = Flatter.size_reduction_triu(R0; algorithm = :blocked, blocksize = 6)
            @test Ra == Rb && Ua == Ub

            Rc, Uc = Flatter.size_reduction_triu(R0; blocksize = 100)                   # :auto
            Rz, Uz = Flatter.size_reduction_triu(R0; algorithm = :zz)
            @test Rc == Rz && Uc == Uz
        end

        @testset "in-place form" begin
            R0 = random_upper_triangular(rng, Int64, 9)
            R = copy(R0)
            U = Matrix{Int64}(undef, 9, 9)
            ret = Flatter.size_reduction_triu!(R, U; algorithm = :zz)
            @test ret === (R, U)
            @test check_reduction(R0, R, U)
        end

        @testset "works on views" begin
            R0 = random_upper_triangular_big(rng, 14; bits = 64)
            big_R = zeros(BigInt, 20, 20)
            big_U = zeros(BigInt, 20, 20)
            big_R[4:17, 4:17] .= R0
            Flatter.size_reduction_triu!(view(big_R, 4:17, 4:17), view(big_U, 4:17, 4:17);
                                 algorithm = :blocked, blocksize = 4,
                                 strassen_cutoff = 2)
            @test check_reduction(R0, big_R[4:17, 4:17], big_U[4:17, 4:17])
        end

        @testset "rejects bad input" begin
            good = BigInt[3 1; 0 5]
            U2 = Matrix{BigInt}(undef, 2, 2)

            @test_throws DimensionMismatch Flatter.size_reduction_triu(BigInt[1 2 3; 0 4 5])
            @test_throws DimensionMismatch Flatter.size_reduction_triu!(good, Matrix{BigInt}(undef, 3, 3))
            @test_throws ArgumentError Flatter.size_reduction_triu(BigInt[3 1; 2 5])      # not triangular
            @test_throws ArgumentError Flatter.size_reduction_triu(BigInt[0 1; 0 5])      # zero pivot
            @test_throws ArgumentError Flatter.size_reduction_triu(good; algorithm = :nope)
            @test_throws ArgumentError Flatter.size_reduction_triu(good; blocksize = 0)
        end
    end

    @testset "randomised sweep over all three kernels" begin
        rng = MersenneTwister(2026_08_11)
        for trial in 1:150
            n = rand(rng, 1:24)
            bs = rand(rng, 1:9)
            R0 = rand(rng, Bool) ?
                 random_upper_triangular(rng, Int64, n) :
                 random_upper_triangular_big(rng, n; bits = rand(rng, 32:192))
            for alg in (:zz, :ll, :blocked)
                Rred, U = Flatter.size_reduction_triu(R0; algorithm = alg, blocksize = bs)
                @test check_reduction(R0, Rred, U)
            end
        end
    end

end
