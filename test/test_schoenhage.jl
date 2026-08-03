function _test_schoenhage_det2(A::AbstractMatrix{<:Integer})
    return A[1, 1]*A[2, 2] - A[1, 2]*A[2, 1]
end

function _test_schoenhage_congruence(
    G::AbstractMatrix{<:Integer},
    U::AbstractMatrix{<:Integer},
)
    a = BigInt(G[1, 1])
    h = BigInt(G[1, 2])
    c = BigInt(G[2, 2])

    u = BigInt(U[1, 1])
    v = BigInt(U[1, 2])
    w = BigInt(U[2, 1])
    z = BigInt(U[2, 2])

    r11 = a*u*u + 2*h*u*w + c*w*w
    r12 = a*u*v + h*(u*z + v*w) + c*w*z
    r22 = a*v*v + 2*h*v*z + c*z*z
    return BigInt[r11 r12; r12 r22]
end

function _test_schoenhage_check_result(G; cutoff=200)
    R, U = Flatter.schoenhage(G; cutoff=cutoff)

    @test eltype(R) == BigInt
    @test eltype(U) == BigInt
    @test R == _test_schoenhage_congruence(G, U)
    @test R[1, 2] == R[2, 1]
    @test R[1, 1] > 0
    @test _test_schoenhage_det2(R) > 0
    @test R[1, 1] <= R[2, 2]
    @test 2*abs(R[1, 2]) <= R[1, 1]
    @test abs(_test_schoenhage_det2(U)) == 1
    @test _test_schoenhage_det2(R) == _test_schoenhage_det2(G)

    return R, U
end

function _test_schoenhage_random_unimodular(rng, steps)
    U = BigInt[1 0; 0 1]

    for _ in 1:steps
        operation = rand(rng, 1:3)
        t = BigInt(rand(rng, -1000:1000))

        if operation == 1
            U[1, 2] += t*U[1, 1]
            U[2, 2] += t*U[2, 1]
        elseif operation == 2
            U[1, 1] += t*U[1, 2]
            U[2, 1] += t*U[2, 2]
        else
            u11 = U[1, 1]
            u21 = U[2, 1]
            U[1, 1] = U[1, 2]
            U[2, 1] = U[2, 2]
            U[1, 2] = u11
            U[2, 2] = u21
        end
    end

    return U
end

@testset "Schoenhage reduction" begin
    @testset "basic examples" begin
        G = BigInt[4 1; 1 7]
        R, U = _test_schoenhage_check_result(G)
        @test R == G
        @test U == BigInt[1 0; 0 1]

        G = BigInt[4 -1; -1 7]
        R, U = _test_schoenhage_check_result(G)
        @test R == BigInt[4 1; 1 7]
        @test U == BigInt[1 0; 0 -1]

        _test_schoenhage_check_result(BigInt[5 2; 2 3])
        _test_schoenhage_check_result(BigInt[100 99; 99 100])
        _test_schoenhage_check_result(BigInt[13 8; 8 5])
        _test_schoenhage_check_result(BigInt[1 0; 0 1])

        # Integer inputs are accepted and converted to BigInt.
        R, U = _test_schoenhage_check_result([10 7; 7 6])
        @test eltype(R) == BigInt
        @test eltype(U) == BigInt
    end

    @testset "large recursive input" begin
        # This unimodular matrix has entries large enough to take the default
        # cutoff through the recursive path.
        n = (BigInt(1) << 320) + 123456789
        V = BigInt[1 n; n n*n + 1]
        @test _test_schoenhage_det2(V) == 1

        reduced_seed = BigInt[97 13; 13 131]
        G = _test_schoenhage_congruence(reduced_seed, V)
        _test_schoenhage_check_result(G)
    end

    @testset "random unimodular equivalents" begin
        rng = MersenneTwister(0x5c40e19a)

        for _ in 1:100
            a = BigInt(rand(rng, 1:500))
            h = BigInt(rand(rng, -div(Int(a), 2):div(Int(a), 2)))
            c = BigInt(rand(rng, Int(a):Int(a) + 500))
            reduced_seed = BigInt[a h; h c]

            V = _test_schoenhage_random_unimodular(rng, 16)
            @test abs(_test_schoenhage_det2(V)) == 1

            G = _test_schoenhage_congruence(reduced_seed, V)
            _test_schoenhage_check_result(G; cutoff=24)
        end
    end

    @testset "input validation" begin
        @test_throws DimensionMismatch Flatter.schoenhage(BigInt[1 0 0; 0 1 0])
        @test_throws ArgumentError Flatter.schoenhage(BigInt[2 1; 0 2])
        @test_throws ArgumentError Flatter.schoenhage(BigInt[1 2; 2 1])
        @test_throws ArgumentError Flatter.schoenhage(BigInt[0 0; 0 1])
        @test_throws ArgumentError Flatter.schoenhage(BigInt[1 0; 0 1]; cutoff=1)
    end
end
