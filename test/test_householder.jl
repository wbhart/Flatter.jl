using Test
using Random
using LinearAlgebra
using Flatter

function test_matrix(rng::AbstractRNG, m::Int, n::Int)
    A = Matrix{BigFloat}(undef, m, n)
    denominator = BigFloat(997)
    for index in eachindex(A)
        A[index] = BigFloat(rand(rng, -1000:1000)) / denominator
    end
    return A
end

function unpack_compact_wy(result)
    factors = result.factors
    compact_T = result.T
    m, n = size(factors)
    r = min(m, n)

    V = zeros(BigFloat, m, r)
    for j in 1:r
        V[j, j] = one(BigFloat)
        for i in j+1:m
            V[i, j] = factors[i, j]
        end
    end

    R = zeros(BigFloat, m, n)
    for j in 1:n
        for i in 1:min(j, m)
            R[i, j] = factors[i, j]
        end
    end

    Q = Matrix{BigFloat}(I, m, m) - V * compact_T * transpose(V)
    return Q, R, V
end

function check_householder(A::Matrix{BigFloat}; tolerance::BigFloat)
    original = copy(A)
    result = Flatter.householder(A)

    # The non-mutating interface must leave its input unchanged.
    @test A == original

    m, n = size(A)
    r = min(m, n)
    @test size(result.factors) == (m, n)
    @test size(result.T) == (r, r)

    # T must be upper triangular exactly, not merely approximately.
    for j in 1:r
        for i in j+1:r
            @test iszero(result.T[i, j])
        end
    end

    Q, R, _ = unpack_compact_wy(result)
    identity_m = Matrix{BigFloat}(I, m, m)

    matrix_scale = max(norm(original), one(BigFloat))
    orthogonality_scale = BigFloat(max(m, 1))

    @test norm(original - Q * R) <= tolerance * matrix_scale
    @test norm(transpose(Q) * original - R) <= tolerance * matrix_scale
    @test norm(transpose(Q) * Q - identity_m) <=
          tolerance * orthogonality_scale
end

@testset "unblocked Householder QR in compact-WY form" begin
    setprecision(BigFloat, 200) do
        rng = MersenneTwister(0x484f555345484f4c)

        # This leaves about 50 guard bits relative to the 200-bit working
        # precision, while still requiring roughly 45 correct decimal digits.
        tolerance = BigFloat(2)^(-150)

        @testset "all square dimensions through 32" begin
            for n in 1:32
                check_householder(test_matrix(rng, n, n); tolerance = tolerance)
            end
        end

        @testset "rectangular matrices" begin
            for (m, n) in ((2, 1), (3, 2), (8, 3), (16, 7), (32, 16),
                           (1, 2), (2, 3), (5, 8), (7, 16), (16, 32))
                check_householder(test_matrix(rng, m, n); tolerance = tolerance)
            end
        end

        @testset "structured and rank-deficient inputs" begin
            check_householder(zeros(BigFloat, 8, 5); tolerance = tolerance)
            check_householder(Matrix{BigFloat}(I, 12, 12); tolerance = tolerance)

            diagonal = zeros(BigFloat, 10, 10)
            for i in 1:10
                diagonal[i, i] = isodd(i) ? BigFloat(i) : -BigFloat(i)
            end
            check_householder(diagonal; tolerance = tolerance)

            rank_deficient = test_matrix(rng, 12, 8)
            rank_deficient[:, 7] .= rank_deficient[:, 3]
            rank_deficient[:, 8] .= zero(BigFloat)
            check_householder(rank_deficient; tolerance = tolerance)
        end

        @testset "allocation-controlled in-place kernel" begin
            original = test_matrix(rng, 17, 9)
            factors = copy(original)
            r = min(size(factors)...)
            compact_T = zeros(BigFloat, r, r)
            work = zeros(BigFloat, r)

            result = Flatter.householder!(factors, compact_T, work)
            @test result.factors === factors
            @test result.T === compact_T

            Q, R, _ = unpack_compact_wy(result)
            scale = max(norm(original), one(BigFloat))
            @test norm(original - Q * R) <= tolerance * scale
        end

        @testset "workspace dimension checks" begin
            A = test_matrix(rng, 6, 4)
            @test_throws DimensionMismatch Flatter.householder!(
                copy(A), zeros(BigFloat, 3, 4), zeros(BigFloat, 4)
            )
            @test_throws DimensionMismatch Flatter.householder!(
                copy(A), zeros(BigFloat, 4, 4), zeros(BigFloat, 3)
            )
        end
    end
end

# Recursive compact-WY Householder QR tests with square-Strassen-backed
# rectangular updates.

function check_householder_block(
    A::Matrix{BigFloat};
    cutoff::Int,
    strassen_cutoff::Int = Flatter.DEFAULT_STRASSEN_CUTOFF,
    tolerance::BigFloat,
)
    original = copy(A)
    result = Flatter.householder_block(
        A;
        cutoff = cutoff,
        strassen_cutoff = strassen_cutoff,
    )

    # The non-mutating interface must leave its input unchanged.
    @test A == original

    m, n = size(A)
    r = min(m, n)
    @test size(result.factors) == (m, n)
    @test size(result.T) == (r, r)

    # T is exactly upper triangular.
    for j in 1:r
        for i in j + 1:r
            @test iszero(result.T[i, j])
        end
    end

    Q, R, _ = unpack_compact_wy(result)
    identity_m = Matrix{BigFloat}(I, m, m)

    matrix_scale = max(norm(original), one(BigFloat))
    orthogonality_scale = BigFloat(max(m, 1))

    @test norm(original - Q * R) <= tolerance * matrix_scale
    @test norm(transpose(Q) * original - R) <= tolerance * matrix_scale
    @test norm(transpose(Q) * Q - identity_m) <=
          tolerance * orthogonality_scale
end

@testset "recursive Householder QR in compact-WY form" begin
    setprecision(BigFloat, 200) do
        rng = MersenneTwister(0x424c4f434b515257)
        tolerance = BigFloat(2)^(-140)

        @test Flatter.DEFAULT_QR_BLOCK_CUTOFF == 32

        @testset "all square dimensions through 32" begin
            # A cutoff of one forces QR recursion down to single-reflector
            # leaves.  The default Strassen cutoff keeps this exhaustive test
            # reasonably small while separately tested cases below force the
            # square-Strassen path.
            for n in 1:32
                check_householder_block(
                    test_matrix(rng, n, n);
                    cutoff = 1,
                    tolerance = tolerance,
                )
            end
        end

        @testset "square-Strassen-backed update path" begin
            # A small Strassen cutoff forces both exact square products and
            # square-padded blocks arising from rectangular products through
            # Flatter.strassen!.
            for (m, n) in (
                (8, 8),
                (15, 15),
                (16, 16),
                (24, 16),
                (16, 24),
                (31, 31),
                (32, 32),
            )
                check_householder_block(
                    test_matrix(rng, m, n);
                    cutoff = 2,
                    strassen_cutoff = 2,
                    tolerance = tolerance,
                )
            end
        end

        @testset "several QR recursion cutoffs" begin
            for cutoff in (2, 4, 8, 16, 32)
                check_householder_block(
                    test_matrix(rng, 32, 32);
                    cutoff = cutoff,
                    tolerance = tolerance,
                )
            end

            # Exercise both default cutoff keywords explicitly.
            A = test_matrix(rng, 32, 32)
            result = Flatter.householder_block(A)
            Q, R, _ = unpack_compact_wy(result)
            @test norm(A - Q * R) <=
                  tolerance * max(norm(A), one(BigFloat))
        end

        @testset "tall and wide matrices" begin
            for (m, n, cutoff) in (
                (2, 1, 1),
                (3, 2, 1),
                (8, 3, 1),
                (16, 7, 2),
                (32, 16, 4),
                (1, 2, 1),
                (2, 3, 1),
                (5, 8, 1),
                (7, 16, 2),
                (16, 32, 4),
            )
                check_householder_block(
                    test_matrix(rng, m, n);
                    cutoff = cutoff,
                    tolerance = tolerance,
                )
            end
        end

        @testset "structured and rank-deficient inputs" begin
            check_householder_block(
                zeros(BigFloat, 16, 12);
                cutoff = 2,
                strassen_cutoff = 2,
                tolerance = tolerance,
            )
            check_householder_block(
                Matrix{BigFloat}(I, 32, 32);
                cutoff = 4,
                strassen_cutoff = 2,
                tolerance = tolerance,
            )

            diagonal = zeros(BigFloat, 32, 32)
            for i in 1:32
                diagonal[i, i] = isodd(i) ? BigFloat(i) : -BigFloat(i)
            end
            check_householder_block(
                diagonal;
                cutoff = 4,
                strassen_cutoff = 2,
                tolerance = tolerance,
            )

            rank_deficient = test_matrix(rng, 24, 16)
            rank_deficient[:, 13] .= rank_deficient[:, 3]
            rank_deficient[:, 14] .= rank_deficient[:, 7]
            rank_deficient[:, 15] .= zero(BigFloat)
            rank_deficient[:, 16] .= zero(BigFloat)
            check_householder_block(
                rank_deficient;
                cutoff = 2,
                strassen_cutoff = 2,
                tolerance = tolerance,
            )
        end

        @testset "caller-provided QR workspaces" begin
            original = test_matrix(rng, 23, 17)
            factors = copy(original)
            m, n = size(factors)
            r = min(m, n)
            cutoff = 3
            split = div(r, 2)

            compact_T = zeros(BigFloat, r, r)
            block_work = zeros(BigFloat, split, n - split)
            vector_work = zeros(BigFloat, r)

            result = Flatter.householder_block!(
                factors,
                compact_T,
                block_work,
                vector_work;
                cutoff = cutoff,
                strassen_cutoff = 2,
            )

            @test result.factors === factors
            @test result.T === compact_T

            Q, R, _ = unpack_compact_wy(result)
            scale = max(norm(original), one(BigFloat))
            @test norm(original - Q * R) <= tolerance * scale
        end

        @testset "fully workspace-controlled kernel" begin
            original = test_matrix(rng, 20, 14)
            factors = copy(original)
            m, n = size(factors)
            r = min(m, n)
            cutoff = 2
            strassen_cutoff = 2
            split = div(r, 2)

            compact_T = zeros(BigFloat, r, r)
            block_work = zeros(BigFloat, split, n - split)
            vector_work = zeros(BigFloat, r)
            multiply_work = Flatter._householder_block_multiply_workspace(
                BigFloat,
                split,
                strassen_cutoff,
            )

            result = Flatter.householder_block!(
                factors,
                compact_T,
                block_work,
                vector_work,
                multiply_work;
                cutoff = cutoff,
                strassen_cutoff = strassen_cutoff,
            )

            @test result.factors === factors
            @test result.T === compact_T

            Q, R, _ = unpack_compact_wy(result)
            scale = max(norm(original), one(BigFloat))
            @test norm(original - Q * R) <= tolerance * scale
        end

        @testset "base-case-only workspace may be empty" begin
            original = test_matrix(rng, 12, 9)
            factors = copy(original)
            compact_T = zeros(BigFloat, 9, 9)

            result = Flatter.householder_block!(
                factors,
                compact_T,
                zeros(BigFloat, 0, 0),
                zeros(BigFloat, 9);
                cutoff = 9,
                strassen_cutoff = 2,
            )

            Q, R, _ = unpack_compact_wy(result)
            @test norm(original - Q * R) <=
                  tolerance * max(norm(original), one(BigFloat))
        end

        @testset "argument and workspace checks" begin
            A = test_matrix(rng, 8, 6)

            @test_throws ArgumentError Flatter.householder_block(
                A;
                cutoff = 0,
            )
            @test_throws ArgumentError Flatter.householder_block(
                A;
                cutoff = -1,
            )
            @test_throws ArgumentError Flatter.householder_block(
                A;
                strassen_cutoff = 0,
            )

            @test_throws DimensionMismatch Flatter.householder_block!(
                copy(A),
                zeros(BigFloat, 5, 6);
                cutoff = 2,
            )

            # Here r = 6 and split = 3, so block_work must be at least 3-by-3.
            @test_throws DimensionMismatch Flatter.householder_block!(
                copy(A),
                zeros(BigFloat, 6, 6),
                zeros(BigFloat, 2, 3),
                zeros(BigFloat, 6);
                cutoff = 2,
            )
            @test_throws DimensionMismatch Flatter.householder_block!(
                copy(A),
                zeros(BigFloat, 6, 6),
                zeros(BigFloat, 3, 2),
                zeros(BigFloat, 6);
                cutoff = 2,
            )
            @test_throws DimensionMismatch Flatter.householder_block!(
                copy(A),
                zeros(BigFloat, 6, 6),
                zeros(BigFloat, 3, 3),
                zeros(BigFloat, 5);
                cutoff = 2,
            )
        end
    end
end

@testset "forced recursion through dimension 32" begin
    rng = Xoshiro(0x12345678)

    setprecision(BigFloat, 200) do
        tolerance = BigFloat(2)^(-140)

        for n in 1:32
            check_householder_block(
                test_matrix(rng, n, n);
                cutoff = 1,
                tolerance = tolerance,
            )
        end
    end
end

@testset "default cutoff boundary" begin
    rng = Xoshiro(0x23456789)

    setprecision(BigFloat, 200) do
        tolerance = BigFloat(2)^(-140)

        for n in (31, 32, 33, 64, 65, 66)
            check_householder_block(
                test_matrix(rng, n, n);
                cutoff = Flatter.DEFAULT_QR_BLOCK_CUTOFF,
                tolerance = tolerance,
            )
        end
    end
end

@testset "default cutoff rectangular cases" begin
    rng = Xoshiro(0x3456789a)

    setprecision(BigFloat, 200) do
        tolerance = BigFloat(2)^(-140)

        for (m, n) in (
            (66, 33),
            (80, 48),
            (48, 80),
        )
            check_householder_block(
                test_matrix(rng, m, n);
                cutoff = Flatter.DEFAULT_QR_BLOCK_CUTOFF,
                tolerance = tolerance,
            )
        end
    end
end

# Testset for apply_Qt!, to add to test/test_householder.jl.
#
# Assumes `using Test` and `using Random` from runtests.jl. Everything from the
# package is called qualified as `Flatter.<name>`, and no other imports are
# needed: matrix multiplication and `permutedims` come from Base, and the
# oracles below avoid `LinearAlgebra.I` and `LinearAlgebra.norm` so that this
# file adds no dependency of its own.

# Rebuild V explicitly from the packed factors: column j is zero above row j,
# one at row j, and the stored tail below.
function apply_qt_explicit_V(factors::AbstractMatrix{S}, k::Integer) where {S}
    m = size(factors, 1)
    V = zeros(S, m, k)
    for j in 1:k
        V[j, j] = one(S)
        for i in (j + 1):m
            V[i, j] = factors[i, j]
        end
    end
    return V
end

function apply_qt_identity(::Type{S}, n::Integer) where {S}
    M = zeros(S, n, n)
    for i in 1:n
        M[i, i] = one(S)
    end
    return M
end

# Q' = I - V*T'*V', built independently of the routine under test.
function apply_qt_explicit_Qt(factors::AbstractMatrix{S},
                              compact_T::AbstractMatrix{S},
                              k::Integer) where {S}
    m = size(factors, 1)
    V = apply_qt_explicit_V(factors, k)
    return apply_qt_identity(S, m) -
           V * permutedims(compact_T[1:k, 1:k]) * permutedims(V)
end

apply_qt_maxabs(A) = isempty(A) ? zero(eltype(A)) : maximum(abs, A)

@testset "apply_Qt!" begin

    @testset "reproduces R, $m x $n" for (m, n) in ((8, 5), (40, 40), (64, 17), (5, 5))
        rng = MersenneTwister(hash((m, n)))
        A = randn(rng, m, n)
        F, T = Flatter.householder_block(A)

        C = copy(A)
        Flatter.apply_Qt!(C, F, T)

        k = min(m, n)
        scale = max(1.0, apply_qt_maxabs(A))

        # The upper trapezoid must match the R stored in the packed factors,
        # and everything below the diagonal must have been annihilated.
        for j in 1:n, i in 1:min(j, k)
            @test abs(C[i, j] - F[i, j]) <= 1e-9 * scale
        end
        for j in 1:n, i in (min(j, k) + 1):m
            @test abs(C[i, j]) <= 1e-9 * scale
        end
    end

    @testset "matches the explicit Q', $m x $n" for (m, n) in ((9, 4), (48, 30), (33, 33))
        rng = MersenneTwister(hash((m, n, :explicit)))
        A = randn(rng, m, n)
        F, T = Flatter.householder_block(A)
        k = min(m, n)
        Qt = apply_qt_explicit_Qt(F, T, k)

        for n2 in (1, 3, n, 2n)
            C = randn(rng, m, n2)
            expected = Qt * C
            got = Flatter.apply_Qt!(copy(C), F, T)
            @test apply_qt_maxabs(got - expected) <=
                  1e-9 * max(1.0, apply_qt_maxabs(expected))
        end
    end

    @testset "partial application" begin
        rng = MersenneTwister(0xA11)
        m, n = 20, 12
        A = randn(rng, m, n)
        F, T = Flatter.householder_block(A)

        for k in (0, 1, 5, n)
            C = randn(rng, m, 7)
            expected = k == 0 ? copy(C) : apply_qt_explicit_Qt(F, T, k) * C
            got = Flatter.apply_Qt!(copy(C), F, T; reflectors = k)
            @test apply_qt_maxabs(got - expected) <=
                  1e-9 * max(1.0, apply_qt_maxabs(expected))
        end
    end

    @testset "Q' is orthogonal" begin
        rng = MersenneTwister(0x0B7)
        m, n = 30, 18
        A = randn(rng, m, n)
        F, T = Flatter.householder_block(A)

        identity_m = apply_qt_identity(Float64, m)
        Qt = Flatter.apply_Qt!(copy(identity_m), F, T)

        @test apply_qt_maxabs(Qt * permutedims(Qt) - identity_m) <= 1e-9

        # Q'A = R, so applying Q back must recover A.
        @test apply_qt_maxabs(permutedims(Qt) * (Qt * A) - A) <=
              1e-9 * max(1.0, apply_qt_maxabs(A))
    end

    @testset "agrees with the unblocked factorization" begin
        rng = MersenneTwister(0x11B)
        m, n = 25, 14
        A = randn(rng, m, n)
        C = randn(rng, m, 6)

        Fb, Tb = Flatter.householder_block(A)
        Fu, Tu = Flatter.householder(A)

        blocked = Flatter.apply_Qt!(copy(C), Fb, Tb)
        unblocked = Flatter.apply_Qt!(copy(C), Fu, Tu)
        @test apply_qt_maxabs(blocked - unblocked) <=
              1e-9 * max(1.0, apply_qt_maxabs(unblocked))
    end

    @testset "strassen_cutoff does not change the result" begin
        rng = MersenneTwister(0x57A)
        m, n = 70, 40
        A = randn(rng, m, n)
        F, T = Flatter.householder_block(A)
        C = randn(rng, m, 50)

        reference = Flatter.apply_Qt!(copy(C), F, T; strassen_cutoff = 4096)
        scale = max(1.0, apply_qt_maxabs(reference))
        for cutoff in (2, 8, 32)
            got = Flatter.apply_Qt!(copy(C), F, T; strassen_cutoff = cutoff)
            @test apply_qt_maxabs(got - reference) <= 1e-8 * scale
        end
    end

    @testset "BigFloat at fixed precision" begin
        setprecision(BigFloat, 128) do
            rng = MersenneTwister(0xB16)
            m, n = 18, 11
            A = BigFloat.(randn(rng, m, n))
            F, T = Flatter.householder_block(A)
            k = min(m, n)

            C = BigFloat.(randn(rng, m, 5))
            expected = apply_qt_explicit_Qt(F, T, k) * C
            got = Flatter.apply_Qt!(copy(C), F, T)

            tolerance = BigFloat(2)^(-100) * max(one(BigFloat), apply_qt_maxabs(A))
            @test apply_qt_maxabs(got - expected) < tolerance

            # The routine must not have widened the output. Under Julia's
            # rebinding semantics this is the failure to watch for.
            @test Flatter.uniform_precision(got) == 128
        end
    end

    @testset "argument checking" begin
        A = randn(MersenneTwister(1), 10, 6)
        F, T = Flatter.householder_block(A)
        @test_throws DimensionMismatch Flatter.apply_Qt!(randn(9, 3), F, T)
        @test_throws DimensionMismatch Flatter.apply_Qt!(randn(10, 3), F, T;
                                                         reflectors = 7)
    end

end

