# test_smsv.jl
#
# Tests for the column scaling of Algorithm 2 (Saruchi-Morel-Stehle-Villard).
#
# These tests do not call fpLLL. They use an exact rational LLL defined below,
# which is "well-behaved" in the sense the paper requires (it swaps only when
# the Lovasz condition fails, hence only when r_ii decreases), so Theorem 2
# applies to it. That keeps the test suite independent of the shim, and gives
# an implementation of Step 9 whose behaviour is fully known.

using Random
using LinearAlgebra: I, det

# --------------------------------------------------------------------------
# Exact rational Gram-Schmidt, used both by the reference LLL and the checkers
# --------------------------------------------------------------------------

"Round a rational to the nearest integer, halves upwards."
function smsv_round(value::Rational{BigInt})
    return fld(2 * numerator(value) + denominator(value), 2 * denominator(value))
end

"""
Gram-Schmidt orthogonalisation of a column-convention basis over the rationals.
Returns `(orthogonal, mu, norms)` where `norms[i]` is the squared norm of the
`i`th orthogonalised vector and `mu[j, i]` is the coefficient of `b*_j` in `b_i`.
"""
function smsv_gso(B::Matrix{Rational{BigInt}})
    m, n = size(B)
    orthogonal = zeros(Rational{BigInt}, m, n)
    mu = zeros(Rational{BigInt}, n, n)
    norms = zeros(Rational{BigInt}, n)
    for i in 1:n
        vector = B[:, i]
        for j in 1:(i - 1)
            iszero(norms[j]) && continue
            mu[j, i] = sum(B[k, i] * orthogonal[k, j] for k in 1:m) / norms[j]
            vector -= mu[j, i] * orthogonal[:, j]
        end
        orthogonal[:, i] = vector
        norms[i] = sum(x -> x^2, vector)
    end
    return orthogonal, mu, norms
end

function smsv_swapcols!(A::AbstractMatrix, i::Int, j::Int)
    for k in 1:size(A, 1)
        A[k, i], A[k, j] = A[k, j], A[k, i]
    end
    return A
end

"""
    smsv_reference_lll(A; delta) -> (basis, transform)

An exact rational LLL in column convention, matching the `reducer` interface
expected by `scale_for_reduction`: returns `(basis, U)` with `basis == A * U`.
"""
function smsv_reference_lll(A::AbstractMatrix{<:Integer};
                            delta::Rational{BigInt}=Rational{BigInt}(99, 100))
    exact = Matrix{BigInt}(A)
    m, n = size(exact)
    n == 1 && return exact, Matrix{BigInt}(I, 1, 1)

    B = Matrix{Rational{BigInt}}(exact)
    U = Matrix{BigInt}(I, n, n)

    k = 2
    steps = 0
    while k <= n
        steps += 1
        steps > 200_000 && error("the reference LLL failed to terminate")

        _, mu, _ = smsv_gso(B)
        for j in (k - 1):-1:1
            q = smsv_round(mu[j, k])
            iszero(q) && continue
            for i in 1:m
                B[i, k] -= q * B[i, j]
            end
            for i in 1:n
                U[i, k] -= q * U[i, j]
            end
            _, mu, _ = smsv_gso(B)
        end

        _, mu, norms = smsv_gso(B)
        if norms[k] + mu[k - 1, k]^2 * norms[k - 1] >= delta * norms[k - 1]
            k += 1
        else
            smsv_swapcols!(B, k, k - 1)
            smsv_swapcols!(U, k, k - 1)
            k = max(k - 1, 2)
        end
    end
    return exact * U, U
end

"""
Check the size-reduction condition of Definition 1 of the paper,
`|r_ij| <= eta * r_ii + theta * r_jj`, in terms of Gram-Schmidt coefficients.

The `theta` slack is not optional: Lemma 5 of the paper is stated to be false
when `theta = 0`, and the output of Algorithm 2 is only claimed to be reduced
with respect to parameters carrying such a slack.
"""
function smsv_is_size_reduced(A::AbstractMatrix{<:Integer}; eta::Real=0.51, theta::Real=0.5)
    B = Matrix{Rational{BigInt}}(Matrix{BigInt}(A))
    _, mu, norms = smsv_gso(B)
    n = size(B, 2)
    for i in 1:(n - 1), j in (i + 1):n
        (iszero(norms[i]) || iszero(norms[j])) && return false
        ratio = sqrt(Float64(norms[j]) / Float64(norms[i]))
        abs(Float64(mu[i, j])) <= eta + theta * ratio || return false
    end
    return true
end

"Check the Lovasz conditions exactly."
function smsv_satisfies_lovasz(A::AbstractMatrix{<:Integer}; delta::Real=0.75)
    B = Matrix{Rational{BigInt}}(Matrix{BigInt}(A))
    _, mu, norms = smsv_gso(B)
    bound = Rational{BigInt}(rationalize(BigInt, Float64(delta)))
    for i in 1:(size(B, 2) - 1)
        norms[i + 1] + mu[i, i + 1]^2 * norms[i] >= bound * norms[i] || return false
    end
    return true
end

smsv_is_unimodular(V::AbstractMatrix{<:Integer}) =
    abs(det(Matrix{Rational{BigInt}}(Matrix{BigInt}(V)))) == 1

"Largest entry bit-length of an integer matrix."
smsv_bits(A::AbstractMatrix{<:Integer}) =
    maximum(x -> iszero(x) ? 0 : ndigits(BigInt(x); base=2), A; init=0)

# --------------------------------------------------------------------------
# Test inputs
# --------------------------------------------------------------------------

"A small nonsingular integer matrix with entries of about `bits` bits."
function smsv_random_basis(n::Int, bits::Int; seed::Int=0)
    generator = MersenneTwister(seed)
    limit = BigInt(1) << bits
    for _ in 1:200
        A = Matrix{BigInt}(undef, n, n)
        for i in eachindex(A)
            A[i] = rand(generator, -limit:limit)
        end
        iszero(det(Matrix{Rational{BigInt}}(A))) || return A
    end
    error("failed to generate a nonsingular basis")
end

"""
A basis in two blocks whose magnitudes differ by `gap` bits. This is the
situation Algorithm 2 exists for: an unbalanced basis whose naive integer
representation is far larger than its floating-point representation.
"""
function smsv_blocked_basis(n::Int, bits::Int, gap::Int; seed::Int=0)
    half = n ÷ 2
    upper = smsv_random_basis(half, bits; seed=seed)
    lower = smsv_random_basis(n - half, bits; seed=seed + 1)
    A = zeros(BigInt, n, n)
    A[1:half, 1:half] = upper
    A[(half + 1):n, (half + 1):n] = lower .<< gap
    return A
end

# --------------------------------------------------------------------------

@testset "smsv" begin

    @testset "ScaledBasis" begin
        mantissa = BigInt[1 2; 3 4]

        @test size(Flatter.ScaledBasis(mantissa)) == (2, 2)
        @test Flatter.is_integral(Flatter.ScaledBasis(mantissa))
        @test Flatter.integer_matrix(Flatter.ScaledBasis(mantissa)) == mantissa

        scaled = Flatter.ScaledBasis(mantissa, [0, 3])
        @test Flatter.integer_matrix(scaled) == BigInt[1 16; 3 32]

        @test !Flatter.is_integral(Flatter.ScaledBasis(mantissa, [0, -2]))
        @test_throws ArgumentError Flatter.integer_matrix(Flatter.ScaledBasis(mantissa, [0, -2]))

        # More columns than rows cannot have full column rank.
        @test_throws ArgumentError Flatter.ScaledBasis(BigInt[1 2 3; 4 5 6])
        @test_throws ArgumentError Flatter.ScaledBasis(mantissa, [1])

        floats = Flatter.float_matrix(scaled, 128)
        @test precision(floats[1, 1]) == 128
        @test floats == BigFloat[1 16; 3 32]
    end

    @testset "apply_transform" begin
        basis = Flatter.ScaledBasis(BigInt[1 0; 0 1], [0, 5])
        V = BigInt[1 2; 0 1]
        result = Flatter.apply_transform(basis, V)

        # Exactness: the represented product must agree entrywise.
        @test Flatter.integer_matrix(result) == Flatter.integer_matrix(basis) * V
        # The result carries a single exponent, the minimum of the input's.
        @test allequal(result.exponent)
        @test result.exponent[1] == 0

        # A negative exponent is handled by the same lifting.
        negative = Flatter.ScaledBasis(BigInt[2 0; 0 2], [-3, 1])
        lifted = Flatter.apply_transform(negative, BigInt[1 1; 0 1])
        @test allequal(lifted.exponent)
        @test lifted.exponent[1] == -3
    end

    @testset "profile" begin
        # The R-factor of an upper triangular matrix is the matrix itself up to
        # signs, so the profile is the log of the diagonal.
        triangular = BigInt[8 3 1; 0 32 7; 0 0 2]
        prof = Flatter.profile(Flatter.ScaledBasis(triangular), 128)
        @test prof ≈ [3.0, 5.0, 1.0] atol = 1e-9
        @test Flatter.profile_spread(prof) ≈ 4.0 atol = 1e-9

        # Column exponents shift the profile by exactly those amounts.
        shifted = Flatter.profile(Flatter.ScaledBasis(triangular, [0, 10, 20]), 128)
        @test shifted ≈ [3.0, 15.0, 21.0] atol = 1e-9

        # A rank deficient basis is rejected rather than silently mishandled.
        deficient = BigInt[1 2 0; 0 1 0; 0 0 0]
        @test_throws ArgumentError Flatter.profile(Flatter.ScaledBasis(deficient), 128)

	# Blocking must not change the profile.
        wide = smsv_blocked_basis(8, 40, 500; seed=11)
        plain = Flatter.profile(Flatter.ScaledBasis(wide), 512; blocked=false)
        fast  = Flatter.profile(Flatter.ScaledBasis(wide), 512; blocked=true)
        @test plain ≈ fast rtol = 1e-6

	# What matters is the block decomposition, not the profile itself.
        for bits in (64, 96, 160, 320)
            plain = Flatter.compression_shifts(
                Flatter.profile(Flatter.ScaledBasis(wide), bits; blocked=false), 64)[1]
            fast = Flatter.compression_shifts(
                Flatter.profile(Flatter.ScaledBasis(wide), bits; blocked=true), 64)[1]
            @test plain == fast
        end
    end

    @testset "compression_shifts" begin
        # A flat profile is a single block: nothing to scale.
        shifts, truncation, spread = Flatter.compression_shifts(fill(100.0, 5), 40)
        @test all(iszero, shifts)
        @test spread ≈ 0.0 atol = 1e-9
        @test truncation == 100 - 40
        @test Flatter.blocks(shifts) == [1:5]

        # A single large gap is one block boundary, and the scaling removes the
        # gap down to a residue of about one bit.
        profile = [10.0, 10.0, 210.0, 210.0]
        shifts, _, spread = Flatter.compression_shifts(profile, 64)
        @test shifts == [0, 0, 199, 199]
        @test Flatter.blocks(shifts) == [1:2, 3:4]
        @test spread ≈ 1.0 atol = 1e-9

        # Two gaps accumulate, as the prefix sum of Step 6.
        shifts, _, _ = Flatter.compression_shifts([0.0, 100.0, 300.0], 64)
        @test issorted(shifts)
        @test shifts[1] == 0
        @test shifts == [0, 99, 298]

        # Gaps of at most one bit are not worth scaling.
        shifts, _, _ = Flatter.compression_shifts([0.0, 0.5, 1.0], 64)
        @test all(iszero, shifts)

        # A descending profile has no blocks at all: the running minimum from
        # the right never exceeds the running maximum from the left.
        shifts, _, spread = Flatter.compression_shifts([300.0, 200.0, 100.0], 64)
        @test all(iszero, shifts)
        @test spread ≈ 200.0 atol = 1e-9

        # Scaling never increases the spread.
        for profile in ([0.0, 500.0], [7.0, 7.0, 900.0], [1.0, 50.0, 99.0, 900.0])
            _, _, scaled_spread = Flatter.compression_shifts(profile, 64)
            @test scaled_spread <= Flatter.profile_spread(profile) + 1e-9
        end

        # Degenerate but legal.
        shifts, _, spread = Flatter.compression_shifts([42.0], 64)
        @test shifts == [0]
        @test spread ≈ 0.0 atol = 1e-9

        @test_throws ArgumentError Flatter.compression_shifts(Float64[], 64)
        @test_throws ArgumentError Flatter.compression_shifts([1.0, Inf], 64)
    end

    @testset "conjugate_transform" begin
        shifts = [0, 0, 10, 10]

        # Block upper triangular input: above-diagonal blocks pick up the shift,
        # within-block entries are untouched.
        U = BigInt[1 2 3 4; 0 1 5 6; 0 0 1 7; 0 0 0 1]
        V = Flatter.conjugate_transform(U, shifts)
        @test V[1, 2] == 2                 # same block, unchanged
        @test V[1, 3] == 3 * BigInt(2)^10  # across a block boundary
        @test V[2, 4] == 6 * BigInt(2)^10
        @test V[3, 4] == 7                 # same block again
        @test smsv_is_unimodular(V)

        # A nonzero entry below a block boundary would make the conjugation
        # non-integral. This is exactly the assertion in flatter's collect_U.
        violating = Matrix{BigInt}(I, 4, 4)
        violating[3, 1] = 1
        @test_throws ArgumentError Flatter.conjugate_transform(violating, shifts)

        # Non-decreasing shifts are a precondition.
        @test_throws ArgumentError Flatter.conjugate_transform(Matrix{BigInt}(I, 2, 2), [5, 0])
        @test_throws DimensionMismatch Flatter.conjugate_transform(Matrix{BigInt}(I, 2, 2), [0])
    end

    @testset "precision policies" begin
        # Step 1 of the paper: p = 10 + ceil(log2(m^3.5 chi)).
        @test Flatter.householder_precision(16, 1000) == 10 + ceil(Int, 3.5 * 4 + 1000)
        @test Flatter.householder_precision(4, 0) >= 53   # floored at double precision

        # The non-aggressive policy carries the linear term of Theorem 1.
        @test Flatter.lll_precision(100, 8) == 200 + 30 + 16
        @test Flatter.lll_precision(100, 8; aggressive=true) == 130
        @test Flatter.lll_precision(0, 4) == 38
    end

    @testset "end to end, balanced input" begin
        A = smsv_random_basis(6, 20; seed=1)
        result = Flatter.scale_for_reduction(A; reducer=smsv_reference_lll)

        @test smsv_is_unimodular(result.transform)
        @test result.scaled * result.inner == result.reduced
        # A balanced basis is one block, so the scaling is trivial and the
        # conjugation is the identity map on the transformation.
        @test Flatter.blocks(result.shifts) == [1:6]
        @test result.transform == result.inner

        reduced = Matrix{BigInt}(A) * result.transform
        @test smsv_is_size_reduced(reduced)
        @test smsv_satisfies_lovasz(reduced)
        # The lattice is preserved.
        @test abs(det(Matrix{Rational{BigInt}}(reduced))) ==
              abs(det(Matrix{Rational{BigInt}}(A)))
    end

    @testset "end to end, blocked input" begin
        A = smsv_blocked_basis(6, 20, 400; seed=2)
        basis = Flatter.ScaledBasis(A)
        result = Flatter.scale_for_reduction(basis; reducer=smsv_reference_lll)

        # The block structure is found, and the two blocks carry distinct shifts.
        @test length(result.blocks) >= 2
        @test issorted(result.shifts)
        @test result.shifts[1] == 0
        @test result.shifts[end] > 0

        # The whole point: the scaled matrix is far smaller than the input.
        @test smsv_bits(result.scaled) < smsv_bits(A) - 200

        # Theorem 2: the conjugated transformation is unimodular and reduces the
        # original basis.
        @test smsv_is_unimodular(result.transform)
        reduced = Matrix{BigInt}(A) * result.transform
        @test smsv_is_size_reduced(reduced)
        @test smsv_satisfies_lovasz(reduced)
        @test abs(det(Matrix{Rational{BigInt}}(reduced))) ==
              abs(det(Matrix{Rational{BigInt}}(A)))

        # The conjugation really is D^-1 U D.
        for i in 1:6, j in 1:6
            expected = result.shifts[j] >= result.shifts[i] ?
                result.inner[i, j] << (result.shifts[j] - result.shifts[i]) :
                BigInt(0)
            @test result.transform[i, j] == expected
        end
    end

    @testset "end to end, column exponents" begin
        # The same lattice presented in floating-point form, with the block gap
        # carried in the exponents rather than the mantissas.
        mantissa = smsv_blocked_basis(4, 18, 0; seed=3)
        exponent = [0, 0, 300, 300]
        basis = Flatter.ScaledBasis(mantissa, exponent)

        result = Flatter.scale_for_reduction(basis; reducer=smsv_reference_lll)
        @test smsv_is_unimodular(result.transform)
        @test length(result.blocks) >= 2

        # The scaled matrix stays close to the mantissa bit-size rather than
        # blowing up to the full integer representation.
        @test smsv_bits(result.scaled) < smsv_bits(Flatter.integer_matrix(basis)) - 200

        applied = Flatter.apply_transform(basis, result.transform)
        @test smsv_is_size_reduced(Flatter.integer_matrix(applied))
        @test smsv_satisfies_lovasz(Flatter.integer_matrix(applied))
    end

    @testset "single column" begin
        basis = Flatter.ScaledBasis(reshape(BigInt[3, 4], 2, 1))
        result = Flatter.scale_for_reduction(basis; reducer=smsv_reference_lll)
        @test result.shifts == [0]
        @test result.blocks == [1:1]
        @test smsv_is_unimodular(result.transform)
    end

    @testset "overrides and failure modes" begin
        A = smsv_random_basis(4, 30; seed=4)

        # An explicit truncation width is respected.
        result = Flatter.scale_for_reduction(A; reducer=smsv_reference_lll, bits=200)
        @test result.bits == 200

        # Aggressive precision truncates harder than the default.
        default = Flatter.scale_for_reduction(A; reducer=smsv_reference_lll)
        eager = Flatter.scale_for_reduction(A; reducer=smsv_reference_lll, aggressive=true)
        @test eager.bits <= default.bits

        # A steeply descending profile has no block structure to exploit, so
        # truncating below its spread annihilates the short columns. That is
        # reported rather than passed on to the reducer.
        steep = zeros(BigInt, 4, 4)
        steep[1, 1] = BigInt(1) << 300
        for i in 2:4
            steep[i, i] = 1
        end
        @test_throws ArgumentError Flatter.scale_for_reduction(
            steep; reducer=smsv_reference_lll, bits=64)

        # A uniform column exponent, by contrast, is absorbed entirely by the
        # truncation and changes nothing: the scaling is relative to the profile.
        offset = Flatter.scale_for_reduction(Flatter.ScaledBasis(A, fill(-10_000, 4));
                                          reducer=smsv_reference_lll)
        plain = Flatter.scale_for_reduction(A; reducer=smsv_reference_lll)
        @test offset.scaled == plain.scaled
        @test offset.transform == plain.transform

        # A reducer returning the wrong shape is caught.
        wrong_shape(M) = (M, Matrix{BigInt}(I, size(M, 2) + 1, size(M, 2) + 1))
        @test_throws DimensionMismatch Flatter.scale_for_reduction(A; reducer=wrong_shape)

        # A reducer that is not well-behaved breaks the block structure, and the
        # conjugation refuses rather than returning a non-unimodular matrix.
        blocked = smsv_blocked_basis(4, 16, 400; seed=5)
        function mixing_reducer(M)
            n = size(M, 2)
            V = Matrix{BigInt}(I, n, n)
            V[n, 1] = 1               # mixes a late column into an early one
            return M * V, V
        end
        @test_throws ArgumentError Flatter.scale_for_reduction(
            blocked; reducer=mixing_reducer)
    end

    @testset "reference LLL is well-behaved" begin
        # The tests above lean on this, so check it directly: the reference
        # reducer returns a unimodular transformation and a reduced basis.
        A = smsv_random_basis(5, 24; seed=6)
        reduced, U = smsv_reference_lll(A)
        @test reduced == Matrix{BigInt}(A) * U
        @test smsv_is_unimodular(U)
        @test smsv_is_size_reduced(reduced; eta=0.51, theta=0.0)
        @test smsv_satisfies_lovasz(reduced; delta=0.98)
    end

end
