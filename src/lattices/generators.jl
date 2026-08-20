# lattices/generators.jl
#
# Families of lattice bases, for benchmarking and testing reduction.
#
# CONVENTION. Every generator returns a `Matrix{BigInt}` whose COLUMNS are the
# basis vectors, matching Julia's column-major layout and the rest of this
# package, and every one is UPPER TRIANGULAR, which is what
# [`lattice_reduce!`](@ref) requires.
#
# The textbook forms of most of these families are triangular the other way
# round once transposed into column convention, so they are flipped here by
# `_flip_to_upper`: reversing both the row and column order of a lower
# triangular matrix gives an upper triangular one. Reversing the columns is a
# reordering of the basis vectors and reversing the rows is a permutation of
# the ambient coordinates, so neither changes the lattice's geometry — vector
# norms, the determinant, and the difficulty of reduction are all preserved.
#
# Each generator returns a named tuple carrying the basis and whatever is known
# about it in advance, so a benchmark can say not just "how short a vector did
# we find" but "did we find the one that is known to be there".
#
# THESE ARE NOT STANDARD INSTANCES. They reproduce the SHAPE of the families
# they are named after, which is enough to track this package against itself and
# against fplll on the same input, but the parameters are chosen to stress
# particular machinery rather than to match any published convention. Numbers
# from these are not comparable with the literature. For that, generate inputs
# with fplll's `latticegen` or take them from the Darmstadt lattice challenges.

"""
    _flip_to_upper(L) -> Matrix{BigInt}

Reverse both the row and column order of `L`, turning a lower triangular matrix
into an upper triangular one.

`M[i,j] = L[n+1-i, n+1-j]`, so `M[i,j]` is zero exactly when `L[n+1-i, n+1-j]`
is, i.e. when `n+1-i < n+1-j`, i.e. when `j < i`.
"""
function _flip_to_upper(L::AbstractMatrix{<:Integer})
    n = size(L, 1)
    size(L, 2) == n || throw(DimensionMismatch("expected a square matrix"))
    return BigInt[L[n + 1 - i, n + 1 - j] for i in 1:n, j in 1:n]
end

"Assert the result really is upper triangular with a nonzero diagonal."
function _check_generated(B::AbstractMatrix{BigInt}, name::AbstractString)
    n = size(B, 2)
    for j in 1:n
        iszero(B[j, j]) && error("$name produced a zero diagonal entry at $j")
        for i in (j + 1):n
            iszero(B[i, j]) || error("$name produced a non-triangular basis at ($i,$j)")
        end
    end
    return B
end

function _random_bigint(rng, bits::Integer)
    bits <= 0 && return big(0)
    x = big(0)
    for _ in 1:cld(bits, 32)
        x = (x << 32) + rand(rng, 0:(2^32 - 1))
    end
    return x >> max(0, 32 * cld(bits, 32) - Int(bits))
end

_random_signed_bigint(rng, bits::Integer) =
    rand(rng, Bool) ? _random_bigint(rng, bits) : -_random_bigint(rng, bits)

# ---------------------------------------------------------------------------

"""
    knapsack_lattice(rng, n; bits=n, weight=nothing) -> (; basis, name, dimension, log2_determinant, planted_norm2, density)

A Lagarias–Odlyzko subset-sum lattice with a planted solution.

Given weights `a_1..a_n` of `bits` bits and a subset sum `S`, the basis
(row convention) is `I` beside the column `N*a`, with a final row `N*S`. The
combination of rows picked out by the subset, minus the last row, is
`(e, 0)` — a vector of norm `sqrt(|e|)` while everything else in the lattice is
around `N`. Recovering it solves the subset-sum instance, so this is a family
where reduction has a concrete, checkable target.

`N` is chosen larger than `sqrt(n)` so the planted vector really is the
shortest. The reported `density` is `n / bits`; instances near density 1 are the
hard ones, and low density is where lattice attacks succeed.
"""
function knapsack_lattice(rng, n::Integer; bits::Integer = n,
                          weight::Union{Nothing, Integer} = nothing)
    n >= 1 || throw(ArgumentError("dimension must be positive"))
    weights = [_random_bigint(rng, bits) + 1 for _ in 1:n]

    chosen = falses(n)
    target = weight === nothing ? max(1, div(n, 2)) : Int(weight)
    for i in randperm(rng, n)[1:min(target, n)]
        chosen[i] = true
    end
    total = sum(weights[i] for i in 1:n if chosen[i]; init = big(0))

    scale = big(1) << (ndigits(big(n); base = 2) + 2)   # comfortably > sqrt(n)

    # Row convention: upper triangular, (n+1) x (n+1).
    rows = zeros(BigInt, n + 1, n + 1)
    for i in 1:n
        rows[i, i] = big(1)
        rows[i, n + 1] = scale * weights[i]
    end
    rows[n + 1, n + 1] = scale * total

    # Column convention is the transpose, which is lower triangular; flip it.
    basis = _flip_to_upper(permutedims(rows))
    _check_generated(basis, "knapsack_lattice")

    log2_det = sum(_log2_abs(basis[i, i]) for i in 1:(n + 1))

    return (basis = basis, name = "knapsack", dimension = n + 1,
            log2_determinant = log2_det,
            planted_norm2 = big(count(chosen)),
            density = n / bits)
end

"""
    qary_lattice(rng, n; q=nothing) -> (; basis, name, dimension, log2_determinant, planted_norm2, modulus)

A random q-ary (Ajtai / NTRU-shaped) lattice of dimension `2n`.

In column convention the basis is upper triangular by construction:

    [ q*I   H ]
    [  0    I ]

with `H` uniform mod `q`. This is the standard hard instance — no planted short
vector, and the shortest vector is around the Gaussian heuristic — so it
measures reduction quality rather than recovery of a known target.

`planted_norm2` is `nothing`; the Gaussian heuristic prediction is reported by
the benchmark instead.

!!! note "The default modulus grows with dimension"
    `q` defaults to `2^(2n) + 1`, so it carries about `2n + 1` bits and the
    determinant is `2^(2n^2)` — quadratic in the lattice dimension. That is
    deliberate, to make entry sizes grow with dimension and exercise the
    compression machinery, but it is NOT a standard parameterisation and it is
    what makes memory the binding constraint at large dimension. Published
    q-ary benchmarks generally fix `q` or scale it polynomially in `n`; pass `q`
    explicitly to match one.
"""
function qary_lattice(rng, n::Integer; q::Union{Nothing, Integer} = nothing)
    n >= 1 || throw(ArgumentError("dimension must be positive"))
    modulus = q === nothing ? big(2)^(2 * max(4, n)) + 1 : big(q)
    modulus > 1 || throw(ArgumentError("the modulus must exceed one"))

    size_ = 2n
    basis = zeros(BigInt, size_, size_)
    for i in 1:n
        basis[i, i] = modulus
        basis[n + i, n + i] = big(1)
        for r in 1:n
            basis[r, n + i] = rand(rng, big(0):(modulus - 1))
        end
    end
    _check_generated(basis, "qary_lattice")

    return (basis = basis, name = "q-ary", dimension = size_,
            log2_determinant = n * _log2_abs(modulus),
            planted_norm2 = nothing,
            modulus = modulus)
end

"""
    spread_lattice(rng, n; spread=8n, offdiag=20) -> (; basis, name, dimension, log2_determinant, planted_norm2, spread)

An upper triangular basis whose diagonal magnitudes climb across `spread` bits.

This is the family flatter's compression exists for: the Gram–Schmidt profile
covers a wide dynamic range, so the working precision needed for a naive
reduction is large while the *compressed* problem is small. Reduction quality
matters less here than whether the compression keeps the precision down.
"""
function spread_lattice(rng, n::Integer; spread::Integer = 8 * n,
                        offdiag::Integer = 20)
    n >= 1 || throw(ArgumentError("dimension must be positive"))
    basis = zeros(BigInt, n, n)
    for j in 1:n
        shift = n == 1 ? 0 : div(Int(spread) * (j - 1), n - 1)
        head = _random_bigint(rng, 8) + 1
        basis[j, j] = head << shift
        for i in 1:(j - 1)
            basis[i, j] = _random_signed_bigint(rng, offdiag)
        end
    end
    _check_generated(basis, "spread_lattice")

    return (basis = basis, name = "spread", dimension = n,
            log2_determinant = sum(_log2_abs(basis[i, i]) for i in 1:n),
            planted_norm2 = nothing,
            spread = Int(spread))
end

"""
    random_triangular_lattice(rng, n; bits=64) -> (; basis, name, dimension, log2_determinant, planted_norm2)

A plain random upper triangular basis with entries of roughly `bits` bits.

The baseline family: no structure, no planted vector, and a profile that is
already fairly flat, so it isolates the cost of the machinery from the
difficulty of the instance.
"""
function random_triangular_lattice(rng, n::Integer; bits::Integer = 64)
    n >= 1 || throw(ArgumentError("dimension must be positive"))
    basis = zeros(BigInt, n, n)
    for j in 1:n
        d = big(0)
        while iszero(d)
            d = _random_signed_bigint(rng, bits)
        end
        basis[j, j] = d
        for i in 1:(j - 1)
            basis[i, j] = _random_signed_bigint(rng, bits)
        end
    end
    _check_generated(basis, "random_triangular_lattice")

    return (basis = basis, name = "random-triangular", dimension = n,
            log2_determinant = sum(_log2_abs(basis[i, i]) for i in 1:n),
            planted_norm2 = nothing)
end

"""
    scrambled_lattice(rng, n; bits=32, operations=6n) -> (; basis, name, dimension, log2_determinant, planted_norm2)

A well-conditioned lattice deliberately spoiled by random integer column
operations, then re-triangularised by rounding its exact R factor.

The lattice is easy — it starts near-orthogonal — but the basis handed to the
reducer is badly skewed, so a correct reduction should undo the damage and
recover something close to the original quality. That makes it a check on
whether reduction is actually working, as opposed to merely terminating.
"""
function scrambled_lattice(rng, n::Integer; bits::Integer = 32,
                           operations::Integer = 6 * n)
    n >= 1 || throw(ArgumentError("dimension must be positive"))

    basis = zeros(BigInt, n, n)
    for j in 1:n
        basis[j, j] = big(1) << rand(rng, 0:Int(bits))
    end

    # Column operations that keep the matrix upper triangular: adding a
    # multiple of column k to column j > k cannot create an entry below the
    # diagonal, since column k is zero below row k.
    for _ in 1:Int(operations)
        j = rand(rng, 2:n)
        k = rand(rng, 1:(j - 1))
        c = rand(rng, -8:8)
        iszero(c) && continue
        for i in 1:k
            basis[i, j] += c * basis[i, k]
        end
    end
    _check_generated(basis, "scrambled_lattice")

    return (basis = basis, name = "scrambled", dimension = n,
            log2_determinant = sum(_log2_abs(basis[i, i]) for i in 1:n),
            planted_norm2 = nothing)
end

"""
    lattice_families() -> Vector

The generators the benchmark sweeps over, each as
`(; name, generate)` where `generate(rng, n)` returns a basis bundle.

`n` is the size parameter, not necessarily the lattice dimension: the knapsack
family produces `n+1` columns and the q-ary family `2n`.
"""
lattice_families() = [
    (name = "random-triangular", generate = (rng, n) -> random_triangular_lattice(rng, n)),
    (name = "spread",            generate = (rng, n) -> spread_lattice(rng, n)),
    (name = "scrambled",         generate = (rng, n) -> scrambled_lattice(rng, n)),
    (name = "knapsack",          generate = (rng, n) -> knapsack_lattice(rng, n)),
    (name = "q-ary",             generate = (rng, n) -> qary_lattice(rng, n)),
]
