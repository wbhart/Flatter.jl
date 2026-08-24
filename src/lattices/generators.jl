# lattices/generators.jl
#
# Families of lattice bases, for benchmarking and testing reduction.
#
# CONVENTION. Every generator returns a `Matrix{BigInt}` whose COLUMNS are the
# basis vectors, matching Julia's column-major layout and the rest of this
# package, and every one is UPPER TRIANGULAR, which is what
# [`lattice_reduce!`](@ref) requires.
#
# Shapes are left as the textbook gives them. `reduce_basis` accepts a basis
# triangular in any of the four corner orientations, and dense bases too, so a
# generator does not have to contort its output into upper triangular form.
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
Assert the generated basis is usable: square or tall, and either dense or
triangular in one of the four corner orientations with a nonzero diagonal.

`reduce_basis` accepts all of those, so a generator only has to produce a valid
basis rather than a particular shape.
"""
function _check_generated(B::AbstractMatrix{BigInt}, name::AbstractString;
                          dense::Bool = false)
    m, n = size(B)
    m >= n || error("$name produced a $m x $n basis; needs at least as many rows")
    all(iszero, B) && error("$name produced a zero basis")
    dense && return B
    triangular_orientation(B) === nothing &&
        error("$name was expected to be triangular in some orientation")
    return B
end

"""
    _log2_determinant(B) -> Float64

`log2 |det B|`, exactly, for a square basis.

Read off the diagonal when the basis is triangular in some orientation, and
computed by fraction-free elimination otherwise. Dense determinants cost
`O(n^3)` big-integer operations, which is fine at benchmark sizes and would not
be at cryptographic ones.
"""
function _log2_determinant(B::AbstractMatrix{BigInt})
    m, n = size(B)
    m == n || error("determinant of a $m x $n basis")

    if triangular_orientation(B) !== nothing
        oriented = copy(B)
        flips = triangular_orientation(B)
        _flip!(oriented, flips[1], flips[2])
        return sum(_log2_abs(oriented[i, i]) for i in 1:n)
    end

    M = Matrix{BigInt}(B)
    previous = big(1)
    for k in 1:(n - 1)
        if iszero(M[k, k])
            row = findfirst(r -> !iszero(M[r, k]), (k + 1):n)
            row === nothing && return -Inf
            pivot = k + row
            for c in 1:n
                M[k, c], M[pivot, c] = M[pivot, c], M[k, c]
            end
        end
        for i in (k + 1):n, j in (k + 1):n
            M[i, j] = div(M[i, j] * M[k, k] - M[i, k] * M[k, j], previous)
        end
        previous = M[k, k]
    end
    return _log2_abs(M[n, n])
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

    # Column convention is the transpose, which is LOWER triangular. That is
    # left as it is: `reduce_basis` recognises all four corner orientations, so
    # pre-flipping here would only hide a path worth exercising.
    basis = permutedims(rows)
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
                           operations::Integer = 6 * n, dense::Bool = true)
    n >= 1 || throw(ArgumentError("dimension must be positive"))

    basis = zeros(BigInt, n, n)
    for j in 1:n
        basis[j, j] = big(1) << rand(rng, 0:Int(bits))
    end
    log2_det = sum(_log2_abs(basis[i, i]) for i in 1:n)

    for _ in 1:Int(operations)
        j = rand(rng, 1:n)
        k = rand(rng, 1:n)
        j == k && continue
        c = rand(rng, -8:8)
        iszero(c) && continue
        # Unconstrained column operations, so the result is a genuinely dense
        # basis of the same lattice rather than a triangular one. Restricting
        # to k < j would keep it triangular, which was necessary before
        # `reduce_basis` could accept anything else.
        limit = dense ? n : k
        for i in 1:limit
            basis[i, j] += c * basis[i, k]
        end
    end
    _check_generated(basis, "scrambled_lattice"; dense = dense)

    return (basis = basis, name = "scrambled", dimension = n,
            log2_determinant = log2_det,
            planted_norm2 = nothing)
end


"""
    relation_lattice(rng, degree; bits=nothing, radicand=2) -> (; basis, name, dimension, log2_determinant, planted_norm2, minimal_polynomial)

The integer relation lattice that recovers the minimal polynomial of
`radicand^(1/degree)`.

Given a real `alpha` known to `bits` binary places, the columns

    b_i = e_i  +  round(2^bits * alpha^i) * e_last,     i = 0 .. degree

span a lattice whose short vectors are integer coefficient sequences with
`sum a_i alpha^i` close to zero — that is, near-relations among the powers of
`alpha`. The genuine relation is the minimal polynomial, so this family has a
*known answer* rather than merely a target to be short: `x^degree - radicand`,
of squared norm `radicand^2 + 1`.

This is the LLL application that made the algorithm famous outside lattice
cryptography — integer relation detection, and with it algebraic number
recognition and the experimental identification of closed forms. `bits` must
grow with the degree for the relation to be the shortest vector; the default
scales accordingly.

The basis is `(degree + 2) x (degree + 1)`, so it also exercises the
rectangular case: more ambient coordinates than basis vectors.
"""
function relation_lattice(rng, degree::Integer; bits::Union{Nothing, Integer} = nothing,
                          radicand::Integer = 2)
    degree >= 1 || throw(ArgumentError("degree must be positive"))
    radicand >= 2 || throw(ArgumentError("radicand must be at least two"))

    n = Int(degree) + 1
    # The relation has to be shorter than a generic lattice vector, which needs
    # the weight to dominate the coefficient sizes.
    scale = bits === nothing ? 16 * Int(degree) + 64 : Int(bits)

    powers = with_precision(scale + 64) do
        alpha = BigFloat(radicand)^(1 // Int(degree))
        [round(BigInt, ldexp(alpha^(i - 1), scale)) for i in 1:n]
    end

    basis = zeros(BigInt, n + 1, n)
    for j in 1:n
        basis[j, j] = big(1)
        basis[n + 1, j] = powers[j]
    end
    _check_generated(basis, "relation_lattice"; dense = true)

    # x^degree - radicand: coefficients (-radicand, 0, ..., 0, 1).
    planted = big(radicand)^2 + 1

    return (basis = basis, name = "relation", dimension = n,
            log2_determinant = NaN,          # rectangular; no determinant
            planted_norm2 = planted,
            minimal_polynomial = "x^$degree - $radicand")
end

"""
    ideal_lattice(rng, n; bits=8, polynomial=:x_minus_one) -> (; basis, name, dimension, log2_determinant, planted_norm2)

The ideal `(g)` in `Z[x]/(f)`, as a lattice of coefficient vectors.

Column `j` holds the coefficients of `x^(j-1) * g mod f`, so the lattice is the
ideal generated by `g` and its short vectors are the small elements of that
ideal. This is ideal-SVP: the structured-lattice problem underlying much of
lattice cryptography, and the same computation that arises when looking for
small generators of ideals in class group and unit group algorithms.

`polynomial` selects `f`: `:x_minus_one` gives `x^n - x - 1`, irreducible for
most `n`, and `:cyclotomic` gives `x^n + 1`, which needs `n` a power of two and
is the usual choice in cryptography.

The basis is dense and non-triangular, so it exercises that path. Its
determinant is the norm of `g`, computed here by elimination rather than as a
resultant.
"""
function ideal_lattice(rng, n::Integer; bits::Integer = 8,
                       polynomial::Symbol = :x_minus_one)
    n = Int(n)
    n >= 2 || throw(ArgumentError("dimension must be at least two"))

    # f as a reduction rule: x^n = reduction, given as coefficients of degree < n.
    reduction = zeros(BigInt, n)
    if polynomial === :x_minus_one
        reduction[2] = big(1)                     # x^n = x + 1
        reduction[1] = big(1)
    elseif polynomial === :cyclotomic
        ispow2(n) || throw(ArgumentError("x^n + 1 needs n a power of two, got $n"))
        reduction[1] = big(-1)                    # x^n = -1
    else
        throw(ArgumentError("unknown polynomial $polynomial"))
    end

    generator = BigInt[_random_signed_bigint(rng, bits) for _ in 1:n]
    all(iszero, generator) && (generator[1] = big(1))

    basis = zeros(BigInt, n, n)
    current = copy(generator)
    for j in 1:n
        basis[:, j] = current
        # Multiply by x: shift up, then fold the overflowing term back in.
        overflow = current[n]
        for i in n:-1:2
            current[i] = current[i - 1]
        end
        current[1] = big(0)
        if !iszero(overflow)
            for i in 1:n
                current[i] += overflow * reduction[i]
            end
        end
    end
    _check_generated(basis, "ideal_lattice"; dense = true)

    return (basis = basis, name = "ideal", dimension = n,
            log2_determinant = _log2_determinant(basis),
            planted_norm2 = nothing,
            generator_norm2 = sum(x -> x^2, generator))
end

"""
    lattice_families() -> Vector

The generators the benchmark sweeps over, each as
`(; name, generate)` where `generate(rng, n)` returns a basis bundle.

Each entry carries the size range that suits it. `n` is a size parameter, not
necessarily the lattice dimension: knapsack produces `n+1` columns, q-ary `2n`,
and relation `n+1` columns in `n+2` rows.

The ranges differ because the families do not cost the same at a given
dimension. q-ary is by far the most expensive; relation and ideal are cheap
enough to run considerably further, which is worth doing since both carry more
information than a bare shortest-vector figure -- relation has a known answer,
and ideal has structure that shows up as vectors below the Gaussian heuristic.
"""
lattice_families() = [
    (name = "random-triangular", sizes = [8, 16, 32, 48],
     generate = (rng, n) -> random_triangular_lattice(rng, n)),
    (name = "spread", sizes = [8, 16, 32, 48],
     generate = (rng, n) -> spread_lattice(rng, n)),
    (name = "scrambled", sizes = [8, 16, 32, 48],
     generate = (rng, n) -> scrambled_lattice(rng, n)),
    (name = "knapsack", sizes = [8, 16, 32, 48],
     generate = (rng, n) -> knapsack_lattice(rng, n)),
    # Doubles: these are dimensions 16 to 96, and the most expensive family.
    (name = "q-ary", sizes = [8, 16, 32, 48],
     generate = (rng, n) -> qary_lattice(rng, n)),
    # Cheap per dimension, and the planted answer stays checkable, so this runs
    # further than the others.
    (name = "relation", sizes = [8, 16, 32, 48, 64, 96],
     generate = (rng, n) -> relation_lattice(rng, n)),
    (name = "ideal", sizes = [8, 16, 32, 48, 64, 96, 128],
     generate = (rng, n) -> ideal_lattice(rng, n)),
]
