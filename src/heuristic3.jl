# heuristic3.jl
#
# flatter's serial phase-3 representation update.
#
# A phase-3 iteration reduces one or two sublattice windows.  Their transforms
# are embedded into one block-diagonal U, after which the representation is
# updated tile by tile rather than by QR-factorising the entire basis:
#
#     U_i          <- block diagonal of the sublattice transforms
#     B_next[r, c] <- B[r, c] * U_i[c, c]
#     R[c, c]      <- QR(B_next[c, c])
#     R[r, c]      <- relative size reduction of tile c against tile r
#
# The off-diagonal reductions are performed from the bottom upwards so that each
# row tile sees the state produced by lower tiles.  Diagonal QR factors are
# retained as packed Householder reflectors plus tau and reused by the
# orthogonal relative-size-reduction kernel.
#
# This file implements the serial Heuristic3 path.  flatter's Threaded3 is a
# separate implementation and is not reproduced here.

"""
    Tile

A contiguous range of columns, and whether a sublattice reduction acted on it.

`reduce = false` marks a gap between windows. Those tiles are carried through the
update unchanged -- there is no transform to apply and no factorisation to
redo -- which is most of what makes the tiled update cheaper than the generic
one.
"""
struct Tile
    range::UnitRange{Int}
    reduce::Bool
end

Base.length(t::Tile) = length(t.range)
Base.first(t::Tile) = first(t.range)
Base.last(t::Tile) = last(t.range)

"""
    tile_partition(windows, n) -> (tiles, reducible)

Partition `1:n` into tiles from a sorted, disjoint list of sublattice windows.

Each window becomes a tile marked for reduction; every gap becomes a tile that is
not. `reducible` indexes the tiles that came from windows, in order, so a caller
holding one transform per window can match them up without searching.

A port of `Heuristic3::init_tiles`.
"""
function tile_partition(windows::AbstractVector{UnitRange{Int}}, n::Integer)
    n = Int(n)
    n >= 0 || throw(ArgumentError("n must be nonnegative"))

    tiles = Tile[]
    reducible = Int[]
    previous = 0

    for window in windows
        isempty(window) && continue
        first(window) > previous || throw(ArgumentError(
            "windows must be sorted and disjoint; $window follows column $previous"))
        last(window) <= n || throw(ArgumentError("window $window exceeds $n columns"))

        if first(window) > previous + 1
            push!(tiles, Tile((previous + 1):(first(window) - 1), false))
        end
        push!(reducible, length(tiles) + 1)
        push!(tiles, Tile(window, true))
        previous = last(window)
    end

    previous < n && push!(tiles, Tile((previous + 1):n, false))
    return tiles, reducible
end

"""
    embed_transforms!(U, tiles, reducible, transforms) -> U

Place each sublattice transform into its own diagonal block of `U`, leaving the
rest the identity.

The result is block diagonal, which is what lets the basis update below work one
tile pair at a time: column tile `c` is only ever multiplied by `U[c, c]`.
"""
function embed_transforms!(U::AbstractMatrix{T}, tiles::Vector{Tile},
                           reducible::Vector{Int},
                           transforms::Vector{<:AbstractMatrix{T}}) where {T<:Integer}
    length(reducible) == length(transforms) || throw(DimensionMismatch(
        "$(length(reducible)) reducible tiles but $(length(transforms)) transforms"))
    _set_identity!(U)

    for (slot, index) in enumerate(reducible)
        range = tiles[index].range
        block = transforms[slot]
        size(block) == (length(range), length(range)) || throw(DimensionMismatch(
            "transform $slot is $(size(block)) but its tile spans $(length(range))"))
        for (j, column) in enumerate(range), (i, row) in enumerate(range)
            U[row, column] = block[i, j]
        end
    end
    return U
end

"""
    tiled_basis_update!(B_next, B, U, tiles) -> B_next

`B_next = B * U`, for a `U` that is block diagonal over `tiles`.

Column tile `c` depends only on `U[c, c]`, so the product is a set of tile-sized
multiplications rather than one of order `n`. A tile that was not reduced has the
identity on its diagonal block, so its columns are copied rather than multiplied.

Only the blocks at or above the diagonal are touched: `B` is upper triangular by
tile, so everything below is structurally zero.

A port of `Heuristic3::update_b`.
"""
function tiled_basis_update!(B_next::AbstractMatrix{T}, B::AbstractMatrix{T},
                             U::AbstractMatrix{T},
                             tiles::Vector{Tile};
                             telemetry::Union{Nothing, ReductionTelemetry} = nothing
                             ) where {T<:Integer}
    for (c, column_tile) in enumerate(tiles)
        columns = column_tile.range
        for r in 1:c
            rows = tiles[r].range
            if !column_tile.reduce
                # Nothing acted on this tile, so its columns pass straight
                # through. This is where the saving is: a run with one window
                # per iteration leaves most tiles untouched.
                write_started = _tick()
                for j in columns, i in rows
                    B_next[i, j] = B[i, j]
                end
                telemetry === nothing ||
                    (telemetry.h3_time_basis_write += _tock(write_started))
                continue
            end
            materialise_started = _tick()
            left = Matrix{T}(view(B, rows, columns))
            right = Matrix{T}(view(U, columns, columns))
            telemetry === nothing ||
                (telemetry.h3_time_basis_materialise += _tock(materialise_started))
            mul_started = _tick()
            product = _reduction_mul(left, right)
            telemetry === nothing ||
                (telemetry.h3_time_basis_mul += _tock(mul_started))
            write_started = _tick()
            for (jj, j) in enumerate(columns), (ii, i) in enumerate(rows)
                B_next[i, j] = product[ii, jj]
            end
            telemetry === nothing ||
                (telemetry.h3_time_basis_write += _tock(write_started))
        end
    end
    return B_next
end

"""
    tiled_diagonal_qr!(R, tau, B_next, tiles, precision) -> (R, tau)

Factorise each reduced tile's diagonal block on its own, writing the result into
the corresponding block of `R`.

This is the step that removes the global factorisation: instead of one QR of
order `n`, there is one per reduced tile, of that tile's order. Tiles that were
not reduced keep whatever `R` already held for them.

A port of `Heuristic3::qr`.
"""
# Direct packed Householder QR used by Heuristic3 tiles.  flatter sends MPFR
# QRFactorization to its unblocked HouseholderMPFR implementation; it does not
# build a compact-WY factor here.  Reuse the allocation-controlled reflector
# primitives from fused QR so BigFloat arithmetic mutates MPFR values in place.
function _heuristic3_direct_qr!(factors::Matrix{S}) where {S<:AbstractFloat}
    m, n = size(factors)
    r = min(m, n)
    tau = Vector{S}(undef, r)
    ws = _fused_temporaries(S, _FUSED_WORKSPACE_SIZE)

    for k in 1:r
        tau_k = _fused_larfg!(factors, k, m, ws)
        tau[k] = tau_k
        for j in (k + 1):n
            _fused_larf_col!(factors, j, k, tau_k, m, ws)
        end
    end
    return factors, tau
end

function tiled_diagonal_qr!(R::AbstractMatrix{S}, tau::AbstractVector{S},
                            B_next::AbstractMatrix{T}, tiles::Vector{Tile},
                            precision::Integer) where {T<:Integer, S<:AbstractFloat}
    bits = Int(precision)
    for tile in tiles
        range = tile.range

        if !tile.reduce
            # A gap tile passes through untouched, so its block is still upper
            # triangular and is therefore its own R factor. flatter can skip
            # this because its `R` persists between iterations and already holds
            # the answer; ours is built fresh each time, and leaving the block
            # zero puts a -Inf in the profile.
            with_precision(bits) do
                for j in range, i in range
                    R[i, j] = i <= j ? S(B_next[i, j]) : zero(S)
                end
                for i in range
                    tau[i] = zero(S)
                end
            end
            continue
        end
        # A tile's factorisation needs the precision ITS conditioning demands,
        # not the driver's, which is chosen for the whole matrix. The monolithic
        # fused QR never meets this because it size reduces each column before
        # taking its reflector, so entries stay small; here the raw block is
        # factorised as it stands. On a knapsack basis -- mostly unit diagonal
        # entries beside one enormous one -- the driver's figure is far too
        # little and a diagonal cancels to zero on the very first iteration.
        window = view(B_next, range, range)
        widest = 0
        for value in window
            iszero(value) || (widest = max(widest, ndigits(value; base = 2)))
        end
        tile_bits = max(bits, householder_precision(length(range), Float64(widest)))

        factors, tile_tau = with_precision(tile_bits) do
            F = Matrix{S}(undef, length(range), length(range))
            for (j, column) in enumerate(range), (i, row) in enumerate(range)
                F[i, j] = S(B_next[row, column])
            end
            _heuristic3_direct_qr!(F)
        end

        # Converted down, so `R` stays uniform at the driver's precision: a
        # matrix holding entries of two widths is what `assert_precision`
        # refuses, and the profile would be read off a mixture.
        with_precision(bits) do
            for (j, column) in enumerate(range), (i, row) in enumerate(range)
                R[row, column] = S(factors[i, j])
            end
            for (i, row) in enumerate(range)
                tau[row] = S(tile_tau[i])
            end
        end
    end
    return R, tau
end

"""
    tiled_size_reduction!(B, B_next, U_i, U_sr, R, tau, tiles, precision)

Size reduce the tiled representation, one tile pair at a time.

For each column tile, the tiles above it are reduced against their own diagonal
block, working BOTTOM UP so that a reduction is never undone by one below it.
Each step produces a transform for that tile pair, which is then propagated to
the tiles above it in the same column and folded into the running `U_i`.

`U_sr` AND `U_i` ARE NOT THE SAME THING, and confusing them is easy. `U_sr`
holds each tile pair's transform on its own, as scratch handed from one step to
the next -- flatter's `U_sr`, passed into `update_b_next` and `update_u`. `U_i`
accumulates their PRODUCT, and is the transform satisfying

    B_next == B_on_entry * U_i

With one or two tiles the two coincide, because there is at most one pair and no
cross term; from three tiles on they diverge, since `U_i[t1, t3]` picks up
`U_i[t1, t2] * U_sr[t2, t3]` and `U_sr` never sees it. Callers want `U_i`.

The diagonal block a tile is reduced against comes in one of two forms, and
telling `relative_size_reduction!` which saves it a factorisation:

  * a REDUCED tile already has its QR in `R` and `tau` from
    [`tiled_diagonal_qr!`](@ref), consumed directly by the reflector kernel;
  * a tile that was not reduced is still upper triangular from the previous
    iteration, so `triangular = true` and no factorisation is needed at all.

A port of `Heuristic3::sr`, `update_b_next` and `update_u`.
"""
function tiled_size_reduction!(B::AbstractMatrix{T}, B_next::AbstractMatrix{T},
                               U_i::AbstractMatrix{T}, U_sr::AbstractMatrix{T},
                               R::AbstractMatrix{S}, tau::AbstractVector{S},
                               tiles::Vector{Tile};
                               precision::Integer,
                               strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
                               telemetry::Union{Nothing, ReductionTelemetry} = nothing
                               ) where {T<:Integer, S<:AbstractFloat}
    _set_identity!(U_sr)
    bits = Int(precision)

    # BigFloat arithmetic in the direct-reflector relative size reduction takes
    # its precision from the global default rather than from its operands.
    # `R` and `tau` already hold values at `bits`, so without this the two would
    # disagree and `assert_precision` would refuse them. See Notes.md: it is the
    # COMPUTATION that has to be wrapped, not just the allocations.
    return with_precision(bits) do
        _tiled_size_reduction_body!(B, B_next, U_i, U_sr, R, tau, tiles, bits,
                                    Int(strassen_cutoff), telemetry)
    end
end

function _tiled_size_reduction_body!(B::AbstractMatrix{T}, B_next::AbstractMatrix{T},
                                     U_i::AbstractMatrix{T}, U_sr::AbstractMatrix{T},
                                     R::AbstractMatrix{S}, tau::AbstractVector{S},
                                     tiles::Vector{Tile}, bits::Int,
                                     strassen_cutoff::Int,
                                     telemetry::Union{Nothing, ReductionTelemetry}
                                     ) where {T<:Integer, S<:AbstractFloat}
    for column_index in 1:length(tiles)
        columns = tiles[column_index].range

        # Bottom up: reducing against a lower diagonal block changes the entries
        # a higher one would have seen, so the order is not free.
        for row_index in (column_index - 1):-1:1
            row_tile = tiles[row_index]
            rows = row_tile.range

            telemetry === nothing || (telemetry.h3_sr_pairs += 1)
            materialise_started = _tick()
            B1 = Matrix{T}(view(B, rows, rows))
            B2 = Matrix{T}(view(B_next, rows, columns))
            block = Matrix{T}(undef, length(rows), length(columns))
            telemetry === nothing ||
                (telemetry.h3_time_sr_materialise += _tock(materialise_started))

            if row_tile.reduce
                # This is flatter's Heuristic3 rule: a reduced row tile has just
                # been factorised at the iteration's working precision, so pass
                # that R/tau factorisation straight to RelativeSizeReduction.
                # Do not compare it with relative_size_reduction_precision(B1):
                # that generic policy is based on absolute integer bit-length,
                # whereas Heuristic3 deliberately chooses precision from the
                # compressed post-child profile spread.
                telemetry === nothing || (telemetry.h3_sr_reuse_qr += 1)
                factorprep_started = _tick()
                # Keep these as views: flatter passes submatrices of its
                # persistent R/tau storage directly to RelativeSizeReduction.
                factors = view(R, rows, rows)
                scalars = view(tau, rows)
                R2 = Matrix{S}(undef, length(rows), length(columns))
                telemetry === nothing ||
                    (telemetry.h3_time_sr_factor_setup += _tock(factorprep_started))
                reduce_started = _tick()
                _relative_orthogonal_reflectors!(B1, B2, block, factors, scalars, R2;
                    deadband = RELATIVE_SIZE_REDUCTION_DEADBAND,
                    max_passes = RELATIVE_SIZE_REDUCTION_MAX_PASSES)
                elapsed = _tock(reduce_started)
                if telemetry !== nothing
                    telemetry.h3_time_sr_reduce += elapsed
                    telemetry.h3_time_sr_orthogonal += elapsed
                end
            else
                telemetry === nothing || (telemetry.h3_sr_triangular += 1)
                # Untouched since the previous iteration, so still triangular;
                # no floating-point factorisation is needed at all.
                reduce_started = _tick()
                _, _, R2 = relative_size_reduction!(B1, B2, block;
                                                    triangular = true,
                                                    precision = bits,
                                                    strassen_cutoff = strassen_cutoff)
                elapsed = _tock(reduce_started)
                if telemetry !== nothing
                    telemetry.h3_time_sr_reduce += elapsed
                    telemetry.h3_time_sr_triangular += elapsed
                end
            end

            writeback_started = _tick()
            for (jj, j) in enumerate(columns), (ii, i) in enumerate(rows)
                B_next[i, j] = B2[ii, jj]
                U_sr[i, j] = block[ii, jj]
            end

            # The coordinates of the reduced block in this tile's frame are the
            # off-diagonal block of R -- flatter's `params.R2`. Without them the
            # compression sees a block-diagonal R and produces a basis that has
            # lost everything above the diagonal.
            if R2 !== nothing
                # Converted, not assigned: keep every block of `R` at the
                # driver's uniform working precision.  This runs inside a
                # `with_precision(bits)` block, so the conversion lands there.
                for (jj, j) in enumerate(columns), (ii, i) in enumerate(rows)
                    R[i, j] = S(R2[ii, jj])
                end
            end
            telemetry === nothing ||
                (telemetry.h3_time_sr_writeback += _tock(writeback_started))

            # Propagate: the same transform applies to every tile above this one
            # in the same column, for the basis and for the running transform.
            propagate_started = _tick()
            _tiled_accumulate!(B_next, B, U_sr, tiles, row_index, column_index,
                               1:(row_index - 1))
            _tiled_accumulate!(U_i, U_i, U_sr, tiles, row_index, column_index,
                               1:row_index)

            for j in columns, i in 1:last(rows)
                B[i, j] = B_next[i, j]
            end
            telemetry === nothing ||
                (telemetry.h3_time_sr_propagate += _tock(propagate_started))
        end
    end
    return B_next, U_sr
end

"""
`target[k, col] += source[k, row] * U_sr[row, col]`, over the tiles in `above`.

Split out because the basis and the running transform need the same propagation
against different sources: the basis reads the PREVIOUS iterate while the
transform accumulates into itself.
"""
function _tiled_accumulate!(target::AbstractMatrix{T}, source::AbstractMatrix{T},
                            U_sr::AbstractMatrix{T}, tiles::Vector{Tile},
                            row_index::Int, column_index::Int,
                            above) where {T<:Integer}
    rows = tiles[row_index].range
    columns = tiles[column_index].range
    factor = Matrix{T}(view(U_sr, rows, columns))
    all(iszero, factor) && return nothing

    for k in above
        band = tiles[k].range
        product = _reduction_mul(Matrix{T}(view(source, band, rows)), factor)
        for (jj, j) in enumerate(columns), (ii, i) in enumerate(band)
            target[i, j] += product[ii, jj]
        end
    end
    return nothing
end

"""
    heuristic3_update!(working, window, sub_transform, n, precision; kwargs...)

One iteration's representation update, done tile by tile.

Returns `(basis, transform, R, tau)`: the updated basis, the transform taking the
old basis to it, and the R factor assembled from the tile factorisations.

This is `Heuristic3::update_representation`, and it replaces the generic route of
folding in the sublattice transform and then re-factorising the whole matrix. The
columns are cut into tiles at the window boundaries, each reduced tile is
factorised on its own, and the off-diagonal blocks of `R` come from relative size
reduction between tiles.

The R factor is therefore assembled locally rather than being a single global
factorisation. flatter accepts that: the profile it yields steers the reduction
rather than certifying it, and the exact invariants live in the integer matrices
either way -- `basis == working * transform` holds regardless of how the tiles
are cut.
"""
function heuristic3_update!(working::AbstractMatrix{T},
                            windows::AbstractVector{UnitRange{Int}},
                            sub_transforms::AbstractVector{<:AbstractMatrix{T}},
                            n::Int, precision::Integer;
                            float_type::Type{S} = BigFloat,
                            strassen_cutoff::Integer = DEFAULT_STRASSEN_CUTOFF,
                            telemetry::Union{Nothing, ReductionTelemetry} = nothing
                            ) where {T<:Integer, S<:AbstractFloat}
    # PRECONDITION. `tiled_basis_update!` writes only blocks at or above the tile
    # diagonal, and gap tiles are treated as triangular during relative size
    # reduction.  The working representation must therefore be fully upper
    # triangular.  The phase-3 driver guarantees this by compressing the R factor
    # before each recursive iteration.
    telemetry === nothing || (telemetry.h3_calls += 1)
    precheck_started = _tick()
    for j in 1:n, i in (j + 1):n
        iszero(working[i, j]) || throw(ArgumentError(
            "the working basis must be upper triangular; working[$i,$j] is " *
            "nonzero. The tiled update reads only the blocks at or above the " *
            "tile diagonal, so anything below would be lost."))
    end
    telemetry === nothing ||
        (telemetry.h3_time_precheck += _tock(precheck_started))

    setup_started = _tick()
    tiles, reducible = tile_partition(windows, n)

    transform = Matrix{T}(undef, n, n)
    embed_transforms!(transform, tiles, reducible,
                      [Matrix{T}(u) for u in sub_transforms])
    telemetry === nothing || (telemetry.h3_time_setup += _tock(setup_started))

    basis_started = _tick()
    basis = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        basis[i, j] = zero(T)
    end
    tiled_basis_update!(basis, working, transform, tiles; telemetry = telemetry)
    telemetry === nothing || (telemetry.h3_time_basis += _tock(basis_started))

    qr_started = _tick()
    # Allocated INSIDE the precision block: a `zeros(BigFloat, ...)` outside it
    # would hold entries at the default precision, and the tiles written by the
    # factorisation would then disagree with the ones left untouched.
    R, tau = with_precision(Int(precision)) do
        (zeros(S, n, n), zeros(S, n))
    end
    tiled_diagonal_qr!(R, tau, basis, tiles, precision)
    telemetry === nothing || (telemetry.h3_time_qr += _tock(qr_started))

    sr_setup_started = _tick()
    # `tiled_size_reduction!` reads the previous iterate from its first argument
    # and writes the current one to its second, so they start equal.
    previous = Matrix{T}(basis)
    scratch = Matrix{T}(undef, n, n)
    telemetry === nothing ||
        (telemetry.h3_time_sr_setup += _tock(sr_setup_started))
    sr_started = _tick()
    tiled_size_reduction!(previous, basis, transform, scratch, R, tau, tiles;
                          precision = precision,
                          strassen_cutoff = strassen_cutoff,
                          telemetry = telemetry)
    telemetry === nothing || (telemetry.h3_time_sr += _tock(sr_started))

    return basis, transform, R, tau
end

"""
Single-window form, for the schedule that only ever reduces one window at a
time. flatter's phase 3 yields two on odd iterations, which is the case the
tiling was designed around.
"""
heuristic3_update!(working::AbstractMatrix{T}, window::UnitRange{Int},
                   sub_transform::AbstractMatrix{T}, n::Int, precision::Integer;
                   kwargs...) where {T<:Integer} =
    heuristic3_update!(working, [window], [sub_transform], n, precision; kwargs...)
