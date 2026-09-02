# Implementation notes

This file records properties of the current implementation that are useful when
reading, testing, or optimising it.  Algorithm history belongs in version
control, not here.

## Serial heuristic dispatcher

`reduce_basis(B)` defaults to `algorithm = :heuristic`.  Its dispatch mirrors
flatter's serial heuristic pipeline:

* one or two columns are handled before phase dispatch; a square `2 x 2` basis
  below 1400 bits uses Lagrange and other two-column bases use Schoenhage;
* phase 0 is `Irregular`, which recognises the four triangular orientations and
  otherwise sends dense input to phase 1;
* phase 1 uses `CondUnknown` when the condition number is unknown and
  `Heuristic1` when a positive `log_cond` is supplied;
* phase 2 is `Heuristic2`;
* phase 3 is `Heuristic3`;
* in phases 2 and 3, fpLLL is used only when the dimension is at most 32 and the
  represented integer precision is at most 128 bits.  `base_cutoff` remains an
  experimental/test override of the dimension 32 cutoff; its default is 32.

`Threaded3` is deliberately absent.  The implementation described here is the
serial heuristic algorithm.

`algorithm = :teaching` is independent.  It retains the QR-and-round dense
route and the generic recursive driver because those are useful for exposition,
cross-checking, and controlled experiments.  Teaching-only controls such as
`schedule`, `tiled`, `low_memory`, and `schoenhage_threshold` should not be
interpreted as flatter heuristic-policy knobs.

## Heuristic phases

### Irregular and CondUnknown

`Irregular` tests the four corner orientations of a square triangular basis.
The oriented triangular basis is exactly size reduced before entering phase 2.
A genuinely dense heuristic input enters `CondUnknown`, which starts at low
MPFR precision, identifies a resolvable independent prefix, reduces it through
phase 2, applies the transform to the exact basis, and raises precision until
all remaining dependencies are represented by exact zero columns.  Rank-
deficient dense input is therefore supported.

### Heuristic1

Heuristic1 is the known-condition phase-1 route.  Its simulated representation
is rectangular and carries the auxiliary right-hand block `B2` together with
`U2`.  The exact auxiliary invariant is

    B2_out == B2_in + B_in * U2.

The phase-1 split performs left, right, and whole-window updates; the whole
window is handed to phase 3.  A newly constructed phase-1 profile contains NaNs,
matching flatter's `Profile(n)` state; the first representation update produces
the usable profile.

### Heuristic2

Heuristic2 carries a compressed upper-triangular phase-2 representation and the
same `B2/U2` auxiliary relation where required.  Its split schedule is left,
right, then whole.  Partial children remain in phase 2 and the whole-window
child is handed to phase 3.

### Heuristic3

Heuristic3 updates the representation tile by tile.  Sublattice transforms are
embedded block-diagonally, the corresponding integer basis blocks are updated,
diagonal tiles are QR-factorised, and off-diagonal tiles are relatively size
reduced from the bottom upwards.  The phase-3 split tree shares child nodes in
the same way as flatter; the shared iteration state is intentional.

The serial MPFR path uses direct Householder reflectors and reuses the saved
`R/tau` factorisation during orthogonal relative size reduction.  Compact-WY
blocked QR remains implemented in `householder.jl` as an experimental building
block but is not the default Heuristic3 factorisation path.

## Fused QR and size reduction

`fused_qr_size_reduction.jl` implements flatter's MPFR `Columnwise` algorithm:
size reduction of a column is interleaved with generation of its Householder
reflector so that the working R factor never represents the unreduced basis for
longer than necessary.

`ColumnwiseDouble` is intentionally not implemented.  Flatter selects it only
when the supplied R matrix has hardware-double element type.  The serial
heuristic recursion allocates R as MPFR (starting at 53 bits) and changes its
precision rather than its element type, so the heuristic pipeline selects
`Columnwise`, not `ColumnwiseDouble`.

`LazyRefine`, `Iterated`, and `SeysenRefine` are also not implemented because the
serial heuristic dispatcher does not select them.  They are not prerequisites
for fidelity of the heuristic pipeline.

The fused implementation returns the Householder `tau` vector rather than
building compact-WY `T` eagerly.  `compact_wy_from_reflectors` is available for
callers that need `T`; Heuristic3 consumes the saved reflectors directly.

## Deliberate numerical differences from C++ flatter

These differences are intentional; differential tests should compare exact
lattice invariants and reduction properties rather than require bit-for-bit
identical intermediate matrices.

* Compression shifts use Julia's arithmetic right shift for negative integers.
  This is floor division by a power of two.  Some flatter relative-size-
  reduction code uses GMP truncating division instead; the discarded low bits
  can therefore differ by one for negative values.
* Julia has no direct high-level `mpfr_mul_z`.  Multiplying a `BigFloat` by a
  `BigInt` can involve promotion of the integer before multiplication.  The
  refinement loops are designed to tolerate the resulting extra rounding.
* The Float64 orthogonal relative-size-reduction path performs an actual
  refinement loop.  The corresponding upstream double implementation has a
  non-updated bound that makes its loop execute once.
* The fused stagnation guard compares exponent sizes, consistent with the
  relative-size-reduction code, instead of applying an absolute precision slack
  to multiplier magnitudes.
* The fused three-tier size-reduction test performs its boundary comparison in
  the working floating type rather than converting MPFR operands to `double`.
* The recursive teaching driver has a stagnation exit and an iteration cap.
  These are safety controls for that independent reducer and do not change the
  exact lattice invariant.

## Julia arbitrary-precision discipline

`Matrix{BigFloat}` has no matrix-wide precision encoded in its type.  Arithmetic
uses Julia's current default BigFloat precision, so kernels that allocate or
compute with BigFloat values must establish the intended precision explicitly.

Use `with_precision(f, bits)` for package kernels.  It changes the default
precision directly and restores it in `finally`; unlike the scoped
`setprecision(BigFloat, bits) do ... end` form, it avoids the `ScopedValue`
lookup in allocation-heavy inner loops.  It is not task-safe, which is one of
the issues that must be revisited before adding threading.

`uniform_precision` and `assert_precision` should be used when a matrix is
expected to have one precision.  Reading `precision(A[1, 1])` alone is not a
sufficient check because BigFloat entries can have different precisions.

`Float64(::BigInt)` can overflow on lattice-sized values.  Use
`float64_matrix`, which extracts a common power-of-two scale without changing
column ratios.

`BigInt` and `BigFloat` entries are mutable objects.  In-place GMP/MPFR helpers
may mutate shared objects, so they are used only where the callee owns the
entries.  Ordinary matrix copies of arbitrary-precision values must not be
assumed to make deep copies of the entry objects.

## Exact arithmetic and multiplication

The exact size-reduction implementation batches predecessor updates.  The
current default block size improves cache/allocation behaviour even when the
tiles are too small to reach Strassen.  The result is exact and independent of
the chosen block size.

Global transform products, Heuristic3 tile products, transform folding, and the
final exact application can consume a substantial part of a reduction.
Multi-modular integer matrix multiplication is therefore a separate optimisation
target from the heuristic representation logic.

Blocked QR is intentionally retained even though it is not selected by the
serial MPFR heuristic path.  It provides a tested Strassen-backed QR building
block for experiments and may be useful to other reduction pipelines.

## Memory and garbage collection

Large reductions hold many arbitrary-precision matrix entries whose limbs are
allocated outside Julia's small-object pools.  `ReductionTelemetry` can record
live/data estimates, and the recursive teaching driver can request periodic
collections.  Process resident-size measurements should be made in fresh
processes when comparing memory use because `Sys.maxrss` is a process-wide high
water mark.

Transform accumulation is drained while finalising so factors can be released
as they are consumed.  A full product tree at that point can increase the live
set substantially because a whole level of wide partial products coexists with
the next level.

## Telemetry

Telemetry is intended to locate optimisation targets, not to trace control-flow
debugging.  Retained timers partition useful work such as:

* Irregular entry/orientation work;
* CondUnknown rank discovery and exact applications;
* Heuristic1 and Heuristic2 representation QR, relative reduction, transform
  composition, compression, and exact application;
* Heuristic3 basis products, diagonal QR, orthogonal/triangular relative size
  reduction, writeback, and propagation;
* fused QR components, exact matrix products, compression, base cases,
  transform folding/finalisation, and garbage collection.

Nested child wall timers are intentionally not recorded because they overlap
with the child algorithms' own timers and make aggregate accounting misleading.

## Benchmarks

`src/lattices/large_benchmark.jl` compares deterministic lattice families with
cached fpLLL/reference timings.  The default benchmark algorithm is
`:heuristic`.  `h1-known-cond` uses the dense exact matrix

    B = I + v*v'

with `v_i = +/-1`; its singular values are `1` and `n + 1`, so its condition
number is known exactly and the case reliably exercises Heuristic1.

Julia JIT warm-up must cross the same dispatch boundaries and use the same
keyword signature as a measured call.  In particular, the large benchmark warms
the blocked triangular size-reduction path and passes a telemetry object so
compilation is not charged to the first timing.

The benchmark telemetry should be used to decide which optimisation is relevant
for a given family.  Orthogonal relative size reduction and exact integer matrix
products are independent targets; improving one does not remove the other.

## Not implemented

The following are outside the present serial-heuristic scope:

* `Threaded3` and threaded relative size reduction;
* flatter's proved reduction pipeline;
* fused-QR backends not selected by the serial heuristic dispatcher;
* multi-modular exact matrix multiplication.
