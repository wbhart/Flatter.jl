# Notes

Things to watch out for. Each entry is a reminder, not an explanation — the
reasoning is in the code comments and docstrings.

---

## Design choices in this package

* **`smsv.jl` defaults to blocked Householder QR.** `profile(...; blocked=true)` reaches for the recursive implementation to get n^omega with omega < 3 via Strassen. Only the diagonal of R is used for the profile, so the extra machinery buys complexity at the cost of a larger trusted surface. `blocked=false` is the conservative fallback if the profile ever looks wrong.

* **The tiled size reduction beats the elementary one from dimension 128, and the mechanism is NOT Strassen.** Measured: 1.11x at dim 32, 1.10x at 64, 2.30x at 128, 1.99x at 256. `blocksize = 16` wins at every dimension and larger blocks are monotonically worse — which rules Strassen out, since 16 is exactly the BigInt cutoff so those tile products fall straight through to the base multiplication. What the tiling buys is cache locality and allocation behaviour. (An earlier note here guessed the opposite, that the block size was too small to reach Strassen. It was wrong.) The option of packing a tile into one huge integer for a single GMP multiply — Kronecker substitution — is also not the answer: it turns `n^3` multiplies of `b` bits into `n^2` multiplies of `nb` bits, worse by roughly `n^0.58` since GMP's multiplication is superlinear. Multi-modular multiplication is the technique that applies.

* **Both the legacy window schedule and the phase-3 `SublatticeSplit` feeder are available.** `schedule = :legacy` keeps the earlier hard-coded cycle for comparison. `schedule = :split` carries the phase-3 split tree through recursion, including its even-iteration stopping points and the recursive left/right reset on advance; this is the feeder `Heuristic3` expects. This is still not a full flatter port: the phase-1/phase-2 dispatcher is outside the current teaching implementation.

* **`fused_qr_size_reduction.jl` ports only flatter's `Columnwise`.** `ColumnwiseDouble` (same algorithm on doubles, plus a global power-of-two exponent shift to stop R overflowing) is not ported — BigFloat's exponent range makes the shifting unnecessary, and the `Float64` path here throws rather than silently producing `Inf`. `LazyRefine` needs a driver supplying prereduced prefixes; `Iterated` and `SeysenRefine` are unreachable from flatter's own dispatcher.

* **The two-column base case splits between Lagrange and Schoenhage at `schoenhage_threshold` (512 bits).** Both give a Gauss-reduced basis, so the crossover is a pure cost trade: Lagrange is quadratic in the bit size, Schoenhage quasi-linear, but Schoenhage pays for forming the Gram matrix and entering its recursion. flatter splits the same way at `prec < 1400`. Neither involves a precision policy.

* **`schoenhage` takes a Gram matrix, not a basis.** It returns `R == transpose(U) * G * U`; the driver forms `G = Bᵀ B` exactly and applies the returned `U` to the columns. Its reduction condition is exactly Gauss-reduced, i.e. what Lagrange produces — which is what makes the two cross-checkable.

---

## Divergences from flatter

* **Shift rounding.** flatter uses floor (`mpz_div_2exp`) in the compression path but truncation (`mpz_tdiv_q_2exp`) in `relative_size_reduction/triangular.cpp`. We use floor (`>>`) everywhere. Differs by one on negative entries, so no bit-for-bit agreement with the C++; harmless, since the shift is discarding low bits anyway. Compare invariants, not matrices.

* **No `mpfr_mul_z` in Julia.** flatter rounds `float * exact integer` once; `::BigFloat * ::BigInt` promotes first and rounds twice. Only lossy when the multiplier exceeds the working precision, i.e. mostly on the first refinement pass. Absorbed by the refinement loop — symptom would be more passes than the C++ needs, not a wrong answer. `ccall` to `mpfr_mul_z` if it ever matters.

* **flatter's `OrthogonalDouble` refinement loop is dead.** `max_mu_size` is never assigned, so the `while(true)` runs exactly once. We implement the real refinement (as in the MPFR kernel), since 53 bits needs it more, not less. Don't differential-test against the C++ double kernel.

* **The fused stagnation guard diverges from flatter.** flatter compares multiplier *magnitudes* against an absolute slack of `prec/2`, which is dimensionally odd: for large multipliers the slack vanishes and for small ones `mu_max - prec/2` goes negative so it always fires. We compare exponents instead, matching `relative_size_reduction.jl`.

* **The fused three-tier reduction test compares in `S`, not `double`.** flatter converts both operands out of MPFR and compares the quotient in `double` to save a division. Differs only for entries within a double's rounding error of the threshold, where either answer is fine.

* **Fused QR returns `tau`, not compact-WY `T`.** Reflectors are generated one at a time, so `T` does not exist naturally. The generic `apply_Qt!`/compact-WY paths can build it with `compact_wy_from_reflectors`; Heuristic3 instead consumes the cached `R,tau` directly, matching flatter's MPFR orthogonal kernel. flatter zeroes `tau[n-1]` at the end (its `Columnwise` discards Q anyway); we keep it, so the returned factorisation is genuine and `Q'B = R` is testable.

* **Fused QR returns R packed, subdiagonal not cleared.** Matches `householder_block`'s convention. flatter clears it because its `Columnwise` asserts `tau` empty. Use `triu(R)` if you want R alone.

* **The stagnation exit is ours; flatter has it commented out.** `Proved3::is_reduced` contains `if (iterations % 3 == 0 && !lattice_changed) { //return true; }` — written but disabled, with nothing else consuming `lattice_changed`. Without it the loop grinds indefinitely at a goal it cannot quite reach. We enable it: if a full cycle of windows leaves the true-scale profile unmoved, stop. `info.stopped` reports `:goal`, `:stagnated`, `:cap` or `:base_case`.

* **flatter's reduction loop has no iteration cap** (`for(iterations=0;;iterations++)`) — it relies on the goal being reachable. We add `max_iterations`. Hitting it is not an error; the basis is still valid, just less reduced, and `info.goal_met` reports which happened.

* **`goal_from_drop`'s proved branch drops the flag, but the branch is unreachable.** It calls `from_slope(n, slope)` instead of `from_slope(n, slope, proved)`, so it would return a *heuristic* goal — except all three call sites in flatter use the two-argument form. Reproduced verbatim anyway.

* **`params.proved` and `goal.proved` are independent flags.** The first (from `FLATTER_PROVED`) picks the *implementation*; the second picks the *acceptance test*. `params.cpp` always builds the initial goal with the two-argument `from_RHF`, so it is heuristic regardless; the only genuinely proved goal in flatter is constructed at `proved_1.cpp:100`. So reaching `Proved2`/`Proved3` without passing through `Proved1` gives proved implementations checked against a heuristic goal. Worth verifying before porting `Proved1/2/3`; irrelevant to the heuristic path.

---

## Julia numeric hazards

* **BigFloat arithmetic uses the global default precision, not the operands'.** Measured on 1.12.6: `arith=default`, `setindex=rebind`, `mul!=widened`. So a `Matrix{BigFloat}` has no precision as a matter of type — only of discipline.

* **Use `with_precision(f, bits)`, not `setprecision(BigFloat, bits) do ... end`.** The block form installs a `ScopedValue`, and every subsequent `BigFloat` allocation then resolves the precision through a `PersistentDict` lookup — ruinous in code that allocates temporaries in inner loops. `with_precision` sets the default directly and restores it in a `finally`. It is **not task-safe**; revisit every call site if this package ever threads. A bare `setprecision` with no restoration remains wrong — it leaks the precision to the caller.

* **Wrapping a precision block around the caller hides the bug; the callee must wrap itself.** Every `heuristic3.jl` test passed while the driver failed, because the tests wrapped the call in `setprecision` and the driver did not. `compact_wy_from_reflectors` allocates its `T` at the DEFAULT precision, so it disagreed with an `R` built at the working precision and `assert_precision` refused it. Any function that allocates BigFloats and is called from unwrapped code has to establish its own scope — and at least one test must call it without a surrounding block, or the whole class of fault is invisible.

* **Wrap whole computations, not just allocations.** `Matrix{BigFloat}(undef, ...)` pins nothing; it is the arithmetic that reads the default. Applies to `householder_block`, `apply_Qt!` and every BigFloat kernel — none of them establish their own scope.

* **`precision(A[1,1])` does not describe the matrix.** Entries can differ after a stray write. Use `uniform_precision` / `assert_precision`.

* **Widening costs speed, not accuracy.** Extra bits are noise on an already-correct value. But MPFR cost scales with precision, and the "precision exhausted" guard is calibrated against the precision you think you're at.

* **Convert BigFloat matrices explicitly across recursion boundaries** with `at_precision`. Returning one from a scoped block is safe; doing further arithmetic on it outside a matching block is not.

* **`Float64(::BigInt)` overflows to `Inf` on lattice-sized entries.** Use `float64_matrix`, which pulls out a common power-of-two scale. Common, not per-column — per-column would distort the ratios the multipliers depend on.

* **In-place MPFR/MPZ mutate the OBJECT, not the matrix slot.** Only safe on values the callee owns. `R` inside the fused factorisation qualifies, because it is filled entry by entry with `_to_float`. Two tests guard this: "the caller's matrices are never mutated in place" and "copy of a BigFloat matrix is shallow".

* **`Matrix{BigInt}(undef, n, n)` holds undefined REFERENCES, not zeros.** `BigInt` and `BigFloat` are mutable, so an `undef` matrix of them has genuinely unassigned slots and any generic iteration over it throws `UndefRefError`. A matrix can legitimately sit in that state mid-computation — the driver's output transform `U` is only filled at the end. Guard with `isassigned` when walking a matrix that might not be fully populated.

* **Duplicating a `BigFloat` is harder than it looks, and there are two traps.** `copy` and `Matrix{T}(A)` are SHALLOW — they duplicate the array of references, so every entry is still the same object. And `BigFloat(x)` on a `BigFloat` returns **x itself** when the precision already matches, so the obvious `[BigFloat(x) for x in A]` is also a no-op. The reliable form is `_mpfr_set!(BigFloat(), x)`: allocate, then write. `fill!(v, zero(S))` is worse still — one shared object in every slot. Both traps are pinned by the "duplicating a BigFloat matrix" testset; between them they produced failures that looked numerical but were one code path reading another's output.

* **`at_precision` allocates and writes rather than constructing.** Its whole purpose is an independent copy, and `BigFloat(x; precision = p)` short-circuits to `x` when `p` already matches. Same reasoning applies anywhere else that needs a genuine duplicate.

* **The in-place kernels accumulate with `mpfr_fma`**, which rounds once where the generic path rounds twice, so the two are **not bit-identical** — the in-place path is slightly more accurate. Tests compare with a tolerance.

---

## Algorithm facts that look like bugs

* **`Profile::get_drop` is not first-minus-last.** It is the spread *less the sum of clean upward gaps* — places where every entry to the right sits strictly above every entry to the left. Those gaps are exactly what block compression removes, so they don't count as un-reducedness. `get_spread` is the plain `max - min`. They coincide on a decreasing profile, which is why a first-minus-last stand-in passes most tests. Invariant: `0 <= drop <= spread`.

* **The heuristic `goal_check` has three conditions and all three do work.** Total drop, half-mean separation, and middle-span drop. A profile flat in both halves with a step between them at 95% of the budget is rejected despite its total drop fitting, and the middle-span condition alone rejects profiles the other two accept. Don't simplify it to the drop condition — that's the *proved* goal's test, and it would let the recursion settle for a basis still improvable across the half boundary.

* **`with_best_slope` preserves the budget, it does not move it.** A heuristic goal's target slope splits as `best_slope + gap`; changing the base slope re-attributes the same total and leaves `goal_max_drop` exactly unchanged. Its precondition is `slope < target slope`.

* **`goal_from_rhf` clamps below `2^(BKZ_BEST_SLOPE/2) ≈ 1.0109`.** Asking for a better root Hermite factor than BKZ achieves silently gives the clamp.

* **`goal_slope` returns different kinds of thing for the two goal families.** Proved: an actual slope. Heuristic: the raw `quality` scalar, which is *not* a slope. For the slope a heuristic goal really targets, use `goal_max_drop(g) / g.n`.

* **Heuristic `goal_max_drop(g) / n` recovers the input slope exactly.** The `shape` function cancels between `from_slope` and `max_drop`, so `quality` is a pure scale factor with no dimensional meaning of its own. `_goal_shape(1) == 0`, so dimension-one goals are guarded.

* **Heuristic3 must reuse the QR it just computed.** An earlier safety check compared `relative_size_reduction_precision(B1)` (an absolute-magnitude policy for the generic kernel) with Heuristic3's working precision (chosen from the post-child profile spread), and therefore refactorised almost every reduced tile. That was unlike flatter and slow. The apparent need for it was masking a cache bug: `householder_block` returns compact-WY `T` as its second result, and linear indexing of that matrix had been stored as if it were `tau`, making almost every cached coefficient zero. Heuristic3 now stores real `tau` and passes cached `R,tau` directly to the direct-reflector kernel. Standalone `relative_size_reduction!` still chooses its own precision when it has to build a factorisation itself.

* **Relative size reduction's precision requirement is set by the block magnitude ratio, not just conditioning.** Coordinates are known only to `max|B2| * 2^-p`, so reducing to the deadband needs `p > log2(max|B2| / min|R_jj|)` with margin. The `max|B2|` is the magnitude *after* reduction — only the component of `B2` in `B1`'s span can shrink, so a large orthogonal component sets a floor that no number of refinement passes can lower. Under-provisioning isn't an error: the convergence guard stops, `U` stays exact and unimodular, and only reduction quality suffers.

* **The second convergence guard is load-bearing.** `max_mu >= prev_max` (precision exhausted) is what stops a badly conditioned input at 53 bits looping forever. It is not redundant with the unit-magnitude test.

* **Size reduction bounds entries against the DIAGONAL they are reduced against, not against `max|B|`.** So `max|B|` need not fall, and can drift up by a few parts per billion as other rows' updates propagate. A basis whose off-diagonal entries are already small relative to the diagonal is already size-reduced, and a test that expects the maximum to shrink there is testing nothing. To exercise a reduction, the off-diagonal entries must exceed the diagonal of the block they are reduced against.

* **Size-reduction bound isn't 1/2 on the orthogonal kernels.** flatter uses a 0.51 deadband and terminates on a unit-magnitude test, so expect ~0.51 plus float error. Only `Triangular` achieves a strict 1/2. The exactly testable invariant on every path is `B2_new == B2_old + B1*U` over `BigInt`.

* **Relative size reduction never reduces `B2`'s columns against each other.** When checking against `smsv_gso([B1 B2])`, assert only on `mu[j, n1+c]` for `j <= n1`.

* **`blocked` size reduction is not a reordering of the elementary one.** `diag_above` multiplies the *original* tile by `U(j,j)`, where the elementary sweep would use the already-reduced columns. Different valid reduced representative — agrees with `:zz` on only ~2/3 of inputs. Test the contract, not equality.

* **`U` from fused QR is unimodular but not triangular.** Columns are processed left to right, but each is reduced against *all* its predecessors. Test `abs(det(U)) == 1` (Bareiss), not the triangular structure.

* **`reduce_basis` accepts any full-rank integer basis; `lattice_reduce` requires upper triangular.** Prefer the former unless the input is known triangular. It detects all four corner orientations and flips into place — reversing columns reorders basis vectors, reversing rows permutes coordinates, so neither changes the lattice. `info.path` reports `:triangular`, `:reoriented` or `:dense`.

* **flatter's double-precision paths are switched off, deliberately.** `Heuristic3::set_precision` guards the swap with `if (prec <= 53 && R.type() == MPFR && false)` — the `&& false` is in the source. `OrthogonalDouble`'s dead refinement loop is consistent with that: nobody leaves a never-executing loop in a path that runs. So `ColumnwiseDouble`, `OrthogonalDouble` and `ElementaryLL` exist but are unreachable.

* **Why a 53-bit driver is not straightforward, and what would make it possible.** The obstacle is not the implementation — `fused_qr_size_reduction` already takes `float_type = Float64` — it is the precision policy. `lll_precision` is `2*spread + 30 + 2n`, so 53 bits would require the post-compression profile spread to be near zero, and compression bounds the spread without making it tiny. What could work is iterative refinement: recompute the coordinates from the EXACT integer basis each pass, so low precision plus an exact residual converges. Both the fused QR's re-orthogonalisation and the relative size reduction's refinement loop already have this shape, so the experiment is not unreasonable — the open question is whether 53 bits converges for the spreads that actually arise, and how many extra passes it costs. flatter did not find out: its double paths are disabled with `&& false`.

* **A wider fixed-precision type only fits the dense path, and only for medium entries.** The driver runs at roughly `4n` bits (the compressed spread is about `n`), which passes 106 bits at around dimension 26 — below `base_cutoff`, so a double-double can never serve the fused QR. The dense path is different: `householder_precision` is about `34.5 + entry_bits`, so entries of 19-71 bits fall between `Float64` and a double-double. `float_tiers` is the seam; pass `(Float64, Double64, BigFloat)` with DoubleFloats loaded. Nothing in the package depends on it.

* **`aggressive` precision is free but small: measured 1.0-1.3x, with quality identical on all 33 benchmark instances.** `log2|b1|` matched to the last digit everywhere, so the concern that `spread + 30` would undershoot did not materialise — even though ball arithmetic puts the certifying threshold well above it. Certification is a worst-case statement about radii; the refinement loop and the 0.51 deadband absorb imprecision that a rigorous bound cannot ignore, which is what those heuristics are for. Left off by default anyway: the gain is small, it is not certifiable, and it is only tested to dimension 128. flatter likewise makes it an opt-in environment variable.

* **Precision is not the performance lever at these widths.** Cutting the working precision by roughly threefold bought only 1.0-1.3x, because at 120-420 bits (two to seven limbs) MPFR's cost is dominated by per-operation overhead — allocation, call, indirection — rather than by limb arithmetic. Fewer bits does not mean less overhead. The lever is fewer operations: panel-blocking the fused QR, or cheaper integer products.

* **`aggressive` is already a second precision policy, and it may undershoot.** `lll_precision` gives `2*spread + 30 + 2n` by default and `spread + 30` under `aggressive` — 606 against 286 bits at dimension 32 with spread 256, where ball arithmetic put the certifying threshold near 406. So the default has headroom AND the aggressive setting may be too little; the useful policy is probably between them. `precision_tradeoff()` in the benchmark compares both on identical instances, reporting `log2|b1|` and the stopping reason alongside the time, because a precision that is too low shows up as a longer vector or a run that starts stagnating, not as an error.

* **The Arb probe says the precision policy is over-provisioned, and cutting it turned out not to matter much.** At `2*spread + 30 + 2n` every quotient certifies with 200-500 bits of margin. But the measured benefit of actually reducing it was 1.0-1.3x — see the entries above on why. `headroom()` in `lattices/arb_probe.jl` bisects for the certifying minimum if the question comes up again.

* **Our only hardware-precision path is the dense-path QR.** Everything else is BigFloat and BigInt, and in the driver a Float64 path is unreachable by construction: `lll_precision` is `2*spread + 30 + 2n`, above 53 bits for any n over about 12, and `base_cutoff` keeps the driver from seeing smaller. The dense path differs because `householder_precision` bottoms out at exactly 53, which is what a small-entry basis gets — so that factorisation runs in `Float64` via LAPACK's `geqrf!` — the same Householder algorithm as this package's generic version, but with BLAS-3 panel updates, and it leaves R in the upper trapezoid, which is the only part `to_integer_lattice` reads. Falls back to BigFloat when 53 bits will not resolve the profile.

* **The memory instrumentation is not free and is off by default.** `held_bytes` walks every entry of four matrices and inspects each value's allocation — O(n^2) big-integer inspections per iteration, at every level. It was written for a one-off investigation and then left running in every benchmark, where it was untimed and so showed up only as "unaccounted". Set `telemetry.measure_memory = true` when the question is memory; `lattices/memory_scaling.jl` does.

* **Reporting `info.profile` costs a factorisation, and it is charged to us in comparisons.** The reduced basis is no longer triangular, so a profile means a full arbitrary-precision QR. fplll computes nothing of the kind, so a benchmark that reports our time against theirs is comparing a reduction-plus-factorisation with a bare reduction. `want_profile = false` skips it; the benchmark prints its share when it exceeds 1% of a run. Internally nothing needs it: the driver tracks its own profile from the R diagonal, and the dense path's round test reads the approximation's diagonal, which is triangular and therefore free.

* **The base-case profile was computed and thrown away.** `lattice_reduce!` returned `info.profile` from every base case, which costs a full arbitrary-precision factorisation — and every recursive call discards the info it is handed. q-ary at dimension 96 makes 507 base-case calls, so that was 507 wasted factorisations. Now computed only at `_depth == 0`; recursive base cases return an empty profile. A large share of the "unaccounted" time in the benchmark was this.

* **The dense path's QR dominates it, so precision policy matters there.** Measured on ideal lattices: the factorisation is 41-49% of the whole run, against 3.6-6.8% for the fused QR inside the driver. Use `householder_precision`, not `lll_precision` — the latter's `2n` term describes the compressed basis inside the driver and is meaningless for factorising a raw one, asking 302 bits where 53 suffice at dimension 128 with 8-bit entries. The starting estimate is optimistic for an ill-conditioned basis, so `_dense_approximation` doubles until the factor is usable, which is `CondUnknown`'s discovery loop in miniature.

* **The dense path diverges from flatter.** flatter routes a non-triangular basis to `CondUnknown` (phase 1, `log_cond = 0` deliberately, since `Heuristic1` asserts it positive). We instead QR-factor, round the R factor to an integer triangular lattice with `to_integer_lattice`, reduce that, and apply the resulting transform to the original — valid because ANY unimodular transform is valid, so a poor approximation costs quality and never correctness. Iterated a few rounds, since each re-factorises an already better basis. Rank-deficient input is reported rather than handled; that is what `CondUnknown` is for and it is not ported.

* **After flipping, the transform needs its ROWS reversed, not its columns.** Reducing `Q B P` to `Q B P U'` means the transform for the original basis is `P U'`. flatter does the same via `flip_mat(U, flip_cols, false)`, which is easy to misread as a column flip.

* **Input to the driver is upper triangular; output is a general basis.** `B_out = B_in * U` has a meaningless diagonal that may contain zeros — reading it as a profile gives `-Inf`. The profile is in `info.profile`, from the R factor. flatter is the same. Re-triangularise before feeding a result back in.

* **Compression acts on the R factor, never on the basis product.** `B_next = B * U_window` is *not* triangular, so it must be re-factored by the fused QR first, and the resulting R is what gets compressed. Compressing `B_next` directly produces a singular basis within a couple of iterations.

* **Compression only fires on profiles that jump *upward*.** The gap test needs the right block higher than the left, which never happens on a descending profile — so on a normal lattice profile the shifts are all zero and only the uniform truncation applies.

* **Compression cannot vanish a diagonal entry**, given the precision policy: the smallest column retains about `spread + 30 + 2n` bits. If a diagonal ever does hit zero, suspect the precision policy, not the shifts.

* **`collect_U` pairing is off-by-one by design.** The compression stack holds one more entry than the transform stack. The last compression is discarded, then transform *k* pairs with compression *k−1*. Getting this wrong throws inside `conjugate_transform`'s block-structure check rather than silently corrupting the result.

* **`compression_shifts`' truncation depends linearly on the precision argument.** Call with `precision = 0` to recover the constant: `truncation = offset - lll_precision(spread, n)`.

* **Compare profiles at true scale, not compressed.** Compression shifts the profile every iteration; the stagnation test uses `profile .+ offsets`.

* **The exact invariants hold unconditionally.** `B_out == B_in * U` and `|det U| == 1` are true regardless of working precision, of whether the goal was met, and of the iteration cap. When something looks wrong, check these first: if they hold, it's a precision or goal problem, not an algorithmic one.

---

## Performance

* **The Strassen cutoff depends on element type.** `BigInt` wants 16, everything else 32. For floats the leaf is BLAS and recursing far is a loss; for `BigInt` the leaf is a scalar loop and the entries dominate. If you preallocate a workspace, size it with `strassen_workspace_length(T, n)` — the type-free method uses the generic cutoff and would under-size the buffer for `BigInt`.

* **Never materialise the embedded window transform.** It is the identity outside one `w x w` block, so a general product costs `O(n^3)` to compute something touching `O(n*w^2)` entries. `_apply_window_right`/`_apply_window_left` do the block product and copy the rest.

* **Measure before optimising here.** Three separate predictions about the bottleneck were wrong: the fused QR was expected to dominate and was the smallest phase; the finalise phase was invisible until instrumented; the final size reduction was expected to dominate and was negligible. The real costs turned out to be Julia-level overheads — scoped-precision lookups and GMP allocation — not the algorithm.

* **Benchmark timings vary by up to 2x** with GC on BigInt work. `lattices/benchmark.jl` repeats cheap runs and keeps the fastest; the Strassen sweep in `lattices/profile_lift.jl` collects before each sample and prints the noise floor. Treat any single-run difference smaller than that floor as meaningless.

* **Memory is the binding constraint at large dimension, and it is live data, not collection lag.** `--heap-size-hint` made no difference at dim 260, and measured live bytes grow steeply with dimension — the collector cannot free what is reachable. The driver still asks for `GC.gc(false)` per iteration at `gc_dimension` (200) since arbitrary-precision limbs are malloc'd and freed by finalizers, but expect little from it. `telemetry.peak_live` and the benchmark's memory line are how to tell the two apart.

* **`Sys.maxrss` deltas are misleading across repeated runs.** It is a high-water mark for the whole process, so a later, larger run can report a delta of zero simply because an earlier one already raised the mark. Use `gc_live_bytes` for comparisons.

* **O(log k) partial products is not O(log k) bytes.** Entry widths add across a product, so a partial product of 2^j factors is ~2^j times as wide as one factor; summing over the balanced stack gives Theta(k) bits — the same order as keeping every factor. The balanced tree buys multiplication speed, not memory.

* **`low_memory` (bounded accumulation) is off by default but does raise the ceiling.** It folds every factor into one running product rather than keeping O(log k) partial products. An early measurement suggested it was much worse, but that used the contaminated absolute live-bytes counter and should not be trusted. What is observed: with it enabled the reducer clears dimensions it otherwise cannot. Re-measure before drawing conclusions either way.

* **The peak is in the finalise phase, not the loop.** The transform stack's partial products are at their widest when they are finally combined, so that is where a run dies — symptom: memory is comfortable throughout and the process is killed near the end. `_drain_transform` folds by popping, releasing each factor as it is consumed; a product tree would hold an entire level while building the next, roughly doubling the live set at exactly the wrong moment. The balanced tree is still used while the stack is being BUILT, where factors are narrow and the pairings keep multiplications cheap.

* **Finalizable objects survive one collection.** `BigInt` and `BigFloat` both carry finalizers; the first GC pass queues the finalizer, the second releases the limbs. So a single `GC.gc()` leaves them counted as live, and an incremental `GC.gc(false)` cannot reclaim them at all. Measure retained memory after **two** full passes, and use `gc_full` if periodic collection is to have any chance of helping.

* **Memory steps with recursion depth, it does not grow smoothly.** Each extra level adds another level's working state to the live set, so the curve jumps whenever depth increases (q-ary: depth 2 at dim 128, depth 3 at dim 160, levels 559 -> 6865). Fitting a growth exponent across a depth transition badly overstates the trend. Fit within a depth, and expect any projection past the next transition to be low.

* **Measure memory as `Sys.maxrss` in a fresh process. Nothing else here is trustworthy.** `gc_live_bytes` is absolute (so earlier runs in the same session inflate it) AND over-reports for this workload: Julia's malloc'd-byte counter drifts upward under heavy GMP/MPFR reallocation, to the point where it exceeded the process's actual resident size. `Sys.maxrss` is a process-wide high-water mark, so within one session a later run can report zero growth because an earlier one already raised it. `lattices/memory_scaling.jl` spawns one process per measurement for exactly these reasons.

* **Measured resident memory, q-ary family, isolated processes** (includes ~250 MB of Julia runtime): dim 64 → 512 MB, dim 96 → 1126 MB, dim 128 → 1590 MB, dim 160 → 3648 MB. Roughly `dim^3.7` across the last pair, but that pair straddles a depth increase, so treat it as an upper bound on the trend.

* **The memory measures are gated by dimension, and the gates matter more than the measures.** Collection is worth threefold in memory and costs a little time, so it only pays where memory is actually a constraint: excess with no collection is 44 MB at dim 64, 580 at 96, 1141 at 128, 3092 at 160. Defaults: collect every 2 iterations above dimension 128, switch to bounded accumulation above 256. Below those, reduction runs unencumbered. Setting the gates too low costs speed at small dimensions for no benefit, and shows up as run-to-run variance.

* **`malloc_trim` measured no saving at all**, so `trim_memory` defaults off. The call is retained for the case where a different allocator behaves differently.

* **`malloc_trim` does nothing here.** Measured saving across dim 64-160: 0%. Whatever holds the memory is not free pages the C allocator can return.

* **On WSL, the memory ceiling is not the machine's.** WSL2 defaults to half of Windows RAM and does not readily return freed pages, so an OOM there can be a VM limit rather than an algorithmic one. `.wslconfig` sets it.

* **Families declare their own size ranges.** They do not cost the same at a given dimension — q-ary is by far the most expensive, relation and ideal cheap enough to run much further. `run_family(name)` uses the declared range; passing `sizes` overrides.

* **Two families have a known answer, which is a sharper test than "how short".** `relation` recovers the minimal polynomial of `radicand^(1/degree)`, so `found_planted` says whether reduction found the true relation, not merely something short — this is the integer-relation application of LLL. `knapsack` likewise plants a subset-sum solution. `ideal` is the ideal `(g)` in `Z[x]/(f)`, whose short vectors are small elements of the ideal — ideal-SVP, and the same computation as looking for small generators in class group work.

* **The benchmark generators are shape approximations, not standard instances.** The q-ary modulus in particular defaults to `2^(2n)+1`, making `log2 det` quadratic in dimension — chosen to stress compression, not to match any published convention. Fine for tracking this package against itself and against fplll on identical input; not comparable with the literature. Use fplll's `latticegen` or the Darmstadt challenges for that.

* **Which cost dominates depends on the family and on the representation update.** Integer products in `matmul`, transform folding and final application can dominate some families, while QR/size reduction dominates others. Multi-modular integer multiplication remains the lever for the former. For the latter, the useful alternative is the Heuristic3 tiled representation update with direct Householder QR/SR; compact-WY panelling of the monolithic BigFloat QR did not pay in the measured regime.

* **Dimension-256 baseline, AMD Ryzen 7 5800H, after stage 1 (blocked size reduction).** Ours against fpLLL in seconds: random-triangular 8.1/0.87, scrambled 87.3/82.5, knapsack 14.4/3.11, q-ary 424.8/12092.5, relation 981.2/960.4. Against the pre-blocking baseline that is 1.24x, 1.19x, 0.96x, 1.20x and 1.08x, and `size-red` roughly halved on every family. q-ary is 28.5x faster than fpLLL.

  | family | fusedQR | (size-red, reorth, trailing) | finalise | gc | fold-in | matmul |
  |---|---|---|---|---|---|---|
  | random-triangular | 28.9% | 15.9, 2.4, 2.9 | 30.9% | 5.6% | 16.1% | 5.0% |
  | scrambled | 25.3% | 9.2, 6.4, 8.4 | 17.0% | 14.1% | 19.7% | 9.0% |
  | knapsack | 15.0% | 4.6, 3.7, 5.9 | 28.8% | 21.9% | 22.7% | 6.6% |
  | q-ary | 31.4% | 12.1, 8.2, 9.7 | 9.1% | 19.8% | 21.0% | 6.1% |
  | relation | 41.7% | 3.9, 16.2, 21.2 | 37.6% | 2.9% | 2.6% | 1.2% |

  `lattices/reference_times.tsv` holds these and fpLLL's timings, keyed by a hash of the basis. The `tiled` path is NOT what this measures: it is off by default.

* **Global transform products remain a separate lever from Heuristic3.** `fold-in`, final application and other exact matrix products are not removed by changing the representation update, even though Heuristic3 itself also performs structured integer tile products. Multi-modular multiplication is therefore still relevant after Heuristic3 reaches parity. `gc` can also be a material share on BigInt-heavy families and should be re-swept independently.

* **At dimension 256 on q-ary we are 22x faster than fpLLL** — 536s against 12,092s — while `scrambled` and `relation` sit at parity and `random-triangular` and `knapsack` remain behind. That is the flatter result: the recursion is worth its overhead once the dimension is large enough, and the families where it wins are the ones with a profile worth compressing.

* **Whether panel blocking helps depends on the family, at dimension 256.** relation's fused QR is 41.9% of which trailing is 19.7% and re-orthogonalisation 15.7% — 35% addressable — against size reduction at 6.0%. random-triangular is the exact inverse: 30.4% size reduction against 2.4% trailing. So blocking would nearly halve relation's factorisation and do almost nothing for random-triangular.

* **Warm-up must cross the same dispatch boundaries as the benchmark.** Julia specialises aggressively, so a warm-up that stays below `base_cutoff` does not compile recursion, fused representation updates, transform folding or finalisation. Special keyword combinations such as `tiled = true` should likewise be warmed before interpreting a one-shot timing. Best-of-several remains useful for cheap cases because GC noise is large.

* **The standard benchmark warm-up must include recursion.** With `base_cutoff = 32`, warming only dimensions 24 and 16 left the first recursive case, dimension 48, completely cold. Its first timing was 2.27s versus 4.32ms for fpLLL (a spurious 526x), carried about 47% unaccounted compilation time and exceeded `repeat_under`, so no clean repeat was taken. `warm_up()` now exercises a dimension-48 triangular reduction as well as the base and dense paths before measurements begin.

* **The phase mix shifts with dimension, so a verdict taken at one size need not hold at another.** relation's fused QR at dim 97 is size-reduction-dominated (15.9% against 12.6% trailing); at dim 256 it inverts (6.9% against 19.9% trailing, plus 15.0% re-orthogonalisation). random-triangular goes the other way — 18.7% size reduction against 1.3% trailing at 256. So "panel blocking is not worth doing", concluded from dim 96-128 figures, is wrong for relation at 256 and right for random-triangular. Measure at the size you care about.

* **Heuristic3 reaches its crossover by dimension 256 in this Julia port once it is fed and implemented like flatter.** On the knapsack probe, `split/plain` and `split/tiled` measured 3.76s and 3.73s with identical quality; the six Heuristic3 updates cost 0.84s in total, slightly less than the roughly 0.90s fused-update work they replace. Random-triangular measured 1.58s/1.65s but was already at the goal at the root, so it performed no Heuristic3 update and is not a useful kernel timing. At dimension 128 the tiled route is still slower. Characterise the crossover rather than assuming Heuristic3 only helps at dimensions beyond the machine's reach.

* **Do not generalise the win from blocked exact arithmetic to BigFloat Householder work.** Blocking the exact size reduction still pays because it batches `B`/`U` updates and removes BigInt allocation pressure. In contrast, compact-WY BigFloat QR/SR was a major loss at the Heuristic3 tile sizes: switching orthogonal SR to direct saved reflectors cut that kernel substantially, and switching tile QR to direct Householder cut the knapsack/split QR component from about 1.20s to 0.05s. `Matrix{BigFloat}` does not get the ordinary BLAS cache/SIMD payoff that motivates compact-WY. Multi-modular multiplication remains a separate possibility for exact BigInt products, not a reason to block MPFR/BigFloat QR.

* **Blocking the size reduction pays; panelling the trailing update does not.** `blocksize` defers the `B` and `U` updates to one batch per block of predecessors — exact, since only `R` is read during the sweep — and cut random-triangular from 10.11s to 7.86s at dimension 256, with `size-red` falling from 30.4% to 12.0%. `panelsize` defers a reflector's trailing update until its panel is flushed as one compact-WY block; measured at dimension 256 it is SLOWER at every panel width tried, and the factorisation's share of the run rises. A flush allocates a compact-WY factor, a `k` by `n` work matrix and a Strassen workspace per panel, per level, per iteration, while the trailing update it replaces is 2.9% of random-triangular and 6.5% of knapsack. Left in, defaulting to off; the relation family spends 19.7% in the trailing update and has not been measured.

* **Panel blocking the monolithic fused QR was not worthwhile in the measured cases.** Splitting that factorisation shows the trailing rank-1 update is only a modest share on most families, and compact-WY BigFloat panelling adds its own overhead. This result says nothing against Heuristic3: the earlier note that gave Heuristic3 the same ceiling was wrong, because Heuristic3 changes the tile decomposition, relative reductions and phase-3 schedule rather than merely batching the monolithic trailing update.

* **The dominant cost is the reduction itself, which is a good place to be.** `size reduction` computes quotients and applies them to `B`, `U` and `R` — the actual work, not factorisation overhead. Its integer half allocates one `BigInt` per element because `B` and `U` may share entry objects with a copy the caller holds, so slots must be rebound rather than mutated. Letting the caller assert ownership would remove that allocation; that is the remaining lever here, and it is a small one.

* **For MPFR/BigFloat Heuristic3, direct Householder is the right baseline.** flatter's MPFR QR dispatcher uses its direct `HouseholderMPFR` implementation; the blocked QR branch is disabled. The Julia measurements agree: after replacing compact-WY tile QR with the allocation-controlled direct reflector primitives, the knapsack/split QR share fell from about 1.20s to 0.05s. Keep Strassen/multi-modular work aimed at exact integer matrix products unless a future BigFloat matrix-multiplication backend changes this tradeoff.

* **Deferred threading target: orthogonal relative size reduction is column-parallel.** Upstream flatter parallelises the target columns in its MPFR orthogonal relative-size-reduction kernel with an OpenMP `taskloop`; the Julia port is deliberately serial for now. The corresponding loop is `for column in 1:n2` inside `_relative_orthogonal_reflectors!` in `relative_size_reduction.jl`. For a fixed reference block, `B1`, the saved Householder `factors` and `tau` are read-only, while target column `j` writes only `B2[:, j]`, `U[:, j]` and, when requested, `R2[:, j]`. Thus those columns can be distributed independently across Julia threads. Each worker must have its own floating workspace `r` and its own `previous`/`passes` state; do not share the current single `r` vector. Keep the reflector precision fixed before entering the threaded region rather than changing BigFloat precision inside workers. This one change would accelerate both H2's `LatRedRelSR` path and H3's orthogonal SR path, because both call `_relative_orthogonal_reflectors!`. Do **not** parallelise H3's bottom-up row-tile loop: reducing against a lower diagonal tile changes what the higher tile sees, so that ordering is a real dependency. A future implementation should retain the serial path for one-thread runs and probably use a size threshold before spawning tasks, since small `n2` blocks will not amortise scheduling/workspace overhead.
