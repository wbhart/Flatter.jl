# Notes

Things to watch out for. Each entry is a reminder, not an explanation — the
reasoning is in the code comments and docstrings.

---

## Design choices in this package

* **`smsv.jl` defaults to blocked Householder QR.** `profile(...; blocked=true)` reaches for the recursive implementation to get n^omega with omega < 3 via Strassen. Only the diagonal of R is used for the profile, so the extra machinery buys complexity at the cost of a larger trusted surface. `blocked=false` is the conservative fallback if the profile ever looks wrong.

* **`size_reduction_triu.jl` blocking almost certainly isn't reaching Strassen.** At the current block size the tile products sit at or below the Strassen cutoff and go straight to the base multiplication — raise the block size to actually benefit. Blocking still pays for itself: cache effects, one workspace grabbed up front instead of per-operation allocation, and the option of packing a whole tile into one huge integer for a single GMP multiply (Kronecker substitution). That last is not implemented here, nor in the C++ as far as I know, presumably because it only wins once the BigInt entries are genuinely large.

* **`recursive_reduction.jl` ports `RecursiveGeneric` with `Proved3`'s hardcoded window schedule**, not the `SublatticeSplit` tree. `Proved2`/`Proved3` reference no split object, so this is the shortest route to a working end-to-end reducer. Swapping in the split tree later changes `_reduction_window` and nothing else in that file.

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

* **Fused QR returns `tau`, not compact-WY `T`.** Reflectors are generated one at a time, so `T` doesn't exist naturally; use `compact_wy_from_reflectors` before handing the factorisation to `apply_Qt!` or the relative size reduction kernels. flatter zeroes `tau[n-1]` at the end (its `Columnwise` discards Q anyway); we keep it, so the returned factorisation is genuine and `Q'B = R` is testable.

* **Fused QR returns R packed, subdiagonal not cleared.** Matches `householder_block`'s convention. flatter clears it because its `Columnwise` asserts `tau` empty. Use `triu(R)` if you want R alone.

* **The stagnation exit is ours; flatter has it commented out.** `Proved3::is_reduced` contains `if (iterations % 3 == 0 && !lattice_changed) { //return true; }` — written but disabled, with nothing else consuming `lattice_changed`. Without it the loop grinds indefinitely at a goal it cannot quite reach. We enable it: if a full cycle of windows leaves the true-scale profile unmoved, stop. `info.stopped` reports `:goal`, `:stagnated`, `:cap` or `:base_case`.

* **flatter's reduction loop has no iteration cap** (`for(iterations=0;;iterations++)`) — it relies on the goal being reachable. We add `max_iterations`. Hitting it is not an error; the basis is still valid, just less reduced, and `info.goal_met` reports which happened.

* **`goal_from_drop`'s proved branch drops the flag, but the branch is unreachable.** It calls `from_slope(n, slope)` instead of `from_slope(n, slope, proved)`, so it would return a *heuristic* goal — except all three call sites in flatter use the two-argument form. Reproduced verbatim anyway.

* **`params.proved` and `goal.proved` are independent flags.** The first (from `FLATTER_PROVED`) picks the *implementation*; the second picks the *acceptance test*. `params.cpp` always builds the initial goal with the two-argument `from_RHF`, so it is heuristic regardless; the only genuinely proved goal in flatter is constructed at `proved_1.cpp:100`. So reaching `Proved2`/`Proved3` without passing through `Proved1` gives proved implementations checked against a heuristic goal. Worth verifying before porting `Proved1/2/3`; irrelevant to the heuristic path.

---

## Julia numeric hazards

* **BigFloat arithmetic uses the global default precision, not the operands'.** Measured on 1.12.6: `arith=default`, `setindex=rebind`, `mul!=widened`. So a `Matrix{BigFloat}` has no precision as a matter of type — only of discipline.

* **Use `with_precision(f, bits)`, not `setprecision(BigFloat, bits) do ... end`.** The block form installs a `ScopedValue`, and every subsequent `BigFloat` allocation then resolves the precision through a `PersistentDict` lookup — ruinous in code that allocates temporaries in inner loops. `with_precision` sets the default directly and restores it in a `finally`. It is **not task-safe**; revisit every call site if this package ever threads. A bare `setprecision` with no restoration remains wrong — it leaks the precision to the caller.

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

* **Relative size reduction's precision requirement is set by the block magnitude ratio, not just conditioning.** Coordinates are known only to `max|B2| * 2^-p`, so reducing to the deadband needs `p > log2(max|B2| / min|R_jj|)` with margin. The `max|B2|` is the magnitude *after* reduction — only the component of `B2` in `B1`'s span can shrink, so a large orthogonal component sets a floor that no number of refinement passes can lower. Under-provisioning isn't an error: the convergence guard stops, `U` stays exact and unimodular, and only reduction quality suffers.

* **The second convergence guard is load-bearing.** `max_mu >= prev_max` (precision exhausted) is what stops a badly conditioned input at 53 bits looping forever. It is not redundant with the unit-magnitude test.

* **Size-reduction bound isn't 1/2 on the orthogonal kernels.** flatter uses a 0.51 deadband and terminates on a unit-magnitude test, so expect ~0.51 plus float error. Only `Triangular` achieves a strict 1/2. The exactly testable invariant on every path is `B2_new == B2_old + B1*U` over `BigInt`.

* **Relative size reduction never reduces `B2`'s columns against each other.** When checking against `smsv_gso([B1 B2])`, assert only on `mu[j, n1+c]` for `j <= n1`.

* **`blocked` size reduction is not a reordering of the elementary one.** `diag_above` multiplies the *original* tile by `U(j,j)`, where the elementary sweep would use the already-reduced columns. Different valid reduced representative — agrees with `:zz` on only ~2/3 of inputs. Test the contract, not equality.

* **`U` from fused QR is unimodular but not triangular.** Columns are processed left to right, but each is reduced against *all* its predecessors. Test `abs(det(U)) == 1` (Bareiss), not the triangular structure.

* **`reduce_basis` accepts any full-rank integer basis; `lattice_reduce` requires upper triangular.** Prefer the former unless the input is known triangular. It detects all four corner orientations and flips into place — reversing columns reorders basis vectors, reversing rows permutes coordinates, so neither changes the lattice. `info.path` reports `:triangular`, `:reoriented` or `:dense`.

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

* **Remaining structural cost: the fused QR is BLAS-2.** The trailing update applies one reflector to one column at a time, so it contains no matrix products at all. Panel blocking — accumulating reflectors and applying them as a compact-WY block update — would route it through `_householder_block_mul!` and hence Strassen. This is what flatter's `Heuristic3` does — and note that `Heuristic3` is flatter's SERIAL path (`Threaded3` is the threaded one) and contains no OpenMP at all, so its tiling is a cache and BLAS-3 optimisation, not a parallelisation device. Not yet attempted.
