* **`smsv.jl` defaults to blocked Householder QR.** `profile(...; blocked=true)` reaches for the recursive implementation to get n^omega with omega < 3 via Strassen. Only the diagonal of R is used for the profile, so the extra machinery buys complexity at the cost of a larger trusted surface. `blocked=false` is the conservative fallback if the profile ever looks wrong.

* **`size_reduction_triu.jl` blocking almost certainly isn't reaching Strassen.** At the current block size the tile products sit at or below the Strassen cutoff and go straight to the base multiplication — raise the block size to actually benefit. Blocking still pays for itself: cache effects, one workspace grabbed up front instead of per-operation allocation, and the option of packing a whole tile into one huge integer for a single GMP multiply (Kronecker substitution). That last is not implemented here, nor in the C++ as far as I know, presumably because it only wins once the BigInt entries are genuinely large.

* **Shift rounding differs from flatter.** flatter uses floor (`mpz_div_2exp`) in the compression path but truncation (`mpz_tdiv_q_2exp`) in `relative_size_reduction/triangular.cpp`. We use floor (`>>`) everywhere. Differs by one on negative entries, so no bit-for-bit agreement with the C++; harmless, since the shift is discarding low bits anyway. Compare invariants, not matrices.

* **No `mpfr_mul_z` in Julia.** flatter rounds `float * exact integer` once; `::BigFloat * ::BigInt` promotes first and rounds twice. Only lossy when the multiplier exceeds the working precision, i.e. mostly on the first refinement pass. Absorbed by the refinement loop — symptom would be more passes than the C++ needs, not a wrong answer. `ccall` to `mpfr_mul_z` if it ever matters.

* **flatter's `OrthogonalDouble` refinement loop is dead.** `max_mu_size` is never assigned, so the `while(true)` runs exactly once. We implement the real refinement (as in the MPFR kernel), since 53 bits needs it more, not less. Don't differential-test against the C++ double kernel.

* **The second convergence guard is load-bearing.** `max_mu >= prev_max` (precision exhausted) is what stops a badly conditioned input at 53 bits looping forever. It is not redundant with the unit-magnitude test.

* **Julia BigFloat arithmetic uses the global default precision, not the operands'.** Measured on 1.12.6: `arith=default`, `setindex=rebind`, `mul!=widened`. So a `Matrix{BigFloat}` has no precision as a matter of type — only of discipline.

* **Wrap whole computations in `setprecision(BigFloat, p) do ... end`, not just allocations.** `Matrix{BigFloat}(undef, ...)` pins nothing; it's the arithmetic that reads the default. Applies to `householder_block`, `apply_Qt!` and every BigFloat kernel — none of them establish their own block. Use the `do` form; the bare call doesn't restore.

* **`precision(A[1,1])` does not describe the matrix.** Entries can differ after a stray write. Use `uniform_precision` / `assert_precision` (in `precision.jl`) — the counterpart of flatter's `assert(R.prec() == r_col.prec() == tau.prec())`.

* **Widening costs speed, not accuracy.** Extra bits are noise on an already-correct value. But MPFR cost scales with precision, and the "precision exhausted" guard is calibrated against the precision you think you're at.

* **Convert BigFloat matrices explicitly across recursion boundaries** with `at_precision`. Returning one from a `setprecision` block is safe (precision travels with the object); doing further arithmetic on it outside a matching block is not.

* **`Float64(::BigInt)` overflows to `Inf` on lattice-sized entries.** Use `float64_matrix`, which pulls out a common power-of-two scale. Common, not per-column — per-column would distort the ratios the multipliers depend on.

* **Size-reduction bound isn't 1/2 on the orthogonal kernels.** flatter uses a 0.51 deadband (stops entries oscillating on rounding noise) and terminates on a unit-magnitude test, so expect ~0.51 plus float error. Only `Triangular` achieves a strict 1/2. The exactly testable invariant on every path is `B2_new == B2_old + B1*U` over `BigInt`.

* **Relative size reduction never reduces `B2`'s columns against each other.** When checking against `smsv_gso([B1 B2])`, assert only on `mu[j, n1+c]` for `j <= n1`; the rest is unconstrained.

* **`blocked` size reduction is not a reordering of the elementary one.** `diag_above` multiplies the *original* tile by `U(j,j)`, where the elementary sweep would use the already-reduced columns. Different valid reduced representative — agrees with `:zz` on only ~2/3 of inputs. Test the contract, not equality.

* **Relative size reduction's precision requirement is set by the block magnitude ratio, not just conditioning.** Coordinates are known only to `max|B2| * 2^-p`, so reducing to the deadband needs `p > log2(max|B2| / min|R_jj|)` with margin. The `max|B2|` is the magnitude *after* reduction — only the component of `B2` in `B1`'s span can shrink, so a large orthogonal component sets a floor that no number of refinement passes can lower. Under-provisioning isn't an error: the convergence guard stops, `U` stays exact and unimodular, and only reduction quality suffers.

* **`fused_qr_size_reduction.jl` ports only flatter's `Columnwise`.** `ColumnwiseDouble` (same algorithm on doubles, plus a global power-of-two exponent shift to stop R overflowing) is not ported — BigFloat's exponent range makes the shifting unnecessary, and the `Float64` path here throws rather than silently producing `Inf`. `LazyRefine` needs a driver supplying prereduced prefixes; `Iterated` and `SeysenRefine` are unreachable from flatter's own dispatcher.

* **The fused stagnation guard diverges from flatter.** flatter compares multiplier *magnitudes* against an absolute slack of `prec/2`, which is dimensionally odd: for large multipliers the slack vanishes (so it means "stopped shrinking") and for small ones `mu_max - prec/2` goes negative so it always fires. We compare exponents instead, matching `relative_size_reduction.jl`. Verified: ≤2 passes per column on random, wide-profile and scrambled bases.

* **The fused three-tier reduction test compares in `S`, not `double`.** flatter converts both operands out of MPFR and compares the quotient in `double` to save a division. We compare in the working float type. Differs only for entries within a double's rounding error of the threshold, where either answer is fine — a spurious reduction is a valid transvection, a missed one is caught next pass.

* **Fused QR returns `tau`, not compact-WY `T`.** Reflectors are generated one at a time, so `T` doesn't exist naturally; use `compact_wy_from_reflectors` to convert before handing the factorization to `apply_Qt!` or the relative size reduction kernels. Note flatter zeroes `tau[n-1]` at the end (its `Columnwise` discards Q anyway); we keep it, so the returned factorization is genuine and `Q'B = R` is testable.

* **Fused QR returns R packed, subdiagonal not cleared.** Matches `householder_block`'s convention: upper trapezoid is R, below-diagonal column `j` holds `v_j`'s tail. flatter clears the subdiagonal because its `Columnwise` asserts `tau` empty. Use `triu(R)` if you want R alone.

* **`U` from fused QR is unimodular but not triangular.** Columns are processed left to right, but each is reduced against *all* its predecessors, so `U` is unit upper triangular only when no reduction crosses a column boundary. Test `abs(det(U)) == 1` (Bareiss), not the triangular structure.

* **`Profile::get_drop` is not first-minus-last.** It is the spread *less the sum of clean upward gaps* — places where every entry to the right sits strictly above every entry to the left. Those gaps are exactly what block compression removes, so they don't count as un-reducedness. `get_spread` is the plain `max - min`. The two coincide on a decreasing profile (a reduced basis has no upward jump), which is why a first-minus-last stand-in passes most tests; they diverge on any profile that jumps up, where drop can be 0 while spread is large. Invariant: `0 <= drop <= spread`.

* **`goal_from_drop`'s proved branch drops the flag, but the branch is unreachable.** It calls `from_slope(n, slope)` instead of `from_slope(n, slope, proved)`, so it would return a *heuristic* goal — except all three call sites in flatter (`heuristic_1/2/3.cpp`) use the two-argument form, so `proved` is always false and the correct `else` branch runs. Reproduced verbatim anyway. Harmless as it stands; only matters if a future caller passes `proved = true`.

* **`params.proved` and `goal.proved` are independent flags.** The first (from `FLATTER_PROVED`) picks the *implementation* — `Proved1/2/3` over `Heuristic1/2/3`. The second picks the *acceptance test* in `check`. `params.cpp` always builds the initial goal with the two-argument `from_RHF`, so it is heuristic regardless; the only genuinely proved goal in flatter is constructed at `proved_1.cpp:100`, and `subgoal` propagates it from there. So reaching `Proved2`/`Proved3` without passing through `Proved1` gives proved implementations checked against a heuristic goal. Whether that is actually reachable depends on the top-level params constructor and CLI setup, neither of which I have seen — and the heuristic test is the *stricter* of the two, so the effect would be over-reduction, not a violated guarantee. Worth verifying before porting `Proved1/2/3`; irrelevant to the heuristic path.

* **Heuristic `goal_max_drop(g) / n` recovers the input slope exactly.** The `shape` function cancels completely between `from_slope` and `max_drop`. Useful invariant, and it means `quality` is a pure scale factor with no dimensional meaning of its own.

* **`goal_from_rhf` clamps below `2^(BKZ_BEST_SLOPE/2) ≈ 1.0109`.** Asking for a better root Hermite factor than BKZ achieves silently gives the same goal as asking for the clamp. Not a bug, but it means an over-ambitious `rhf` request is quietly ignored rather than rejected.

* **`with_best_slope` preserves the budget, it does not move it.** A heuristic goal's target slope splits as `best_slope + gap` with `gap = quality * shape / n`; changing the base slope re-attributes the same total between the two terms and leaves `goal_max_drop` exactly unchanged. Counterintuitive from the name — it sounds like it should make the goal stricter or looser. Its precondition is `slope < target slope`, not merely a small slope.

* **`goal_slope` returns different kinds of thing for the two goal families.** Proved: an actual slope. Heuristic: the raw `quality` scalar, which is *not* a slope. flatter's behaviour, preserved. For the slope a heuristic goal really targets, use `goal_max_drop(g) / g.n`.

* **`_goal_shape(1) == 0`**, so a one-dimensional goal has no budget to scale and `from_slope` would divide by zero. Guarded — dimension-one goals get `quality = 0` and are accepted unconditionally by `goal_check` anyway.

* **The heuristic `goal_check` has three conditions and all three do work.** Total drop, half-mean separation, and middle-span drop. Verified: a profile flat in both halves with a step between them at 95% of the budget is rejected despite its total drop fitting; and the middle-span condition alone rejects profiles the other two accept (696 of 200k random cases). Don't simplify it to the drop condition — that's the *proved* goal's test, and it would let the recursion settle for a basis still improvable across the half boundary.
