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
