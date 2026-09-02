# Flatter.jl

A Julia implementation of the serial heuristic lattice-reduction pipeline from
[flatter](https://github.com/keeganryan/flatter).

The default `reduce_basis` dispatcher implements the same heuristic phases as
flatter: irregular-input handling, condition discovery for dense input,
Heuristic1 for a known condition bound, Heuristic2, and Heuristic3.  The exact
basis relation

    B_out == B_in * U

is maintained throughout, with `U` integral and unimodular.

An independent `algorithm = :teaching` reducer is retained for exposition and
cross-checking.  It is not part of the faithful heuristic dispatcher.

Threaded3 and the proved reduction pipeline are not implemented.  Blocked QR is
retained as an experimental building block, while the serial heuristic path uses
the direct MPFR-style Householder kernels selected by flatter.

The project is intended to be licensed under GNU LGPLv3.
