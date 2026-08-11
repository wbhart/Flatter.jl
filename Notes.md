* In smsv.jl (the implementation of the Saruchi-Morel-Stehle-Villard Algorithm 2 [1]), blocked
  Householder QR is set as the default, as opposed to unblocked (see profile(...; blocked=true)).
  This attempts to give n^\omega complexity with \omega < 3 (using Strassen) for the Householder
  QR. But a more conservative choice would be non-blocked implementation.

* In size_reduction_triu.jl, the blocked algorithm almost certainly doesn't benefit from Strassen
  at the given block size. The block size should be made larger, to benefit. There are other
  benefits to blocking though: some small cache effects and the possibility to use packing into
  large integers and unpacking, to do multiple operations in the matmul at once (not currently
  implemented -- and not implemented in the C++ code either, as far as I know -- probably because
  it is likely only useful for matrices with truly large BigInt entries). There's also a
  benefit in grabbing a workspace up front and reusing.
