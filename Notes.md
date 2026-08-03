* In smsv.jl (the implementation of the Saruchi-Morel-Stehle-Villard Algorithm 2 [1]), blocked
  Householder QR is set as the default, as opposed to unblocked (see profile(...; blocked=true)).
  This attempts to give n^\omega complexity with \omega < 3 (using Strassen) for the Householder
  QR. But a more conservative choice would be non-blocked implementation.
