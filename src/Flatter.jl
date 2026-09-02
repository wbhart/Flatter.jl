module Flatter

using LinearAlgebra
using Random
using FPLLL

include("strassen.jl")
include("householder.jl")
include("schoenhage.jl")
include("smsv.jl")
include("size_reduction_triu.jl")
include("precision.jl") # manage mpfr precision
include("relative_size_reduction.jl")
include("fused_qr_size_reduction.jl")
include("goal.jl") # determines if reduction goal has been met
include("sublattice_split.jl")
include("recursive_reduction.jl") # very basic flatter recursion
include("irregular.jl") # support "irregular" lattices
include("heuristic3.jl")
include("heuristic2.jl")
include("heuristic1.jl")
include("cond_unknown.jl")

include("lattices/generators.jl") # generate lattices of different kinds for benchmarking

end # module Flatter
