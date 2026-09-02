module Flatter

using LinearAlgebra
using Random
using FPLLL

include("strassen.jl")
include("householder.jl")
include("schoenhage.jl")
include("smsv.jl")
include("size_reduction_triu.jl")
include("precision.jl")
include("relative_size_reduction.jl")
include("fused_qr_size_reduction.jl")
include("goal.jl")
include("sublattice_split.jl")
include("recursive_reduction.jl")
include("irregular.jl")
include("heuristic3.jl")
include("heuristic2.jl")
include("heuristic1.jl")
include("cond_unknown.jl")

include("lattices/generators.jl")

end # module Flatter
