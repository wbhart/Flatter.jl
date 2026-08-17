module Flatter

using LinearAlgebra
using FPLLL

include("strassen.jl")
include("householder.jl")
include("schoenhage.jl")
include("smsv.jl")
include("size_reduction_triu.jl")
include("precision_utils.jl") # helpers to manage mpfr precision
include("relative_size_reduction.jl")
include("fused_qr_size_reduction.jl")
include("goal.jl") # determines if reduction goal has been met

end # module Flatter
