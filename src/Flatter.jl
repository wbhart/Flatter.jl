module Flatter

using LinearAlgebra
using FPLLL

include("strassen.jl")
include("householder.jl")
include("schoenhage.jl")
include("smsv.jl")
include("size_reduction_triu.jl")
include("precision_utils.jl") # helpers to manage mpfr precision

end # module Flatter
