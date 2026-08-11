module Flatter

using LinearAlgebra
using FPLLL

include("strassen.jl")
include("householder.jl")
include("schoenhage.jl")
include("smsv.jl")
include("size_reduction_triu.jl")

end # module Flatter
