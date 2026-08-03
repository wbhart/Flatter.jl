module Flatter

using LinearAlgebra
using FPLLL

include("strassen.jl")
include("householder.jl")
include("schoenhage.jl")
include("smsv.jl")

end # module Flatter
