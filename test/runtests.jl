using Test
using Flatter

@testset "Flatter" begin
    include("test_strassen.jl")
    include("test_householder.jl")
    include("test_schoenhage.jl")
    include("test_smsv.jl")
    include("test_size_reduction_triu.jl")
end
