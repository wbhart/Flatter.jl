using Test
using Flatter

@testset "Flatter" begin
    include("test_strassen.jl")
    include("test_householder.jl")
    include("test_schoenhage.jl")
    include("test_smsv.jl")
    include("test_size_reduction_triu.jl")
    include("test_relative_size_reduction.jl")
    include("test_fused_qr_size_reduction.jl")
    include("test_goal.jl")
    include("test_sublattice_split.jl")
    include("test_recursive_reduction.jl")
    include("test_irregular.jl")
    include("test_heuristic3.jl")
    include("test_heuristic2.jl")
    include("test_cond_unknown.jl")
end
