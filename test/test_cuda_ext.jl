using Test
using CUDA
using LinearAlgebra
using BEAST: IntegralOperator, Space, lagrangec0d1, Helmholtz3D, Maxwell3D, raviartthomas, DoubleNumSauterQstrat, scalartype
using CompScienceMeshes: meshsphere, meshrectangle
using H2Trees
using ParallelKMeans
using LinearMaps
using BlockSparseMatrices

include("../example/hmatrix/skeletons.jl")
include("../example/hmatrix/hmatrix.jl")
include("../example/hmatrix/calculate_error.jl")

struct Problem
    op::IntegralOperator
    X::Space
end

@testset "ACA CUDA Extension" begin
    if !CUDA.functional()
        @warn "CUDA is not functional on this machine; skipping CUDA extension tests."
        return nothing
    end

    ResolutionList = (1.0, 0.5)
    ProblemList::Vector{Problem} = []
    for res in ResolutionList

    # I try to limit testing to domains that are interesting in terms of correctness/runtime
    # 1. Low wavenumber on smooth surface with piecewise constant basis
    # 2. High wavenumber on smooth surface with piecewise constant basis
    # 3. High wavenumber on surface with discontinuities
    append!(ProblemList, [
        Problem(Helmholtz3D.singlelayer(; wavenumber=0.1), lagrangec0d1(meshsphere(1.0, res))),
        Problem(Helmholtz3D.singlelayer(; wavenumber=10.0), lagrangec0d1(meshsphere(1.0, res))),
        Problem(Helmholtz3D.singlelayer(; wavenumber=10.0), lagrangec0d1(meshrectangle(1.0, 1.0, res))),
        Problem(Maxwell3D.singlelayer(; wavenumber=5.0), raviartthomas(meshsphere(1.0, res)))
        ])
    end
    for problem in ProblemList
        @testset "CUDA-accelerated ACA on operator $(typeof(problem.op)), with basis $(typeof(problem.X))" begin

            ttree = H2Trees.KMeansTree(problem.X.pos, 2; minvalues=5)
            tree = BlockTree(ttree, ttree)

            t_cpu = @elapsed hmat = HMatrix(
                problem.op,
                problem.X,
                problem.X,
                tree;
                nearquadstrat=DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6),
                farquadstrat=DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1),
                gpu=false,
            )
            t_gpu = @elapsed hmat_gpu = HMatrix(
                problem.op,
                problem.X,
                problem.X,
                tree;
                nearquadstrat=DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6),
                farquadstrat=DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1),
                gpu=true,
            )
            @test calculate_error(hmat, hmat_gpu) ≈ 0.0 atol=1e-12
        end
    end
end
