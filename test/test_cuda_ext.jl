using Test
using CUDA
using LinearAlgebra
using BEAST:
    IntegralOperator,
    Space,
    lagrangec0d1,
    Helmholtz3D,
    Maxwell3D,
    raviartthomas,
    DoubleNumSauterQstrat,
    scalartype
using CompScienceMeshes: meshsphere, meshcuboid
using H2Trees
using ParallelKMeans
using LinearMaps
using BlockSparseMatrices
using Statistics

include("../example/hmatrix/skeletons.jl")
include("../example/hmatrix/hmatrix.jl")
include("../example/hmatrix/calculate_error.jl")

struct Problem
    op::IntegralOperator
    X::Space
    resolution::Float64
end

@testset "ACA CUDA Extension" begin
    if !CUDA.functional()
        @warn "CUDA is not functional on this machine; skipping CUDA extension tests."
        return nothing
    end

    ResolutionList = (0.5, 0.2)
    ProblemList::Vector{Problem} = []
    for res in ResolutionList

        # I try to limit testing to domains that are interesting in terms of correctness/runtime differences
        # 1. Low wavenumber on smooth surface with piecewise constant basis
        # 2. High wavenumber on smooth surface with piecewise constant basis
        # 3. High wavenumber on surface with discontinuities
        # 4. Maxwell operator on smooth surface
        append!(
            ProblemList,
            [
                Problem(
                    Helmholtz3D.singlelayer(; wavenumber=0.1),
                    lagrangec0d1(meshsphere(1.0, res)),
                    res,
                ),
                Problem(
                    Helmholtz3D.singlelayer(; wavenumber=10.0),
                    lagrangec0d1(meshsphere(1.0, res)),
                    res,
                ),
                Problem(
                    Helmholtz3D.singlelayer(; wavenumber=10.0),
                    lagrangec0d1(meshcuboid(1.0, 1.0, 1.0, res)),
                    res,
                ),
                Problem(
                    Maxwell3D.singlelayer(; wavenumber=5.0),
                    raviartthomas(meshsphere(1.0, res)),
                    res,
                ),
            ],
        )
    end
    for problem in ProblemList
        @testset "CUDA-accelerated ACA on operator $(typeof(problem.op)), with basis $(typeof(problem.X)) and edge length $(problem.resolution)" begin

            println("starting:CUDA-accelerated ACA on operator $(typeof(problem.op)), with basis $(typeof(problem.X)) and edge length $(problem.resolution)")

            ttree = H2Trees.KMeansTree(problem.X.pos, 2; minvalues=5)
            tree = BlockTree(ttree, ttree)

            println("CPU Matrix construction:")
            t_cpu = @elapsed hmat = HMatrix(
                problem.op,
                problem.X,
                problem.X,
                tree;
                nearquadstrat=DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6),
                farquadstrat=DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1),
                gpu=false,
            )

            println("GPU Matrix construction:")

            CUDA.@profile HMatrix(
                            problem.op,
                            problem.X,
                            problem.X,
                            tree;
                            nearquadstrat=DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6),
                            farquadstrat=DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1),
                            gpu=true,
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
            per_block_errors::Vector = []
            for (level_cpu, level_gpu) in
                zip(hmat.farinteractions, hmat_gpu.farinteractions)
                for (tg_cpu, tg_gpu) in zip(level_cpu, level_gpu)
                    for (blk_cpu, blk_gpu) in zip(tg_cpu, tg_gpu)
                       push!(
                            per_block_errors, estimate_reldifference(blk_cpu.M, blk_gpu.M)
                        )
                    end
                end
            end
            mean_error = mean(per_block_errors)
            max_error = maximum(per_block_errors)
            println("mean_error = $mean_error, max_error = $max_error")
            @test mean_error ≈ 1.0 atol=1e-4
            @test max_error ≈ 1.0 atol=1e-4
        end
    end
end
