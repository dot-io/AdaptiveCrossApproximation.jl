using Test
using CUDA
using LinearAlgebra
using StaticArrays: SVector
using BEAST:
    IntegralOperator, Space, lagrangec0d1, Helmholtz3D, Maxwell3D, raviartthomas, scalartype
using CompScienceMeshes: meshsphere, meshcuboid, translate
using Statistics
using AdaptiveCrossApproximation

# Valid GPU kernels from BEAST's CUDA extension
const GPU_KERNELS = (:gather_tile, :pair_scatter, :sparse, :hybrid_global, :hybrid_shared)

# Separation distance between source and target geometries — chosen so the entire
# operator block is a single far-field admissible pair, which is what
# `compress_block_gpu` is designed for.
const FARFIELD_OFFSET = SVector(4.0, 0.0, 0.0)

struct Problem
    op::IntegralOperator
    X::Space
    Y::Space
    resolution::Float64
end

@testset "ACA CUDA Extension" begin
    if !CUDA.functional()
        @warn "CUDA is not functional on this machine; skipping CUDA extension tests."
        return nothing
    end

    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    if ext === nothing
        @warn "ACACUDAExt not loaded; skipping CUDA extension tests."
        return nothing
    end

    ResolutionList = (0.5, 0.3)

    ProblemList::Vector{Problem} = []
    for res in ResolutionList
        # Build well-separated (test, trial) pairs so the full operator block is
        # a single admissible far-field interaction and ACA can converge.
        sphere = meshsphere(1.0, res)
        sphere_far = translate(sphere, FARFIELD_OFFSET)
        cuboid = meshcuboid(1.0, 1.0, 1.0, res)
        cuboid_far = translate(cuboid, FARFIELD_OFFSET)

        # Coverage targets:
        # 1 low wavenumber, smooth mesh, linear lagrange bfs 
        # 2 high wavenumber, smooth mesh, linear lagrange bfs 
        # 3 high wavenumber, discontinuous mesh, linear lagrange bfs
        # 4 Maxwell/Raviart-Thomas on smooth mesh 
        append!(
            ProblemList,
            [
                Problem(
                    Helmholtz3D.singlelayer(; wavenumber=0.1),
                    lagrangec0d1(sphere),
                    lagrangec0d1(sphere_far),
                    res,
                ),
                Problem(
                    Helmholtz3D.singlelayer(; wavenumber=10.0),
                    lagrangec0d1(sphere),
                    lagrangec0d1(sphere_far),
                    res,
                ),
                Problem(
                    Helmholtz3D.singlelayer(; wavenumber=10.0),
                    lagrangec0d1(cuboid),
                    lagrangec0d1(cuboid_far),
                    res,
                ),
                Problem(
                    Maxwell3D.singlelayer(; wavenumber=5.0),
                    raviartthomas(sphere),
                    raviartthomas(sphere_far),
                    res,
                ),
            ],
        )
    end

    for problem in ProblemList
        @testset "CUDA-accelerated ACA on operator $(typeof(problem.op)), with basis $(typeof(problem.X)) and edge length $(problem.resolution)" begin
            println(
                "starting: CUDA-accelerated ACA on operator $(typeof(problem.op)), with basis $(typeof(problem.X)) and edge length $(problem.resolution)",
            )

            # make ground truth dense matrix
            T = scalartype(problem.op)
            maxrank = 40
            test_ids = collect(1:length(problem.X))
            trial_ids = collect(1:length(problem.Y))
            m = length(problem.X)
            n = length(problem.Y)

            # Assemble the full dense block via the kernel matrix interface
            K = AdaptiveCrossApproximation.AbstractKernelMatrix(
                problem.op, problem.X, problem.Y
            )
            A_dense = zeros(T, m, n)
            for i in 1:m, j in 1:n
                K(view(A_dense, i:i, j:j), [i], [j])
            end
            A_norm = norm(A_dense)

            #cpu
            aca = ACA(
                AdaptiveCrossApproximation.MaximumValue(),
                AdaptiveCrossApproximation.MaximumValue(),
                AdaptiveCrossApproximation.FNormEstimator(1e-4),
            )
            rowbuffer_cpu = zeros(T, maxrank, n)
            colbuffer_cpu = zeros(T, m, maxrank)

            npivots_cpu = aca(
                K,
                colbuffer_cpu,
                rowbuffer_cpu,
                maxrank;
                rowidcs=test_ids,
                colidcs=trial_ids,
            )
            U_cpu = colbuffer_cpu[1:m, 1:npivots_cpu]
            V_cpu = rowbuffer_cpu[1:npivots_cpu, 1:n]

            cpu_err = if A_norm > 0
                norm(A_dense - U_cpu * V_cpu) / A_norm
            else
                norm(A_dense - U_cpu * V_cpu)
            end
            println("  CPU: npivots=$(npivots_cpu), rel_err=$(round(cpu_err; digits=6))")
            @test cpu_err < 1e-2

            #gpu with different impls
            for kernel in GPU_KERNELS
                @testset "kernel=$(kernel)" begin
                    assembler = ext.GPUBlockAssembler(
                        problem.op,
                        problem.X,
                        problem.Y;
                        kernel  = kernel,
                        maxrank = maxrank,
                        tol     = 1e-4,
                    )

                    rowpivoting = BatchedFillDistance(problem.X.pos; batchsize=4)
                    columnpivoting = BatchedFillDistance(problem.Y.pos; batchsize=4)

                    npivots_gpu, U_gpu, V_gpu = ext.compress_block_gpu(
                        assembler,
                        test_ids,
                        trial_ids;
                        rowpivoting=rowpivoting,
                        columnpivoting=columnpivoting,
                    )

                    # Pull factors back to host for verification
                    U_host = Array(U_gpu)
                    V_host = Array(V_gpu)

                    gpu_err = if A_norm > 0
                        norm(A_dense - U_host * V_host) / A_norm
                    else
                        norm(A_dense - U_host * V_host)
                    end

                    println(
                        "  kernel=$(kernel): npivots=$(npivots_gpu), rel_err=$(round(gpu_err; digits=6))",
                    )
                    @test gpu_err < 1e-2
                end
            end
        end
    end
end
