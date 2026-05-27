using AdaptiveCrossApproximation:
    AdaptiveCrossApproximation, PreserveSpaceOrder, SerialScheduler, HMatrix, ACA
using BEAST: Maxwell3D, raviartthomas, Helmholtz3D, lagrangec0d1
using CUDA
using CompScienceMeshes
using StaticArrays: SVector
using H2Trees: TwoNTree
using NVTX

const RES        = 0.5
const WAVENUMBER = 5.0
const OPERATOR   = :maxwell        # :helmholtz or :maxwell
const KERNEL     = :hybrid_global
const BATCHSIZE  = 8
const TOL        = 1e-4
const MAXRANK    = 40
const SEPARATION = 3.0

function build_problem()
    Γ1 = CompScienceMeshes.translate(
        meshsphere(1.0, RES), SVector(-SEPARATION / 2, 0.0, 0.0)
    )
    Γ2 = CompScienceMeshes.translate(
        meshsphere(1.0, RES), SVector(SEPARATION / 2, 0.0, 0.0)
    )
    if OPERATOR == :helmholtz
        op = Helmholtz3D.singlelayer(; wavenumber=WAVENUMBER)
        X1 = lagrangec0d1(Γ1)
        X2 = lagrangec0d1(Γ2)
    else
        op = Maxwell3D.singlelayer(; wavenumber=WAVENUMBER)
        X1 = raviartthomas(Γ1)
        X2 = raviartthomas(Γ2)
    end
    tree = TwoNTree(X1.pos, X2.pos, 1 / 2^10; testminvalues=128, trialminvalues=128)
    return op, X1, X2, tree
end

function build_gpu(op, X1, X2, tree)
    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    gpu_comp = ext.GPUCompressor(
        op, X1, X2; tol=TOL, maxrank=MAXRANK, batchsize=BATCHSIZE, kernel=KERNEL
    )
    return HMatrix(
        op,
        X1,
        X2,
        tree;
        tol=TOL,
        maxrank=MAXRANK,
        compressor=gpu_comp,
        spaceordering=PreserveSpaceOrder(),
        scheduler=SerialScheduler(),
    )
end

function main()
    CUDA.functional() || error("CUDA not available")

    @info "Building problem" RES OPERATOR WAVENUMBER KERNEL BATCHSIZE
    op, X1, X2, tree = build_problem()
    @info "Problem built" ndofs = length(X1)

    # Warm-up: triggers JIT compilation, kernel caching, allocator priming.
    NVTX.@range "warmup" begin
        @info "Warmup"
        @time build_gpu(op, X1, X2, tree)
        CUDA.synchronize()
    end

    # Profiled region — this is what nsys / ncu / CUDA.@profile should focus on.
    @info "Profiled run"
    CUDA.@profile begin
        NVTX.@range "HMatrix_GPU" begin
            @time hmat = build_gpu(op, X1, X2, tree)
            CUDA.synchronize()
        end
    end

    # Also report a CPU baseline for the same problem.
    @info "CPU baseline (FillDistance)"
    cpu_compressor = ACA(
        AdaptiveCrossApproximation.FillDistance(X1.pos),
        AdaptiveCrossApproximation.FillDistance(X2.pos),
        AdaptiveCrossApproximation.FNormEstimator(TOL),
    )
    @time HMatrix(
        op,
        X1,
        X2,
        tree;
        tol=TOL,
        maxrank=MAXRANK,
        compressor=cpu_compressor,
        spaceordering=PreserveSpaceOrder(),
        scheduler=SerialScheduler(),
    )

    return nothing
end

main()
