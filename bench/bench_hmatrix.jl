using AdaptiveCrossApproximation:
    AdaptiveCrossApproximation,
    PreserveSpaceOrder,
    SerialScheduler,
    HMatrix,
    ACA,
    FillDistance,
    FNormEstimator
using BEAST: Maxwell3D, raviartthomas, scalartype, Helmholtz3D, lagrangec0d1
using CUDA: @sync
using CompScienceMeshes
using StaticArrays: SVector
using Statistics
using LinearAlgebra
using BenchmarkTools: @benchmark
using Serialization
using H2Trees: TwoNTree

include("plot_stats.jl")

const GPU_KERNELS = (:pair_scatter, :gather_tile, :hybrid_global, :hybrid_shared, :sparse)
const RESOLUTIONS = (0.4, 0.25, 0.13, 0.07)
const TOL         = 1e-4
const MAXRANK     = 40
const SEPARATION  = 3.0

function build_system(h::AbstractFloat, k::AbstractFloat, operator::Symbol)
    Γ1 = CompScienceMeshes.translate(meshsphere(1.0, h), SVector(-SEPARATION / 2, 0.0, 0.0))
    Γ2 = CompScienceMeshes.translate(meshsphere(1.0, h), SVector(SEPARATION / 2, 0.0, 0.0))
    if operator == :helmholtz
        op = Helmholtz3D.singlelayer(; wavenumber=k)
        X1 = lagrangec0d1(Γ1)
        X2 = lagrangec0d1(Γ2)
    elseif operator == :maxwell
        op = Maxwell3D.singlelayer(; wavenumber=k)
        X1 = raviartthomas(Γ1)
        X2 = raviartthomas(Γ2)
    else
        error("'operator' must be :helmholtz or :maxwell")
    end
    return op, X1, X2, length(X1), length(X2)
end

"""
Build the CPU ACA compressor used as the baseline. FillDistance pivoting on
both axes mirrors the GPU compressor's BatchedFillDistance, so the only axis
that differs between CPU and GPU is the batching itself.
"""
cpu_aca(X1, X2) = ACA(FillDistance(X1.pos), FillDistance(X2.pos), FNormEstimator(TOL))

function dense_reference(operator, testspace, trialspace)
    T = scalartype(operator)
    m = length(testspace)
    n = length(trialspace)
    A = zeros(T, m, n)
    K = AdaptiveCrossApproximation.AbstractKernelMatrix(operator, testspace, trialspace)
    K(A, collect(1:m), collect(1:n))
    return A, norm(A)
end

relative_error(hmat, A_dense, A_norm) = norm(Matrix(hmat) - A_dense) / A_norm

"""
Per-block ranks across all far-field levels of an HMatrix. Each far block is a
LowRankMatrix `U*V` whose rank is `size(U, 2)` (the pivot count chosen by the
compressor for that block).
"""
function block_ranks(hmat)
    rs = Int[]
    for bsm in hmat.farinteractions
        for blk in bsm.blocks
            push!(rs, size(blk.U, 2))
        end
    end
    return rs
end

function rank_stats(hmat)
    rs = block_ranks(hmat)
    isempty(rs) && return (mean_rank=0.0, max_rank=0)
    return (mean_rank=Float64(mean(rs)), max_rank=maximum(rs))
end

"""
Return the cluster (line key) for a record under the given sweep `category`.
"""
function cluster_for(category, kernel, batchsize, k, svd_comp, fnorm_iter, block_size)
    category === :kernel && return kernel
    category === :batchsize && return batchsize
    category === :wavenumber && return k
    category === :svd_compression && return svd_comp
    category === :fnorm_iteration && return fnorm_iter
    category === :block_size && return block_size
    return :default
end

function main(; output=joinpath(@__DIR__, "bench_hmatrix_results.jld"))
    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDA)

    results = []
    print("How many repeats should be run per benchmark: ")
    n_repeats = parse(Int64, readline())
    print(
        "Which category? (kernel, batchsize, svd_compression, fnorm_iteration, wavenumber, block_size): ",
    )
    category = Symbol(strip(readline()))

    Ks               = category === :wavenumber ? collect(1.0:2.0:11.0) : (5.0,)
    kernels_sweep    = category === :kernel ? GPU_KERNELS : (:hybrid_global,)
    batches_sweep    = category === :batchsize ? (1, 2, 4, 8, 16, 17, 32) : (8,)
    svd_sweep        = category === :svd_compression ? (true, false) : (false,)
    fnorm_sweep      = category === :fnorm_iteration ? (true, false) : (false,)
    block_size_sweep = category === :block_size ? (32, 64, 128, 256, 512) : (128,)

    for h in RESOLUTIONS
        for op_sym in (:helmholtz, :maxwell)
            for k in Ks
                operator, X1, X2, m, ncols = build_system(h, k, op_sym)
                ndofs = m * ncols

                @info "Building dense reference" h op_sym k m ncols ndofs
                A_dense, A_norm = dense_reference(operator, X1, X2)

                for block_size in block_size_sweep
                tree = TwoNTree(
                    X1.pos, X2.pos, 1 / 2^10;
                    testminvalues=block_size, trialminvalues=block_size,
                )

                # ─── CPU baseline (FillDistance pivoting) ───
                cpu_compressor = cpu_aca(X1, X2)
                hmat_cpu = HMatrix(
                    operator,
                    X1,
                    X2,
                    tree;
                    tol=TOL,
                    maxrank=MAXRANK,
                    compressor=cpu_compressor,
                    spaceordering=PreserveSpaceOrder(),
                    scheduler=SerialScheduler(),
                )
                cpu_err = relative_error(hmat_cpu, A_dense, A_norm)
                cpu_rstats = rank_stats(hmat_cpu)
                hmat_cpu = nothing

                cpu_bench = @benchmark HMatrix(
                    $operator,
                    $X1,
                    $X2,
                    $tree;
                    tol=$TOL,
                    maxrank=$MAXRANK,
                    compressor=$cpu_compressor,
                    spaceordering=PreserveSpaceOrder(),
                    scheduler=SerialScheduler(),
                ) samples = n_repeats evals = 1
                push!(
                    results,
                    (
                        device          = :cpu,
                        times           = cpu_bench.times,    # nanoseconds
                        err             = cpu_err,
                        n               = ndofs,
                        h               = h,
                        op              = op_sym,
                        k               = k,
                        kernel          = :none,
                        batchsize       = 0,
                        svd_compression = false,
                        fnorm_iteration = false,
                        block_size      = block_size,
                        mean_rank       = cpu_rstats.mean_rank,
                        max_rank        = cpu_rstats.max_rank,
                        category        = category,
                        cluster         = category === :block_size ? block_size : :cpu,
                    ),
                )

                # ─── GPU variants ───
                for kernel in kernels_sweep,
                    batch_size in batches_sweep,
                    svd_compression in svd_sweep,
                    fnormiter in fnorm_sweep

                    gpu_comp = ext.GPUCompressor(
                        operator,
                        X1,
                        X2;
                        tol=TOL,
                        maxrank=MAXRANK,
                        batchsize=batch_size,
                        kernel=kernel,
                        svd_compression=svd_compression,
                        fnorm_iteration=fnormiter,
                    )

                    hmat_gpu = @sync HMatrix(
                        operator,
                        X1,
                        X2,
                        tree;
                        tol=TOL,
                        maxrank=MAXRANK,
                        compressor=gpu_comp,
                        spaceordering=PreserveSpaceOrder(),
                        scheduler=SerialScheduler(),
                    )
                    gpu_err = relative_error(hmat_gpu, A_dense, A_norm)
                    gpu_rstats = rank_stats(hmat_gpu)
                    hmat_gpu = nothing
                    gpu_bench = @benchmark HMatrix(
                        $operator,
                        $X1,
                        $X2,
                        $tree;
                        tol=$TOL,
                        maxrank=$MAXRANK,
                        compressor=$gpu_comp,
                        spaceordering=PreserveSpaceOrder(),
                        scheduler=SerialScheduler(),
                    ) samples = n_repeats evals = 1
                    push!(
                        results,
                        (
                            device          = :gpu,
                            times           = gpu_bench.times,
                            err             = gpu_err,
                            n               = ndofs,
                            h               = h,
                            op              = op_sym,
                            k               = k,
                            kernel          = kernel,
                            batchsize       = batch_size,
                            svd_compression = svd_compression,
                            fnorm_iteration = fnormiter,
                            block_size      = block_size,
                            mean_rank       = gpu_rstats.mean_rank,
                            max_rank        = gpu_rstats.max_rank,
                            category        = category,
                            cluster         = cluster_for(category, kernel, batch_size, k, svd_compression, fnormiter, block_size),
                        ),
                    )
                end
                end  # block_size loop
            end
        end
    end

    serialize(output, results)
    @info "Results saved" output
    plot_bench(results)
    return results
end

results = main()
