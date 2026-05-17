using AdaptiveCrossApproximation
using BEAST: Maxwell3D, raviartthomas, scalartype
using CUDA
using CompScienceMeshes
using StaticArrays: SVector
using Statistics
using CairoMakie
using LinearAlgebra

const GPU_KERNEL  = :pair_scatter   # assembly kernel — held fixed while batch size varies
const BATCH_SIZES = (1, 5)
const RESOLUTIONS = [0.5, 0.3, 0.2, 0.15]
const MAXRANK     = 40
const N_REPEATS   = 10
const TOL         = 1e-4
const SEPARATION  = SVector{3,Float64}(0.0, 0.0, 4.0)

const T_CRIT = Dict(
    1 => 12.706,
    2 => 4.303,
    3 => 3.182,
    4 => 2.776,
    5 => 2.571,
    6 => 2.447,
    7 => 2.365,
    8 => 2.306,
    9 => 2.262,
    10 => 2.228,
    20 => 2.086,
    30 => 2.042,
)

function ci95(samples)
    n = length(samples)
    n == 1 && return 0.0
    return get(T_CRIT, n - 1, 1.96) * std(samples) / sqrt(n)
end

function build_problem(res)
    Γ1 = meshsphere(1.0, res)
    Γ2 = translate(meshsphere(1.0, res), SEPARATION)
    op = Maxwell3D.singlelayer(; wavenumber=2π / 0.5)
    X1 = raviartthomas(Γ1)
    X2 = raviartthomas(Γ2)
    return op, X1, X2
end

# ── CPU block assembly + ACA (MaximumValue pivoting) ─────────────────────────

function run_block_cpu(op, X1, X2)
    aca       = ACA(AdaptiveCrossApproximation.MaximumValue(), AdaptiveCrossApproximation.MaximumValue(), AdaptiveCrossApproximation.FNormEstimator(TOL))
    K         = AdaptiveCrossApproximation.AbstractKernelMatrix(op, X1, X2)
    rowbuffer = zeros(scalartype(op), MAXRANK, length(X2))
    colbuffer = zeros(scalartype(op), length(X1), MAXRANK)
    test_ids  = collect(1:length(X1))
    trial_ids = collect(1:length(X2))
    return aca(K, colbuffer, rowbuffer, MAXRANK; rowidcs=test_ids, colidcs=trial_ids)
end

# ── GPU block assembly + ACA (BatchedFillDistance pivoting) ──────────────────

function run_block_gpu(assembler, X1, X2; batchsize::Int)
    ext       = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    test_ids  = collect(1:length(X1))
    trial_ids = collect(1:length(X2))
    rowpiv    = BatchedFillDistance(X1.pos; batchsize=batchsize)
    colpiv    = BatchedFillDistance(X2.pos; batchsize=batchsize)
    return ext.compress_block_gpu(
        assembler, test_ids, trial_ids; rowpivoting=rowpiv, columnpivoting=colpiv
    )
end

# ── Result type ───────────────────────────────────────────────────────────────

struct BenchResult
    ndofs::Int
    block_size::Tuple{Int,Int}
    batchsize::Int        # 0 → CPU baseline
    t_mean::Float64
    t_ci::Float64
    speedup::Float64      # vs CPU; 1.0 for the CPU entry
    speedup_ci::Float64
end

# ── Run benchmarks ────────────────────────────────────────────────────────────

function run_benchmarks()
    CUDA.functional() || error("CUDA not available")
    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    ext === nothing && error("ACACUDAExt not loaded")

    results = BenchResult[]

    for res in RESOLUTIONS
        op, X1, X2 = build_problem(res)
        m, n       = length(X1), length(X2)
        block_size = (m, n)

        println("\nBlock $(m)×$(n)  ($(m * n) entries, res=$(res), kernel=$(GPU_KERNEL))")

        # ── CPU baseline ──
        print("  CPU warmup ... ")
        run_block_cpu(op, X1, X2)
        println("done")

        print("  CPU ($(N_REPEATS)×) ... ")
        cpu_samples = [@elapsed run_block_cpu(op, X1, X2) for _ in 1:N_REPEATS]
        μ_cpu = mean(cpu_samples)
        ci_cpu = ci95(cpu_samples)
        println("$(round(μ_cpu; digits=3))s ± $(round(ci_cpu; digits=3))s")
        push!(results, BenchResult(m, block_size, 0, μ_cpu, ci_cpu, 1.0, 0.0))

        assembler = ext.GPUBlockAssembler(
            op, X1, X2; kernel=GPU_KERNEL, maxrank=MAXRANK, tol=TOL
        )

        for bs in BATCH_SIZES
            label = "GPU batchsize=$(bs)"

            print("  $(label) warmup ... ")
            run_block_gpu(assembler, X1, X2; batchsize=bs)
            CUDA.synchronize()
            println("done")

            print("  $(label) ($(N_REPEATS)×) ... ")
            gpu_samples = [
                begin
                    t = @elapsed run_block_gpu(assembler, X1, X2; batchsize=bs)
                    CUDA.synchronize()
                    t
                end for _ in 1:N_REPEATS
            ]
            μ_gpu = mean(gpu_samples)
            ci_gpu = ci95(gpu_samples)
            speedup = μ_cpu / μ_gpu
            rel_var = (ci_cpu / μ_cpu)^2 + (ci_gpu / μ_gpu)^2
            speedup_ci = speedup * sqrt(rel_var)

            println(
                "$(round(μ_gpu; digits=3))s ± $(round(ci_gpu; digits=3))s  " *
                "($(round(speedup; digits=2))× ± $(round(speedup_ci; digits=2)))",
            )
            push!(
                results, BenchResult(m, block_size, bs, μ_gpu, ci_gpu, speedup, speedup_ci)
            )
        end
    end

    return results
end

# ── Plot ──────────────────────────────────────────────────────────────────────

function plot_results(results; output="./figure/bench_gpu_threshold.png")
    gpu_results = filter(r -> r.batchsize > 0, results)
    ndofs_vals  = sort(unique(r.ndofs for r in gpu_results))
    n_groups    = length(ndofs_vals)
    n_bars      = length(BATCH_SIZES)
    bar_width   = 0.8 / n_bars

    rmap = Dict((r.ndofs, r.batchsize) => r for r in results)

    wong   = Makie.wong_colors()
    colors = Dict(bs => wong[mod1(i, length(wong))] for (i, bs) in enumerate(BATCH_SIZES))

    xtick_labels = [
        let r = rmap[(ndofs_vals[gi], BATCH_SIZES[1])]
            "$(ndofs_vals[gi])\n($(r.block_size[1])×$(r.block_size[2]))"
        end for gi in 1:n_groups
    ]

    fig = Figure(; size=(1000, 580))
    ax  = Axis(fig[1, 1]; title="BatchedFillDistance: batchsize 1 vs $(BATCH_SIZES[2]) — GPU speedup vs CPU\n" * "Maxwell 3D single-layer, GPU kernel=$(GPU_KERNEL), " * "tol=$(TOL), maxrank=$(MAXRANK)", xlabel="DOFs per sphere  (block rows × cols)", ylabel="Speedup vs CPU  (×)", xticks=(1:n_groups, xtick_labels), yticks=LinearTicks(6))

    hlines!(ax, [1.0]; color=(:black, 0.35), linestyle=:dash, linewidth=1.2)

    for (bi, bs) in enumerate(BATCH_SIZES)
        offset = (bi - (n_bars + 1) / 2) * bar_width
        xs     = [gi + offset for gi in 1:n_groups]

        rows  = [get(rmap, (ndofs_vals[gi], bs), nothing) for gi in 1:n_groups]
        svals = [isnothing(r) ? NaN : r.speedup for r in rows]
        scis  = [isnothing(r) ? NaN : r.speedup_ci for r in rows]

        barplot!(
            ax, xs, svals; width=bar_width * 0.9, color=colors[bs], label="batchsize=$(bs)"
        )
        errorbars!(ax, xs, svals, scis; whiskerwidth=5, color=(:black, 0.6), linewidth=1.2)
    end

    Legend(fig[2, :], ax; orientation=:horizontal, tellwidth=false, nbanks=1)

    save(output, fig)
    println("\nSaved plot to $(output)")
    return fig
end

# ── Entry point ───────────────────────────────────────────────────────────────

results = run_benchmarks()
fig     = plot_results(results; output=joinpath(@__DIR__, "bench_gpu_threshold.png"))
