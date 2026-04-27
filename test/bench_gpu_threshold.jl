using AdaptiveCrossApproximation
using BEAST: DoubleNumSauterQstrat, Maxwell3D, raviartthomas, scalartype
using CUDA
using CompScienceMeshes
using StaticArrays: SVector
using H2Trees
using ParallelKMeans
using LinearMaps
using BlockSparseMatrices
using OhMyThreads
using Statistics
using CairoMakie
using LinearAlgebra

include("../example/hmatrix/skeletons.jl")
include("../example/hmatrix/hmatrix.jl")

const RESOLUTIONS   = [0.20881, 0.66030, 0.03]
const THRESHOLDS    = [1, 8, 32, 256, 2048]
const MINVALUES     = 5
const MAXRANK       = 40
const N_REPEATS     = 1        # repetitions per test
const FARQUADSTRAT  = DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1)
const NEARQUADSTRAT = DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6)
const SEPARATION    = SVector{3,Float64}(0.0, 0.0, 4.0)

# i hardcoded t-distribution critical values for 95% CI
# Index by degrees of freedom which is N_REPEATS - 1
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
    df = n - 1
    t  = get(T_CRIT, df, 1.96)   # fall back to z for large n
    return t * std(samples) / sqrt(n)
end

function build_problem(res)
    Γ1 = meshsphere(1.0, res)
    Γ2 = translate(meshsphere(1.0, res), SEPARATION)
    op = Maxwell3D.singlelayer(; wavenumber=2π / 0.5)
    X1 = raviartthomas(Γ1)
    X2 = raviartthomas(Γ2)
    return op, X1, X2
end

function build_tree(X1, X2)
    ttree1 = H2Trees.KMeansTree(X1.pos, 2; minvalues=MINVALUES)
    ttree2 = H2Trees.KMeansTree(X2.pos, 2; minvalues=MINVALUES)
    return BlockTree(ttree1, ttree2)
end

function time_assembly(op, X1, X2, tree; gpu, threshold)
    @elapsed HMatrix(
        op,
        X1,
        X2,
        tree;
        nearquadstrat        = NEARQUADSTRAT,
        farquadstrat         = FARQUADSTRAT,
        gpu                  = gpu,
        block_size_threshold = threshold,
        maxrank              = MAXRANK,
    )
end

struct BenchResult
    ndofs::Int
    threshold::Int
    t_cpu::Float64        # mean CPU time
    t_cpu_ci::Float64     # 95% CI half-width
    t_gpu::Float64        # mean GPU time
    t_gpu_ci::Float64     # 95% CI half-width
    speedup::Float64      # t_cpu / t_gpu
    speedup_ci::Float64   # propagated 95% CI half-width
end

function BenchResult(ndofs, threshold, cpu_samples, gpu_samples)
    μ_cpu  = mean(cpu_samples)
    ci_cpu = ci95(cpu_samples)
    μ_gpu  = mean(gpu_samples)
    ci_gpu = ci95(gpu_samples)
    sp     = μ_cpu / μ_gpu
    # Error propagation for ratio: σ_sp/sp = sqrt((σ_cpu/μ_cpu)² + (σ_gpu/μ_gpu)²)
    σ_cpu = ci_cpu / get(T_CRIT, length(cpu_samples) - 1, 1.96) * sqrt(length(cpu_samples))
    σ_gpu = ci_gpu / get(T_CRIT, length(gpu_samples) - 1, 1.96) * sqrt(length(gpu_samples))
    sp_ci =
        sp *
        sqrt((σ_cpu / μ_cpu)^2 + (σ_gpu / μ_gpu)^2) *
        get(T_CRIT, length(gpu_samples) - 1, 1.96) / sqrt(length(gpu_samples))
    return BenchResult(ndofs, threshold, μ_cpu, ci_cpu, μ_gpu, ci_gpu, sp, sp_ci)
end

function run_benchmarks()
    CUDA.functional() || error("CUDA not available")

    results = BenchResult[]

    for res in RESOLUTIONS
        op, X1, X2 = build_problem(res)
        ndofs = length(X1)
        tree = build_tree(X1, X2)

        println("$(ndofs) DOFs per sphere (res=$(res))")

        print("CPU ($(N_REPEATS)x)")
        cpu_samples = [
            time_assembly(op, X1, X2, tree; gpu=false, threshold=0) for _ in 1:N_REPEATS
        ]
        μ_cpu = mean(cpu_samples)
        println("$(round(μ_cpu; digits=2))s ± $(round(ci95(cpu_samples); digits=2))s")

        for thr in THRESHOLDS
            print("  GPU threshold=$(thr) ($(N_REPEATS)×) ... ")
            gpu_samples = [
                time_assembly(op, X1, X2, tree; gpu=true, threshold=thr) for
                _ in 1:N_REPEATS
            ]
            μ_gpu = mean(gpu_samples)
            speedup = μ_cpu / μ_gpu
            println(
                "$(round(μ_gpu; digits=2))s +- $(round(ci95(gpu_samples); digits=2))s  ($(round(speedup; digits=2))×)",
            )
            push!(results, BenchResult(ndofs, thr, cpu_samples, gpu_samples))
        end
    end

    return results
end

# ── Plot ──────────────────────────────────────────────────────────────────────

function plot_results(results; output="bench_gpu_threshold.png")
    ndofs_vals = sort(unique(getfield.(results, :ndofs)))
    thr_vals   = sort(unique(getfield.(results, :threshold)))
    n_groups   = length(ndofs_vals)
    n_bars     = length(thr_vals)

    rmap = Dict((r.ndofs, r.threshold) => r for r in results)

    fig        = Figure(; size=(1100, 580))
    ax_time    = Axis(fig[1, 1]; title="Far-field construction time", xlabel="DOFs per sphere", ylabel="Time (s)", xticks=(1:n_groups, string.(ndofs_vals)))
    ax_speedup = Axis(fig[1, 2]; title="GPU speedup vs CPU", xlabel="DOFs per sphere", ylabel="Speedup (×)", xticks=(1:n_groups, string.(ndofs_vals)))
    hlines!(ax_speedup, [1.0]; color=:black, linestyle=:dash, linewidth=1)

    palette   = Makie.wong_colors()[1:n_bars]
    bar_width = 0.8 / n_bars

    for (ti, thr) in enumerate(thr_vals)
        offset = (ti - (n_bars + 1) / 2) * bar_width
        xs     = [gi + offset for gi in 1:n_groups]

        t_means  = [get(rmap, (ndofs_vals[gi], thr), nothing) for gi in 1:n_groups]
        t_gpus   = [isnothing(r) ? NaN : r.t_gpu for r in t_means]
        t_cis    = [isnothing(r) ? NaN : r.t_gpu_ci for r in t_means]
        speedups = [isnothing(r) ? NaN : r.speedup for r in t_means]
        sp_cis   = [isnothing(r) ? NaN : r.speedup_ci for r in t_means]

        barplot!(
            ax_time,
            xs,
            t_gpus;
            width=bar_width * 0.9,
            color=palette[ti],
            label="threshold=$(thr)",
        )
        barplot!(ax_speedup, xs, speedups; width=bar_width * 0.9, color=palette[ti])

        errorbars!(
            ax_time, xs, t_gpus, t_cis; whiskerwidth=5, color=(:black, 0.7), linewidth=1.2
        )
        errorbars!(
            ax_speedup,
            xs,
            speedups,
            sp_cis;
            whiskerwidth=5,
            color=(:black, 0.7),
            linewidth=1.2,
        )
    end

    # CPU baseline: dotted line per group with its own CI band
    for (gi, ndofs) in enumerate(ndofs_vals)
        r = get(rmap, (ndofs, first(thr_vals)), nothing)
        isnothing(r) && continue
        lo = r.t_cpu - r.t_cpu_ci
        hi = r.t_cpu + r.t_cpu_ci
        xlo = (gi - 0.5) / n_groups
        xhi = (gi + 0.5) / n_groups
        hlines!(
            ax_time,
            [r.t_cpu];
            xmin=xlo,
            xmax=xhi,
            color=:black,
            linewidth=2,
            linestyle=:dot,
        )
        # shaded CI band for CPU
        poly!(
            ax_time,
            Point2f[
                (xlo * n_groups + 0.5, lo),
                (xhi * n_groups + 0.5, lo),
                (xhi * n_groups + 0.5, hi),
                (xlo * n_groups + 0.5, hi),
            ];
            color=(:black, 0.10),
            strokewidth=0,
        )
    end

    cpu_elem = LineElement(; color=:black, linestyle=:dot, linewidth=2)
    Legend(
        fig[2, :],
        [[PolyElement(; color=palette[ti]) for ti in 1:n_bars]..., cpu_elem],
        [["threshold=$(thr)" for thr in thr_vals]..., "CPU baseline"];
        orientation=:horizontal,
        tellwidth=false,
    )

    save(output, fig)
    println("\nSaved plot to $(output)")
    return fig
end

# ── Entry point ───────────────────────────────────────────────────────────────

results = run_benchmarks()
fig     = plot_results(results; output=joinpath(@__DIR__, "bench_gpu_threshold.png"))
