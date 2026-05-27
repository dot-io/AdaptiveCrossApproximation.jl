using Serialization
using Statistics
using CairoMakie

const DATA_FILE     = joinpath(@__DIR__, "bench_hmatrix_results.jld")
const OUTPUT_DIR    = @__DIR__
const TOL_REFERENCE = 1e-4
const ALPHA         = 0.05    # two-sided 95 % CI

# Student-t critical values, keyed by (df, two-sided α).
const T_CRIT = Dict(
    (1, 0.2) => 3.078,
    (1, 0.1) => 6.314,
    (1, 0.05) => 12.71,
    (1, 0.025) => 25.45,
    (1, 0.01) => 63.66,
    (1, 0.005) => 127.3,
    (1, 0.001) => 636.6,
    (2, 0.2) => 1.886,
    (2, 0.1) => 2.920,
    (2, 0.05) => 4.303,
    (2, 0.025) => 6.205,
    (2, 0.01) => 9.925,
    (2, 0.005) => 14.09,
    (2, 0.001) => 31.599,
    (3, 0.2) => 1.638,
    (3, 0.1) => 1.886,
    (3, 0.05) => 3.182,
    (3, 0.025) => 4.541,
    (3, 0.01) => 5.841,
    (3, 0.005) => 7.453,
    (3, 0.001) => 12.92,
    (4, 0.2) => 1.533,
    (4, 0.1) => 1.533,
    (4, 0.05) => 2.776,
    (4, 0.025) => 3.747,
    (4, 0.01) => 4.604,
    (4, 0.005) => 5.598,
    (4, 0.001) => 8.610,
    (5, 0.2) => 1.476,
    (5, 0.1) => 1.476,
    (5, 0.05) => 2.571,
    (5, 0.025) => 3.365,
    (5, 0.01) => 4.032,
    (5, 0.005) => 4.773,
    (5, 0.001) => 6.869,
    (10, 0.2) => 1.372,
    (10, 0.1) => 1.372,
    (10, 0.05) => 2.228,
    (10, 0.025) => 2.764,
    (10, 0.01) => 3.169,
    (10, 0.005) => 3.581,
    (10, 0.001) => 4.587,
    (20, 0.2) => 1.325,
    (20, 0.1) => 1.325,
    (20, 0.05) => 2.086,
    (20, 0.025) => 2.528,
    (20, 0.01) => 2.845,
    (20, 0.005) => 3.174,
    (20, 0.001) => 4.043,
)
"""
    div_rv_metrics(num, den)

Delta-method estimator for mean and variance of R = N/D. Returns `(μR, σR², n)`.
"""
function div_rv_metrics(num, den)
    nN, nD = length(num), length(den)
    n = min(nN, nD)
    μN, μD = mean(num), mean(den)
    σN², σD² = var(num), var(den)
    paired = (nN == nD) && n > 1 && σN² > 0 && σD² > 0
    ρ = paired ? cor(num, den) : 0.0
    μR = μN / μD
    σR² = μR^2 * (σN² / μN^2 + σD² / μD^2 - 2 * ρ * sqrt(σN² * σD²) / (μN * μD))
    return μR, max(σR², 0.0), n
end

"""
Two-sided Student-t CI half-width interval for the sample mean.
"""
function ci_t(μ, σ², n; alpha::Float64=ALPHA)
    n < 2 && return μ, μ
    df = n - 1
    crit = get(T_CRIT, (df, alpha), 1.96)
    se = sqrt(σ² / n)
    return μ - crit * se, μ + crit * se
end

cluster_label(c) = string(c)
cluster_label(c::AbstractFloat) = string(round(c; digits=3))

"""
Find the CPU record sharing `(h, op, k)` with `grec`.
"""
function match_cpu(cpu_recs, grec)
    grec_blk = hasproperty(grec, :block_size) ? grec.block_size : nothing
    for c in cpu_recs
        c.h == grec.h && c.op == grec.op && c.k == grec.k || continue
        cblk = hasproperty(c, :block_size) ? c.block_size : nothing
        cblk == grec_blk && return c
    end
    return nothing
end

function sort_clusters(cs)
    v = collect(cs)
    try
        return sort(v)
    catch
        return sort(v; by=string)
    end
end
"""
    plot_bench(data; output_dir, fname)

Build a 2-row × N-operator figure showing speedup (top) and relative error
(bottom) vs DoFs. One line per `cluster` (from the bench category sweep).
"""
function plot_bench(data; output_dir=OUTPUT_DIR, fname="bench.png")
    cpu_recs = [d for d in data if d.device === :cpu]
    gpu_recs = [d for d in data if d.device === :gpu]
    if isempty(gpu_recs)
        @warn "No GPU records — nothing to plot."
        return nothing
    end

    operators = sort(unique(d.op for d in data))
    clusters  = sort_clusters(unique(d.cluster for d in gpu_recs))
    category  = first(gpu_recs).category

    palette = Makie.wong_colors()
    cluster_colors = Dict(
        c => palette[mod1(i, length(palette))] for (i, c) in enumerate(clusters)
    )

    n_ops = length(operators)
    fig = Figure(; size=(560 * n_ops, 1400))

    for (col, op) in enumerate(operators)
        ax_sp = Axis(
            fig[1, col];
            title  = string(op),
            xlabel = "system matrix entries (m·n)",
            ylabel = col == 1 ? "Speedup  T_cpu / T_gpu" : "",
            xscale = log10,
        )
        ax_er = Axis(
            fig[2, col];
            xlabel = "system matrix entries (m·n)",
            ylabel = col == 1 ? "rel. error  ‖H − A‖ / ‖A‖" : "",
            xscale = log10,
            yscale = log10,
        )
        ax_rk = Axis(
            fig[3, col];
            xlabel = "system matrix entries (m·n)",
            ylabel = col == 1 ? "block rank  (solid: max, dashed: mean)" : "",
            xscale = log10,
        )

        hlines!(ax_sp, [1.0]; color=:gray, linestyle=:dash, linewidth=1)
        hlines!(ax_er, [TOL_REFERENCE]; color=:black, linestyle=:dot, linewidth=1)

        for cluster in clusters
            grecs = sort(
                filter(d -> d.op == op && d.cluster == cluster, gpu_recs); by=d -> d.n
            )
            isempty(grecs) && continue
            color = cluster_colors[cluster]
            label = cluster_label(cluster)

            # ── Speedup line + CI band ──
            xs, μs, los, his = Int[], Float64[], Float64[], Float64[]
            for g in grecs
                c = match_cpu(cpu_recs, g)
                c === nothing && continue
                μ, σ², n = div_rv_metrics(c.times, g.times)
                lo, hi = ci_t(μ, σ², n)
                push!(xs, g.n)
                push!(μs, μ)
                push!(los, lo)
                push!(his, hi)
            end
            if !isempty(xs)
                lines!(ax_sp, xs, μs; color=color, linewidth=2, label=label)
                scatter!(ax_sp, xs, μs; color=color, markersize=10)
                length(xs) >= 2 && band!(ax_sp, xs, los, his; color=(color, 0.18))
            end

            # ── Error line ──
            xs_e = [g.n for g in grecs]
            ys_e = [g.err for g in grecs]
            lines!(ax_er, xs_e, ys_e; color=color, linewidth=2, label=label)
            scatter!(ax_er, xs_e, ys_e; color=color, markersize=10)

            # ── Rank line (max solid, mean dashed) ──
            if all(g -> hasproperty(g, :max_rank), grecs)
                xs_r  = [g.n for g in grecs]
                ys_mx = [g.max_rank for g in grecs]
                ys_mn = [g.mean_rank for g in grecs]
                lines!(ax_rk, xs_r, ys_mx; color=color, linewidth=2, label=label)
                scatter!(ax_rk, xs_r, ys_mx; color=color, markersize=10)
                lines!(ax_rk, xs_r, ys_mn; color=color, linewidth=1.2, linestyle=:dash)
            end
        end

        cpu_op_all = filter(d -> d.op == op, cpu_recs)
        if !isempty(cpu_op_all)
            ns = sort(unique(c.n for c in cpu_op_all))
            agg = (field) -> [
                median(getproperty(c, field) for c in cpu_op_all if c.n == nv) for nv in ns
            ]
            lines!(
                ax_er, ns, agg(:err);
                color=:gray, linestyle=:dash, linewidth=1.2, label="CPU ACA",
            )
            if all(c -> hasproperty(c, :max_rank), cpu_op_all)
                lines!(
                    ax_rk, ns, agg(:max_rank);
                    color=:gray, linewidth=1.5, label="CPU ACA",
                )
                lines!(
                    ax_rk, ns, agg(:mean_rank);
                    color=:gray, linestyle=:dash, linewidth=1.0,
                )
            end
        end

        col == n_ops && axislegend(ax_sp; position=:rt, title=string(category))
    end

    out = joinpath(output_dir, fname)
    save(out, fig)
    @info "Saved benchmark plot" out
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    datafile = isempty(ARGS) ? DATA_FILE : ARGS[1]
    @info "Loading benchmark data" datafile
    data = deserialize(datafile)
    plot_bench(data)
end
