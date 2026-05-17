# ---------------------------------------------------------------------------
# GPU vs CPU performance comparison for ACA compression
# ---------------------------------------------------------------------------
#
# Compares wall-clock time for CPU vs GPU assembly+compression across
# varying problem sizes. Uses the new GPUBlockAssembler API.
#
# Prerequisites: BEAST.jl with CUDA support (BEASTCUDAExt)
# ---------------------------------------------------------------------------

using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using StaticArrays
using CUDA

# ── CPU helper ────────────────────────────────────────────────────────────

function run_aca_cpu(op, sp1, sp2; tol=1e-4, maxrank=40)
    aca = ACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FNormEstimator(tol),
    )
    K = AdaptiveCrossApproximation.AbstractKernelMatrix(op, sp1, sp2)
    rowbuffer = zeros(Float64, maxrank, length(sp2.pos))
    colbuffer = zeros(Float64, length(sp1.pos), maxrank)
    res = @timed npivots = aca(
        K,
        colbuffer,
        rowbuffer,
        maxrank;
        rowidcs=collect(1:length(sp1.pos)),
        colidcs=collect(1:length(sp2.pos)),
    )
    return npivots, res.time
end

# ── GPU helper ────────────────────────────────────────────────────────────

function run_aca_gpu(assembler, test_ids, trial_ids; tol=1e-4, maxrank=40)
    res = CUDA.@timed begin
        npivots, U, V = ACACUDAExt.compress_block_gpu(assembler, test_ids, trial_ids)
        npivots
    end
    return res.value, res.time
end

# ── Benchmark loop ────────────────────────────────────────────────────────

problem_sizes = Int[]
cpu_times = Float64[]
gpu_times = Float64[]

N_TESTS = 10

ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
if ext === nothing
    error("ACACUDAExt not available. This benchmark requires BEAST + CUDA.")
end

for i in 1:N_TESTS
    h = 0.5 / i  # finer mesh as i increases
    m1 = meshsphere(; radius=0.5, h=max(h, 0.02))
    m2 = translate(m1, SVector(2.0, 0.0, 0.0))

    op = Helmholtz3D.singlelayer()
    sp1 = lagrangec0d1(m1)
    sp2 = lagrangec0d1(m2)

    n = length(sp1.pos) * length(sp2.pos)
    push!(problem_sizes, n)
    println(
        "\nIteration $i: block size = $(length(sp1.pos))×$(length(sp2.pos)) = $n elements"
    )

    # CPU run
    cpu_npivots, cpu_time = run_aca_cpu(op, sp1, sp2)
    push!(cpu_times, cpu_time)
    println("  CPU: $(round(cpu_time; digits=3))s, pivots=$cpu_npivots")

    # GPU run
    assembler = ext.GPUBlockAssembler(op, sp1, sp2; tol=1e-4, maxrank=40)
    test_ids = collect(1:length(sp1.pos))
    trial_ids = collect(1:length(sp2.pos))
    gpu_npivots, gpu_time = run_aca_gpu(assembler, test_ids, trial_ids)
    push!(gpu_times, gpu_time)
    println("  GPU: $(round(gpu_time; digits=3))s, pivots=$gpu_npivots")
    println("  Speedup: $(round(cpu_time / gpu_time; digits=1))×")
end

println("\n=== Summary ===")
for (i, sz) in enumerate(problem_sizes)
    println(
        "Size $sz: CPU=$(round(cpu_times[i]; digits=3))s  GPU=$(round(gpu_times[i]; digits=3))s  Speedup=$(round(cpu_times[i]/gpu_times[i]; digits=1))×",
    )
end
