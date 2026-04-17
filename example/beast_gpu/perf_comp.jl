"""
Test file to compare the performance of CPU vs GPU-accelerated (partially) code for different system matrix sizes to assemble.
"""

using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using StaticArrays
using CUDA
using Plots

struct AbstractKernel{K}
    blockassembler::Function
end

function AbstractKernelGPU(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; gpu=true)
    )
end

function AbstractKernel(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; gpu=false)
    )
end

function (M::AbstractKernel{K})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)

# Important: program will error if maxrank is larger than number of cells in test/trial space
function run_aca(label, K::AbstractKernel, sp1, sp2, aca; maxrank=40)
    println(label)
    rowbuffer = zeros(Float64, maxrank, length(sp2.pos))
    colbuffer = zeros(Float64, length(sp1.pos), maxrank)
    @time npivots = aca(K, rowbuffer, colbuffer, maxrank)
    println("Pivots: ", npivots)
    return npivots, rowbuffer, colbuffer
end

"""
Overload for the GPU run in order to use CUDA's built-in timing functions which should separate compilation and run time.
"""
function run_aca(label, K::AbstractKernel, sp1, sp2, aca; maxrank=40)
    println(label)
    rowbuffer = zeros(Float64, maxrank, length(sp2.pos))
    colbuffer = zeros(Float64, length(sp1.pos), maxrank)
    CUDA.@time npivots = aca(K, rowbuffer, colbuffer, maxrank)
    println("Pivots: ", npivots)
    return npivots, rowbuffer, colbuffer
end

"""
We create multiple mesh pairs to see the scaling behaviour. Benchmarking against a CPU is not valuable in and of itself but comparing GPU vs CPU scaling behaviour could be interesting.
"""
problem_sizes = []
runtimes_cpu = []
runtimes_gpu = []
N_TESTS = 30
for i in 1:N_TESTS
    m1 = meshsphere(; radius=0.5, h=0.5 * (N_TESTS / i))
    m2 = translate(m1, [0.0, 2.0, 0.0])
    println("Block size 1: ", numcells(m1_1) * numcells(m2_1))

    println("Iteration with problem size ", numcells(m1) * numcells(m2), "...")
    problem_sizes.push(numcells(m1) * numcells(m2))

    op = Helmholtz3D.singlelayer()
    sp1 = lagrangec0d1(m1)
    sp2 = lagrangec0d1(m2)

    aca = ACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FNormEstimator(0.0),
    )

    println("\nCPU/GPU comparison (for spherical mesh)")
    K_cpu = AbstractKernel(op, sp1, sp2)
    res = CUDA.@timed run_aca("CPU run", K_cpu, sp1, sp2, aca)
    runtimes_cpu.push(res.time)
    cpu_npivots, cpu_rowbuffer, cpu_colbuffer = res.value
    K_gpu = AbstractKernelGPU(op, sp1, sp2)
    res = CUDA.@timed run_aca("GPU run", K_gpu, sp1, sp2, aca)
    runtimes_gpu.push(res.time)
    gpu_npivots, gpu_rowbuffer, gpu_colbuffer = res.value

    row_max_diff = maximum(abs.(cpu_rowbuffer - gpu_rowbuffer))
    row_same = isapprox(cpu_rowbuffer, gpu_rowbuffer; rtol=1e-6, atol=1e-6)
    column_same = isapprox(cpu_rowbuffer, gpu_rowbuffer; rtol=1e-6, atol=1e-6)
    pivots_same = cpu_npivots == gpu_npivots

    println("CPU/GPU match: pivots=", pivots_same, " buffer=", row_same, column_same)
    println("Max abs diff: ", row_max_diff)
end
