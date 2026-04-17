using AdaptiveCrossApproximation
using BEAST
using CUDA
using CompScienceMeshes
using StaticArrays

#Define helper Functions

m1 = meshrectangle(1.0, 1.0, 0.1)
m2 = translate(meshrectangle(1.0, 1.0, 0.1), SVector(2.0, 0.0, 0.0))

op = Helmholtz3D.singlelayer()
sp1 = lagrangecxd0(m1)
sp2 = lagrangecxd0(m2)

struct AbstractKernel{K}
    blockassembler::Function
end

function AbstractKernel(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    gpu=false,
)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler_gpu(operator, testspace, trialspace)
    )
end

function (M::AbstractKernel{K})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

print("Testing CPU and GPU kernels for one matrix block...\n")

AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)

##
aca = ACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormEstimator(0.0),
)

function run_aca(label, K, sp1, sp2, aca)
    println(label)
    rowbuffer = zeros(Float64, 50, length(sp2.pos))
    colbuffer = zeros(Float64, length(sp1.pos), 50)
    @time npivots = aca(K, rowbuffer, colbuffer, 50)
    println("Pivots: ", npivots)
    return npivots, rowbuffer, colbuffer
end

K_cpu = AbstractKernel(op, sp1, sp2; gpu=false)
cpu_npivots, cpu_rowbuffer, cpu_colbuffer = run_aca("CPU run", K_cpu, sp1, sp2, aca)

K_gpu = AbstractKernel(op, sp1, sp2; gpu=true)
gpu_npivots, gpu_rowbuffer, gpu_colbuffer = run_aca("GPU run", K_gpu, sp1, sp2, aca)

row_max_diff = maximum(abs.(cpu_rowbuffer - gpu_rowbuffer))
col_max_diff = maximum(abs.(cpu_colbuffer - gpu_colbuffer))
row_same = isapprox(cpu_rowbuffer, gpu_rowbuffer; rtol=1e-6, atol=1e-6)
col_same = isapprox(cpu_colbuffer, gpu_colbuffer; rtol=1e-6, atol=1e-6)
pivots_same = cpu_npivots == gpu_npivots

println(
    "CPU/GPU match: pivots=", pivots_same, " rowbuffer=", row_same, " colbuffer=", col_same
)
println("Max abs diff: rowbuffer=", row_max_diff, " colbuffer=", col_max_diff)

println("----CUDA built-in GPU profiling run-----\n")
CUDA.@profile run_aca("GPU run 2", K_gpu, sp1, sp2, aca)
