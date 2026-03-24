using AdaptiveCrossApproximation
using BEAST
using CUDA
using CompScienceMeshes
using StaticArrays

struct AbstractKernel{K}
    blockassembler::Function
end

function AbstractKernelGPU(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; gpu = true),
    )
end

function (M::AbstractKernel{K})(
    buf::AbstractArray{K},
    i::AbstractArray{Int,1},
    j::AbstractArray{Int,1},
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

# Define nextrc for the abstractkernel struct
AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)

aca = ACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormEstimator(0.0),
)

function run_aca(label, K, sp1, sp2, aca)
    println(label, "\n")
    rowbuffer = zeros(Float64, 50, length(sp2.pos))
    colbuffer = zeros(Float64, length(sp1.pos), 50)
    CUDA.@profile npivots = aca(K, rowbuffer, colbuffer, 50)
    return npivots, rowbuffer, colbuffer
end

m1 = readmesh("./BEAST.jl/test/assets/sphere2.in")
m2 = readmesh("./BEAST.jl/test/assets/torus.msh")

m1 = meshsphere(radius = 0.5, h = 0.01)
m2 = translate(m1, SVector(2.0, 0.0, 0.0))

op = Helmholtz3D.singlelayer()
sp1 = lagrangec0d1(m1)
sp2 = lagrangec0d1(m2)



function profile()
    K_builtin_profiler = AbstractKernelGPU(op, sp1, sp2)
    CUDA.@profile run_aca(
        "---Profiling run---",
        K_builtin_profiler,
        sp1,
        sp2,
        aca,
    )
    return nothing
end