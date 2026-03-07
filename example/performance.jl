using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using StaticArrays

function mesh_path(filename)
    candidates = String[
        joinpath(@__DIR__, filename),
        joinpath(@__DIR__, "..", "..", "BEAST.jl", "test", "assets", filename),
        joinpath(@__DIR__, "..", "..", "BEAST.jl", "examples", filename),
        normpath(joinpath(dirname(pathof(BEAST)), "..", "test", "assets", filename)),
        normpath(joinpath(dirname(pathof(BEAST)), "..", "examples", filename)),
    ]

    for path in candidates
        if isfile(path)
            return path
        end
    end

    error("Mesh file not found: $(filename). Checked: " * join(candidates, ", "))
end
struct AbstractKernel{K}
    blockassembler::Function
end

function AbstractKernelGPU(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; gpu=true)
    )
end

function AbstractKernel(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space)
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

function run_aca(label, K, sp1, sp2, aca; maxrank=1200)
    println(label)
    rowbuffer = zeros(Float64, maxrank, length(sp2.pos))
    colbuffer = zeros(Float64, length(sp1.pos), maxrank)
    @time npivots = aca(K, rowbuffer, colbuffer, maxrank)
    println("Pivots: ", npivots)
    return npivots, rowbuffer, colbuffer
end

println("Loading spherical mesh...")
m1 = readmesh(mesh_path("sphere872.in"))
refine_levels = 1
for _ in 1:refine_levels
    global m1 = barycentric_refinement(m1).mesh
end
m2 = translate(m1, SVector(2.0, 0.0, 0.0))

op = Helmholtz3D.singlelayer()
sp1 = lagrangecxd0(m1)
sp2 = lagrangecxd0(m2)
println("DOFs: test=", length(sp1.pos), " trial=", length(sp2.pos))

aca = ACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormEstimator(0.0),
)

println("\nCPU/GPU comparison (for spherical mesh)")
K_cpu = AbstractKernel(op, sp1, sp2)
cpu_npivots, cpu_rowbuffer, cpu_colbuffer = run_aca("CPU run", K_cpu, sp1, sp2, aca)

println(typeof(sp1), "\t", typeof(sp2))

K_gpu = AbstractKernelGPU(op, sp1, sp2)
gpu_npivots, gpu_rowbuffer, gpu_colbuffer = run_aca("GPU run", K_gpu, sp1, sp2, aca)

row_max_diff = maximum(abs.(cpu_rowbuffer - gpu_rowbuffer))
row_same = isapprox(cpu_rowbuffer, gpu_rowbuffer; rtol=1e-6, atol=1e-6)
pivots_same = cpu_npivots == gpu_npivots

println("CPU/GPU match: pivots=", pivots_same, " buffer=", row_same)
println("Max abs diff: ", row_max_diff)
