using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using CUDA
using StaticArrays
using LinearAlgebra

# Create test geometry with finer mesh
m1 = meshrectangle(1.0, 1.0, 0.05)
m2 = translate(meshrectangle(1.0, 1.0, 0.05), SVector(2.0, 0.0, 0.0))

op = Helmholtz3D.singlelayer()
sp1 = lagrangecxd0(m1)
sp2 = lagrangecxd0(m2)

quadstrat = BEAST.DoubleNumQStrat(4, 4)

println("CPU ACA with far quadrature strategy (4,4)")

aca = ACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormEstimator(1e-4),
)

K_cpu = AdaptiveCrossApproximation.AbstractKernelMatrix(op, sp1, sp2; matrixdata=quadstrat)
rowbuffer_cpu = zeros(Float64, 80, length(sp2.pos))
colbuffer_cpu = zeros(Float64, length(sp1.pos), 80)
@time npivots_cpu = aca(
    K_cpu,
    colbuffer_cpu,
    rowbuffer_cpu,
    80;
    rowidcs=collect(1:length(sp1.pos)),
    colidcs=collect(1:length(sp2.pos)),
)
println("CPU pivots: ", npivots_cpu)

if CUDA.functional()
    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    if ext !== nothing
        println("GPU ACA with DoubleNumQStrat(4,4)")

        assembler = ext.GPUBlockAssembler(op, sp1, sp2; tol=1e-4, maxrank=80)
        test_ids = collect(1:length(sp1.pos))
        trial_ids = collect(1:length(sp2.pos))

        @time npivots_gpu, U_gpu, V_gpu = ext.compress_block_gpu(
            assembler, test_ids, trial_ids
        )
        println("GPU pivots: ", npivots_gpu)

        # Verify approximation quality
        K_verify = AdaptiveCrossApproximation.AbstractKernelMatrix(
            op, sp1, sp2; matrixdata=quadstrat
        )
        A_dense = zeros(Float64, length(sp1.pos), length(sp2.pos))
        for i in 1:length(sp1.pos), j in 1:length(sp2.pos)
            K_verify(view(A_dense, i:i, j:j), [i], [j])
        end
        A_approx = U_gpu * V_gpu
        rel_err = norm(A_dense - A_approx) / norm(A_dense)
        println("Relative approximation error: ", rel_err)
    else
        @warn "ACACUDAExt not available."
    end
else
    @warn "CUDA not available on this system."
end
