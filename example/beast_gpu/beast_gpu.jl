# ---------------------------------------------------------------------------
# GPU-accelerated ACA example using the new GPUBlockAssembler API
# ---------------------------------------------------------------------------
#
# This example demonstrates the new GPU assembly path where:
#   1. Leaf blocks are assembled on GPU and stay in device memory
#   2. ACA compression runs on GPU via CUBLAS
#   3. Only the final low-rank factors (U, V) are transferred to CPU
#
# Prerequisites: BEAST.jl with CUDA support (BEASTCUDAExt)
# ---------------------------------------------------------------------------

using AdaptiveCrossApproximation
using BEAST
using CUDA
using CompScienceMeshes
using StaticArrays

# Create test geometry
m1 = meshrectangle(1.0, 1.0, 0.1)
m2 = translate(meshrectangle(1.0, 1.0, 0.1), SVector(2.0, 0.0, 0.0))

op = Helmholtz3D.singlelayer()
sp1 = lagrangecxd0(m1)
sp2 = lagrangecxd0(m2)

# ── CPU path (for comparison) ─────────────────────────────────────────────

aca = ACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormEstimator(1e-4),
)

println("=== CPU ACA ===")
rowbuffer_cpu = zeros(Float64, 50, length(sp2.pos))
colbuffer_cpu = zeros(Float64, length(sp1.pos), 50)
@time npivots_cpu = aca(
    AdaptiveCrossApproximation.AbstractKernelMatrix(op, sp1, sp2),
    colbuffer_cpu,
    rowbuffer_cpu,
    50;
    rowidcs=collect(1:length(sp1.pos)),
    colidcs=collect(1:length(sp2.pos)),
)
println("CPU pivots: ", npivots_cpu)

# ── GPU path ──────────────────────────────────────────────────────────────

if CUDA.functional()
    # Load the CUDA extension
    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    if ext !== nothing
        println("\n=== GPU ACA (assemble + compress on device) ===")

        # Create the GPU block assembler (pre-computes assembly primer)
        assembler = ext.GPUBlockAssembler(op, sp1, sp2; tol=1e-4, maxrank=50)

        # Compress the full block on GPU
        test_ids = collect(1:length(sp1.pos))
        trial_ids = collect(1:length(sp2.pos))

        @time npivots_gpu, U_gpu, V_gpu = ext.compress_block_gpu(
            assembler, test_ids, trial_ids
        )
        println("GPU pivots: ", npivots_gpu)
        println("U size: ", size(U_gpu), "  V size: ", size(V_gpu))

        # Verify: compare approximation quality
        K_cpu = AdaptiveCrossApproximation.AbstractKernelMatrix(op, sp1, sp2)
        A_dense = zeros(Float64, length(sp1.pos), length(sp2.pos))
        for i in 1:length(sp1.pos), j in 1:length(sp2.pos)
            K_cpu(view(A_dense, i:i, j:j), [i], [j])
        end
        A_approx = U_gpu * V_gpu
        rel_err = norm(A_dense - A_approx) / norm(A_dense)
        println("Relative approximation error: ", rel_err)
    else
        println("\nACACUDAExt not available. Ensure BEAST.jl has CUDA support.")
    end
else
    println("\nCUDA not available on this system.")
end
