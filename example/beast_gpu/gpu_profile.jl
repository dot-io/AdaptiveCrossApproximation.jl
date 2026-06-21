# ---------------------------------------------------------------------------
# CUDA profiler example for GPU-accelerated ACA
# ---------------------------------------------------------------------------
#
# Uses CUDA.@profile to generate a profile of the GPU ACA loop.
# Run with: nsys profile -t cuda julia gpu_profile.jl
#
# Prerequisites: BEAST.jl with CUDA support (BEASTCUDAExt)
# ---------------------------------------------------------------------------

using AdaptiveCrossApproximation
using BEAST
using CUDA
using CompScienceMeshes
using StaticArrays

# Create test geometry
m1 = meshsphere(; radius=0.5, h=0.05)
m2 = translate(m1, SVector(2.0, 0.0, 0.0))

op = Helmholtz3D.singlelayer()
sp1 = lagrangec0d1(m1)
sp2 = lagrangec0d1(m2)

ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
if ext === nothing
    error("ACACUDAExt not available. This profiling script requires BEAST + CUDA.")
end

# Create the GPU block assembler
assembler = ext.GPUBlockAssembler(op, sp1, sp2; tol=1e-4, maxrank=50)
test_ids = collect(1:length(sp1.pos))
trial_ids = collect(1:length(sp2.pos))

# Warmup run (JIT compilation)
println("Warmup run...")
npivots, U, V = ext.compress_block_gpu(assembler, test_ids, trial_ids)
println("Warmup done. Pivots: ", npivots)

# Profiled run
println("\nStarting profiled run...")
CUDA.@profile begin
    for _ in 1:5
        npivots, U, V = ext.compress_block_gpu(assembler, test_ids, trial_ids)
    end
end
println("Profiled run complete. Pivots: ", npivots)
