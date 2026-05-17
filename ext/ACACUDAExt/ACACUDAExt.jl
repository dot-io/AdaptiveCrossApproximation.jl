module ACACUDAExt

using AdaptiveCrossApproximation:
    AdaptiveCrossApproximation,
    ACA,
    PivStrat,
    ConvCrit,
    MaximumValue,
    FNormEstimator,
    BatchedPivStrat,
    BatchedPivStratFunctor,
    batchsize,
    reset!,
    tolerance

using BEAST
using CUDA
using CUDA.CUBLAS
using LinearAlgebra
using BEAST: BEAST
using CUDA: CUDA

include("aca_gpu.jl")
include("assembly_gpu.jl")

function __init__()
    if CUDA.functional()
        @info "ACACUDAExt: CUDA available, GPU assembly path enabled"
    else
        @warn "ACACUDAExt: CUDA not functional, GPU assembly path disabled"
    end
end

end
