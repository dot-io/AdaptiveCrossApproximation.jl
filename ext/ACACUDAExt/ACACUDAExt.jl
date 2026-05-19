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
    BatchedFillDistance,
    batchsize,
    nextrc!,
    reset!,
    tolerance

using BEAST
using CUDA
using CUDA.CUBLAS
using LinearAlgebra
using BEAST: BEAST
using CUDA: CUDA
using CUDA.CUSOLVER

include("assembly_gpu.jl")
include("aca_gpu.jl")
include("gpu_compressor.jl")

function __init__()
    if CUDA.functional()
        @info "ACACUDAExt: CUDA available, GPU assembly path enabled"
    else
        @warn "ACACUDAExt: CUDA not functional, GPU assembly path disabled"
    end
end

end
