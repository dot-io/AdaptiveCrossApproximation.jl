# GPU-compatible kernel type and constructor dispatch.
# Inspired by Joshua Tetzner's `abstractbeastmatrix.jl` definitions.
# BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)
import BEAST, CUDA
struct AbstractKernelGPU{K}
    blockassembler::Any
end

function _make_kernel(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    gpu=false,
)
    # NOTE: BEAST's blockassembler_gpu does not yet support index-subset queries
    # required by ACA. GPU block assembly is not yet available; using CPU.
    if CUDA.functional() && gpu
        @info "CUDA available , using GPU."
        # gpu_assembler = BEASTCUDAExt.blockassembler_gpu(
        #     operator, testspace, trialspace; quadstrat=quadstrat
        # )
        BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)
          if isnothing(BEASTCUDAExt)
              error("BEASTCUDAExt is not loaded. Ensure CUDA is loaded before calling _make_kernel with gpu=true.")
          end
        assembly_functor = BEASTCUDAExt.AssemblyFunctorGPU(
            operator, testspace, trialspace, quadstrat
        )

        return AbstractKernelGPU{scalartype(operator)}(assembly_functor)
    else
        @info "CUDA not available; using CPU AbstractKernel."
        assembler = BEAST.blockassembler(
            operator, testspace, trialspace; quadstrat=quadstrat
        )
        return AdaptiveCrossApproximation.AbstractKernel{
            scalartype(operator),typeof(assembler)
        }(
            assembler
        )
    end
end

function (M::AbstractKernelGPU{K})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernelGPU, i, j) = A(buf, i, j)

export AbstractKernelGPU
