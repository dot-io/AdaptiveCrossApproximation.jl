using BEAST

struct AbstractKernel{K,B}
    blockassembler::B
end

function AbstractKernel(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    gpu=false,
)
    if gpu
        ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDA)
        if ext !== nothing
            return ext.GPUBlockAssembler(operator, testspace, trialspace)
        else
            @warn "CUDA extension not available; falling back to CPU assembly."
        end
    end
    assembler = BEAST.blockassembler(operator, testspace, trialspace; quadstrat=quadstrat)
    return AbstractKernel{scalartype(operator),typeof(assembler)}(assembler)
end

function (M::AbstractKernel{K,B})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K,B}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)

export AbstractKernel
