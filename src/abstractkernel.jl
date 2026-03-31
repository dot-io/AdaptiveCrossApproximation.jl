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
    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    if ext !== nothing
        return ext._make_kernel(operator, testspace, trialspace; quadstrat=quadstrat, gpu=gpu)
    end
    assembler = BEAST.blockassembler(operator, testspace, trialspace; quadstrat=quadstrat)
    return AbstractKernel{scalartype(operator), typeof(assembler)}(assembler)
end

function (M::AbstractKernel{K,B})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K,B}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)

export AbstractKernel
