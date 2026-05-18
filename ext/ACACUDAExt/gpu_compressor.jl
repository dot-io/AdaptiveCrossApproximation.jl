"""
    GPUCompressor{K,Op,TS,BS,RP,CP}
"""
struct GPUCompressor{K,Op,TS,BS,RP,CP}
    op::Op
    testspace::TS
    trialspace::BS
    assembler::GPUBlockAssembler{K}
    rowpivoting::RP
    columnpivoting::CP
    tol::Float64
end

Base.eltype(::GPUCompressor{K}) where {K} = K
Base.eltype(::GPUBlockAssembler{K}) where {K} = K

"""
    GPUCompressor(op, testspace, trialspace;
                  tol=1e-4, maxrank=40, batchsize=1,
                  kernel=:pair_scatter, rowpivoting=nothing, columnpivoting=nothing)

note to self PreserveSpaceOrder must be set. how can i assert this in the api?
"""
function GPUCompressor(
    op,
    testspace,
    trialspace;
    tol::Real=1e-4,
    maxrank::Int=40,
    batchsize::Int=1,
    kernel::Symbol=:pair_scatter,
    rowpivoting=nothing,
    columnpivoting=nothing,
)
    assembler = GPUBlockAssembler(
        op, testspace, trialspace; kernel=kernel, maxrank=maxrank, tol=Float64(tol)
    )
    rpiv = rowpivoting === nothing ?
        BatchedFillDistance(testspace.pos; batchsize=batchsize) : rowpivoting
    cpiv = columnpivoting === nothing ?
        BatchedFillDistance(trialspace.pos; batchsize=batchsize) : columnpivoting
    return GPUCompressor{
        eltype(assembler),typeof(op),typeof(testspace),typeof(trialspace),
        typeof(rpiv),typeof(cpiv),
    }(
        op, testspace, trialspace, assembler, rpiv, cpiv, Float64(tol)
    )
end

# can just return the compressor itself i think
(gc::GPUCompressor)(::Any, ::Int, ::Int, ::Int) = gc

"""
    (gc::GPUCompressor)(kernelmatrix, U_host, V_host, maxrank; rowidcs, colidcs)
"""
function (gc::GPUCompressor{K})(
    ::Any,
    U_host::AbstractMatrix,
    V_host::AbstractMatrix,
    maxrank::Int;
    rowidcs::AbstractVector{<:Integer},
    colidcs::AbstractVector{<:Integer},
    kwargs...,
) where {K}
    test_ids = collect(Int, rowidcs)
    trial_ids = collect(Int, colidcs)
    npivots, U_gpu, V_gpu = compress_block_gpu(
        gc.assembler,
        test_ids,
        trial_ids;
        rowpivoting=gc.rowpivoting,
        columnpivoting=gc.columnpivoting,
    )
    m = length(test_ids)
    n = length(trial_ids)
    if npivots > 0
        copyto!(view(U_host, 1:m, 1:npivots), Array(U_gpu))
        copyto!(view(V_host, 1:npivots, 1:n), Array(V_gpu))
    end
    return npivots
end
