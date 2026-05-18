const _BEASTCUDAExt = Ref{Any}(nothing)

function _beast_cuda_ext()
    if _BEASTCUDAExt[] === nothing
        ext = Base.get_extension(BEAST, :BEASTCUDAExt)
        if ext === nothing
            error(
                "BEASTCUDAExt not found. Ensure BEAST.jl is loaded with CUDA support. " *
                "The GPU assembly path requires BEAST's CUDA extension for " *
                "`assembleblock_primer_gpu` and `assembleblock_body_gpu!`.",
            )
        end
        _BEASTCUDAExt[] = ext
    end
    return _BEASTCUDAExt[]
end

mutable struct GPUBufferPool
    buffers::Dict{Tuple{Type,Int,Int},CuMatrix}
    lock::ReentrantLock

    GPUBufferPool() = new(Dict{Tuple{Type,Int,Int},CuMatrix}(), ReentrantLock())
end

function _acquire_buffer!(pool::GPUBufferPool, ::Type{K}, rows::Int, cols::Int;) where {K}
    key = (K, rows, cols)
    lock(pool.lock) do
        if haskey(pool.buffers, key)
            buf = pool.buffers[key]
            fill!(buf, zero(K))
            return buf
        else
            buf = CUDA.zeros(K, rows, cols)
            pool.buffers[key] = buf
            return buf
        end
    end
end

function _release_all!(pool::GPUBufferPool)
    lock(pool.lock) do
        empty!(pool.buffers)
    end
    return nothing
end

struct GPUBlockAssembler{K}
    biop::Any
    tfs::Any
    bfs::Any

    ctx::NamedTuple

    buffer_pool::GPUBufferPool

    maxrank::Int
    tol::Float64
    block_size_threshold::Int  # blocks with fewer elements go to CPU
    kernel::Symbol             # which GPU kernel variant to use

    function GPUBlockAssembler{K}(
        biop,
        tfs,
        bfs,
        ctx;
        maxrank::Int=40,
        tol::Float64=1e-4,
        block_size_threshold::Int=128,
        kernel::Symbol=:pair_scatter,
    ) where {K}
        return new{K}(
            biop, tfs, bfs, ctx, GPUBufferPool(), maxrank, tol, block_size_threshold, kernel
        )
    end
end

"""
    GPUBlockAssembler(operator, testspace, trialspace; kwargs...)

Create a GPU block assembler from BEAST operator and spaces.
"""
function GPUBlockAssembler(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    kernel::Symbol=:gather_tile_coop,
    maxrank::Int=40,
    tol::Float64=1e-4,
    block_size_threshold::Int=128,
)
    if !CUDA.functional()
        error("CUDA is not functional. Cannot create GPUBlockAssembler.")
    end

    BEASTCUDAExt = _beast_cuda_ext()

    ctx = BEASTCUDAExt.assembleblock_primer_gpu(operator, testspace, trialspace; kernel)

    K = ctx.ZT  # scalar type for this prob
    return GPUBlockAssembler{K}(
        operator, testspace, trialspace, ctx; maxrank, tol, block_size_threshold, kernel
    )
end

"""
    _assemble_block_gpu(assembler, test_ids, trial_ids)

keeping this in if i still want to benchmark "naive dense assembly"
"""
function _assemble_block_gpu(
    assembler::GPUBlockAssembler{K}, test_ids::Vector{Int}, trial_ids::Vector{Int}
) where {K}
    BEASTCUDAExt = _beast_cuda_ext()

    m = length(test_ids)
    n = length(trial_ids)

    block = CUDA.zeros(K, m, n)
    store = BEASTCUDAExt.CuMatrixStore(block)

    BEASTCUDAExt.assembleblock_body_gpu!(
        assembler.biop,
        assembler.tfs,
        test_ids,
        assembler.bfs,
        trial_ids,
        assembler.ctx,
        store;
        kernel=assembler.kernel,
    )

    return block
end

"""
    _assemble_rows_gpu!(dest, assembler, test_ids_selected, trial_ids)
"""
function _assemble_rows_gpu!(
    dest::CuMatrix{K},
    assembler::GPUBlockAssembler{K},
    test_ids_selected::AbstractVector{<:Integer},
    trial_ids::AbstractVector{<:Integer},
) where {K}
    BEASTCUDAExt = _beast_cuda_ext()
    store = BEASTCUDAExt.CuMatrixStore(dest)
    BEASTCUDAExt.assembleblock_body_gpu!(
        assembler.biop,
        assembler.tfs,
        collect(Int, test_ids_selected),
        assembler.bfs,
        collect(Int, trial_ids),
        assembler.ctx,
        store;
        kernel=assembler.kernel,
    )
    return dest
end

"""
    _assemble_cols_gpu!(dest, assembler, test_ids, trial_ids_selected)
"""
function _assemble_cols_gpu!(
    dest::CuMatrix{K},
    assembler::GPUBlockAssembler{K},
    test_ids::AbstractVector{<:Integer},
    trial_ids_selected::AbstractVector{<:Integer},
) where {K}
    BEASTCUDAExt = _beast_cuda_ext()
    store = BEASTCUDAExt.CuMatrixStore(dest)
    BEASTCUDAExt.assembleblock_body_gpu!(
        assembler.biop,
        assembler.tfs,
        collect(Int, test_ids),
        assembler.bfs,
        collect(Int, trial_ids_selected),
        assembler.ctx,
        store;
        kernel=assembler.kernel,
    )
    return dest
end

"""
    compress_block_gpu(assembler, test_ids, trial_ids; rowpivoting, columnpivoting)
"""
function compress_block_gpu(
    assembler::GPUBlockAssembler{K},
    test_ids::Vector{Int},
    trial_ids::Vector{Int};
    rowpivoting,
    columnpivoting,
) where {K}
    m = length(test_ids)
    n = length(trial_ids)

    U_buf = _acquire_buffer!(assembler.buffer_pool, K, m, assembler.maxrank)
    V_buf = _acquire_buffer!(assembler.buffer_pool, K, assembler.maxrank, n)

    rowpiv_fun = rowpivoting(test_ids)
    colpiv_fun = columnpivoting(trial_ids)

    npivots = aca_gpu!(
        assembler,
        U_buf,
        V_buf,
        test_ids,
        trial_ids,
        assembler.maxrank;
        rowpivoting=rowpiv_fun,
        columnpivoting=colpiv_fun,
        tol=assembler.tol,
    )

    U = CuMatrix{K}(undef, m, npivots)
    V = CuMatrix{K}(undef, npivots, n)
    copyto!(U, view(U_buf, 1:m, 1:npivots))
    copyto!(V, view(V_buf, 1:npivots, 1:n))

    return npivots, U, V
end

"""
    use_gpu(assembler, nrows, ncols)

Determine whether a block of size (nrows, ncols) should be processed on GPU.
Blocks below the size threshold are routed to CPU to avoid GPU overhead.
"""
function use_gpu(assembler::GPUBlockAssembler, nrows::Int, ncols::Int)
    return nrows * ncols >= assembler.block_size_threshold
end
