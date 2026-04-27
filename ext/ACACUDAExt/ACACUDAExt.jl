module ACACUDAExt

using AdaptiveCrossApproximation:
    AdaptiveCrossApproximation,
    ACA,
    AbstractKernel,
    PivStrat,
    ConvCrit,
    MaximumValue,
    FNormEstimator,
    tolerance
using BEAST
using CUDA
using CUDA.CUBLAS
using LinearAlgebra
using CompScienceMeshes: legendre, Mesh, domain
using SparseArrays: SparseMatrixCSC, rowvals
using BEAST: BEAST
using CUDA: CUDA

# Cached at module init — safe because ACACUDAExt is only loaded when both BEAST and CUDA are present.
const _BEASTCUDAExt = Ref{Any}(nothing)

function __init__()
    CUDA.functional() && @info "CUDA available"
end


# dont know if this is significant but it caches the extension to prevent additional lookups
function _beast_cuda_ext()
    if _BEASTCUDAExt[] === nothing
        _BEASTCUDAExt[] = Base.get_extension(BEAST, :BEASTCUDAExt)
    end
    return _BEASTCUDAExt[]
end

struct AssemblyFunctorGPU{B,T1,T2,T3}
    biop::B
    tfs::T1
    bfs::T2
    quadstrat::T3
end

struct AbstractKernelGPU{K}
    functor::AssemblyFunctorGPU
    primer::NamedTuple
end

# data structure thats created per ACA call to prevent the previous shared mutable struct
# stores global dof to matrix index map for nextrc
# struct BlockMatrixKernel{M}
    matrix::M
    test_row_map::Dict{Int,Int}   # global test DOF  → row in matrix
    trial_col_map::Dict{Int,Int}  # global trial DOF → col in matrix
end

AdaptiveCrossApproximation.nextrc!(buf, A::BlockMatrixKernel, i, j) = begin
    M = A.matrix
    @inbounds for (kj, gj) in enumerate(j)
        cj = get(A.trial_col_map, gj, 0)
        cj == 0 && continue
        for (ki, gi) in enumerate(i)
            ri = get(A.test_row_map, gi, 0)
            ri == 0 && continue
            buf[ki, kj] += M[ri, cj]
        end
    end
end

function _make_kernel(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    if !CUDA.functional()
        @info "CUDA not available; using CPU AbstractKernel."
        assembler = BEAST.blockassembler(operator, testspace, trialspace; quadstrat)
        return AdaptiveCrossApproximation.AbstractKernel{
            scalartype(operator), typeof(assembler)}(assembler)
    end

    BEASTCUDAExt = _beast_cuda_ext()

    functor = AssemblyFunctorGPU(operator, testspace, trialspace, quadstrat)

    test_l2g, ((test_el_d, test_ad_d), (test_qr_d, test_shp_d)) =
        BEASTCUDAExt.assemble_primer_gpu(operator, testspace, quadstrat.outer_rule)
    trial_l2g, ((trial_el_d, trial_ad_d), (trial_qr_d, trial_shp_d)) =
        BEASTCUDAExt.assemble_primer_gpu(operator, trialspace, quadstrat.inner_rule)

    cv = legendre(quadstrat.sauter_schwab_common_vert, 0.0, 1.0)
    cv_d = CuArray([t for t in zip(cv[1], cv[2])])

    quaddata_d = ((test_qr_d, test_shp_d), (trial_qr_d, trial_shp_d), cv_d)

    test_g2l  = Dict{Int,Int}(g => l for (l, g) in enumerate(test_l2g))
    trial_g2l = Dict{Int,Int}(g => l for (l, g) in enumerate(trial_l2g))

    test_domain  = CUDA.@allowscalar domain(test_el_d[1])
    trial_domain = CUDA.@allowscalar domain(trial_el_d[1])
    numshapes_test  = numfunctions(refspace(testspace),  test_domain)
    numshapes_trial = numfunctions(refspace(trialspace), trial_domain)

    test_ad_cpu  = SparseMatrixCSC(test_ad_d)
    trial_ad_cpu = SparseMatrixCSC(trial_ad_d)

    test_dof_to_elems  = build_dof_to_elements(test_ad_cpu,  length(test_l2g),  numshapes_test)
    trial_dof_to_elems = build_dof_to_elements(trial_ad_cpu, length(trial_l2g), numshapes_trial)

    primer = (;
        test_l2g, trial_l2g, test_g2l, trial_g2l,
        test_el_d, trial_el_d,
        test_ad_d, trial_ad_d,
        quaddata_d,
        test_ad_cpu, trial_ad_cpu,
        test_dof_to_elems, trial_dof_to_elems,
        numshapes_test, numshapes_trial,
    )

    return AbstractKernelGPU{scalartype(operator)}(functor, primer)
end

function (aca::ACA)(
    M::AbstractKernelGPU,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    rows::T, cols::T, rowidcs::T, colidcs::T,
    maxrank::Int64,
) where {K<:Number, T<:Vector{Int64}}
    p = M.primer

    # Collect elements without extra index vector
    test_elems  = let buf = Int[]
        for g in rowidcs; append!(buf, p.test_dof_to_elems[p.test_g2l[g]]); end
        sort!(unique!(buf))
    end
    trial_elems = let buf = Int[]
        for g in colidcs; append!(buf, p.trial_dof_to_elems[p.trial_g2l[g]]); end
        sort!(unique!(buf))
    end

    matrix, test_active_rows, trial_active_rows = _beast_cuda_ext().assemble_block_gpu(
        M.functor.biop,
        refspace(M.functor.tfs), p.test_el_d, p.test_ad_cpu,
        refspace(M.functor.bfs), p.trial_el_d, p.trial_ad_cpu,
        p.quaddata_d, test_elems, trial_elems;
        numshapes_test=p.numshapes_test, numshapes_trial=p.numshapes_trial,
    )

    # global dof to matrix index map directly for nextrc! isntead of previous 2 lookups
    test_row_map  = Dict{Int,Int}(p.test_l2g[l]  => k for (k, l) in enumerate(test_active_rows))
    trial_col_map = Dict{Int,Int}(p.trial_l2g[l] => k for (k, l) in enumerate(trial_active_rows))

    blk = BlockMatrixKernel(matrix, test_row_map, trial_col_map)

    return invoke(
        aca,
        Tuple{Any, AbstractArray{K}, AbstractArray{K}, T, T, T, T, Int},
        blk, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank,
    )
end

function build_dof_to_elements(assemblydata::SparseMatrixCSC, ndofs::Int, nshapes::Int)
    rows = rowvals(assemblydata)
    out = [Int[] for _ in 1:ndofs]
    for column in 1:size(assemblydata, 2)
        element = (column - 1) ÷ nshapes + 1
        for p in assemblydata.colptr[column]:(assemblydata.colptr[column+1]-1)
            push!(out[rows[p]], element)
        end
    end
    foreach(unique!, out); foreach(sort!, out)
    return out
end

build_dof_to_elements(ad_dev::CUDA.CUSPARSE.CuSparseMatrixCSC, ndofs::Int, nshapes::Int) =
    build_dof_to_elements(SparseMatrixCSC(ad_dev), ndofs, nshapes)

export AbstractKernelGPU
end # module
