module ACACUDAExt

# using AdaptiveCrossApproximation
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

if CUDA.functional()
    @info "CUDA available"
    # gpu_assembler = BEASTCUDAExt.blockassembler_gpu(
    #     operator, testspace, trialspace; quadstrat=quadstrat
    # )
end

# GPU-compatible kernel type and constructor dispatch.
# Inspired by Joshua Tetzner's `abstractbeastmatrix.jl` definitions.
# BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)
using BEAST: BEAST
using CUDA: CUDA

struct AssemblyFunctorGPU{B,T1,T2,T3}
    biop::B
    tfs::T1
    bfs::T2
    quadstrat::T3
end

#TODO: store the "computed" var elsewhere to prevent this struct being mutable
mutable struct AssemblyContext
    test_local_global_map::Any
    trial_local_global_map::Any
    test_elements::Any
    trial_elements::Any
    test_assembly_data::Any
    trial_assembly_data::Any
    quaddata::Tuple
    unified_assembly::Bool
    computed::Bool
    test_global_local_map::Any
    trial_global_local_map::Any
    test_dof_to_elems::Any
    trial_dof_to_elems::Any
    test_assemblydata_cpu::Any
    trial_assemblydata_cpu::Any
end

# I think I need to make the struct mutable to append the context later on, TODO: check if it works
mutable struct AbstractKernelGPU{K}
    functor::AssemblyFunctorGPU
    primer::Union{Nothing, NamedTuple}
    context::Union{Nothing, AssemblyContext}
end

function _make_kernel(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)

    if !CUDA.functional()
        @info "CUDA not available; using CPU AbstractKernel."
        assembler = BEAST.blockassembler(operator, testspace, trialspace; quadstrat)
        return AdaptiveCrossApproximation.AbstractKernel{
            scalartype(operator), typeof(assembler)}(assembler)
    end

    functor = AssemblyFunctorGPU(operator, testspace, trialspace, quadstrat)

    # ---- primer: run once per kernel ----
    test_l2g, ((test_el_d, test_ad_d), (test_qr_d, test_shp_d)) =
        BEASTCUDAExt.assemble_primer_gpu(operator, testspace, quadstrat.outer_rule)
    trial_l2g, ((trial_el_d, trial_ad_d), (trial_qr_d, trial_shp_d)) =
        BEASTCUDAExt.assemble_primer_gpu(operator, trialspace, quadstrat.inner_rule)

    cv = legendre(quadstrat.sauter_schwab_common_vert, 0.0, 1.0)
    q  = [t for t in zip(cv[1], cv[2])]
    cv_d = CuArray(q)

    quaddata_d = ((test_qr_d, test_shp_d), (trial_qr_d, trial_shp_d), cv_d)

    test_g2l  = Dict{Int,Int}(g => l for (l, g) in enumerate(test_l2g))
    trial_g2l = Dict{Int,Int}(g => l for (l, g) in enumerate(trial_l2g))

    primer = (;
        test_l2g, trial_l2g, test_g2l, trial_g2l,
        test_el_d, trial_el_d,
        test_ad_d, trial_ad_d,
        quaddata_d,
    )

    return AbstractKernelGPU{scalartype(operator)}(functor, primer, nothing)
end


# function (M::AbstractKernelGPU{K})(
#     buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
# ) where {K}
#     # @views store(v, m, n) = (buf[m, n] += v)
#     # return M.functor(i, j, store)
#     if M.context.unified_assembly

#     end

# end

# AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernelGPU, i, j) = begin
# if A.context === nothing
#     error(
#         "Primer routine not run yet.",
#     )
# end
#     # here I need to pass the pivoting strategy and convergence criterion, to check if they allow row/col by row/col assembly (e.g. max pivoting does not, RS pivoting does)
#     # I could create abstract types which each pivstrat that allows this inherits
#     ctx = A.context
#     if (ctx.unified_assembly)
#        if !ctx.computed

#             A.context.computed = true
#        end
#     else
#         # compute row-by-row TODO: compare performance of these methods
#         println("actually in the right path")
#         store(v, m, n) = (buf[m, n] += v)
#                        store_dev(v, m, n) = store(v, ctx.test_local_global_map[m], ctx.trial_local_global_map[n])
#                        BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)
#                        # TODO: allow specifying indices to only assemble certain rows, cols
#                        gi = getindex.(Ref(ctx.test_local_global_map), i)
#                        gj = getindex.(Ref(ctx.trial_local_global_map), j)
#                        BEASTCUDAExt.assembleblock_body_gpu!(
#                            A.functor.biop,
#                            refspace(A.functor.tfs),
#                            ctx.test_elements,
#                            ctx.test_assembly_data,
#                            refspace(A.functor.bfs),
#                            ctx.trial_elements,
#                            ctx.trial_assembly_data,
#                            ctx.quaddata,
#                            li,
#                            lj,
#                            store_dev,
#                        )
#     end
#     # A(buf, i, j)
# end
AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernelGPU, i, j) = begin

    # println("actually in the right path")
    ctx = A.context
    BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)

    # map the global index sets i and j to locals
    li = Int[ctx.test_global_local_map[g]  for g in i]
    lj = Int[ctx.trial_global_local_map[g] for g in j]

    # get the element indices that will need to be iterated upon in assembleblock_body_gpu!
    test_elems  = sort!(unique!(reduce(vcat, ctx.test_dof_to_elems[li])))
    trial_elems = sort!(unique!(reduce(vcat, ctx.trial_dof_to_elems[lj])))

    li_pos = Dict(l => k for (k, l) in enumerate(li))
    lj_pos = Dict(l => k for (k, l) in enumerate(lj))
    function store_buf(v, m, n)    # m and n are the local dof indices
        k = get(li_pos, m, 0); l = get(lj_pos, n, 0)
        (k > 0 && l > 0) && (buf[k, l] += v)
    end

    BEASTCUDAExt.assembleblock_body_gpu!(
        A.functor.biop,
        refspace(A.functor.tfs), ctx.test_elements,  ctx.test_assemblydata_cpu,
        refspace(A.functor.bfs), ctx.trial_elements, ctx.trial_assemblydata_cpu,
        ctx.quaddata, test_elems, trial_elems, store_buf,
    )
end

function (aca::ACA)(M::AbstractKernelGPU, colbuffer::AbstractArray{K}, rowbuffer::AbstractArray{K}, rows::T, cols::T,
                    rowidcs::T, colidcs::T, maxrank::Int64) where {K<:Number, T<:Vector{Int64}}
    p = M.primer

    test_domain  = CUDA.@allowscalar domain(p.test_el_d[1])
    trial_domain = CUDA.@allowscalar domain(p.trial_el_d[1])
    numshapes_test  = numfunctions(refspace(M.functor.tfs), test_domain)
    numshapes_trial = numfunctions(refspace(M.functor.bfs), trial_domain)

    test_dof_to_elems  = build_dof_to_elements(p.test_ad_d,  length(p.test_l2g),  numshapes_test)
    trial_dof_to_elems = build_dof_to_elements(p.trial_ad_d, length(p.trial_l2g), numshapes_trial)

    test_ad_cpu  = SparseMatrixCSC(p.test_ad_d)
    trial_ad_cpu = SparseMatrixCSC(p.trial_ad_d)

    ctx = AssemblyContext(
        p.test_l2g, p.trial_l2g,
        p.test_el_d, p.trial_el_d,
        p.test_ad_d, p.trial_ad_d,
        p.quaddata_d,
        false, false,
        p.test_g2l, p.trial_g2l,
        test_dof_to_elems, trial_dof_to_elems,
        test_ad_cpu, trial_ad_cpu,
    )
    M.context = ctx
    return invoke(
        aca,
        Tuple{Any, AbstractArray{K}, AbstractArray{K}, T, T, T, T, Int},
        M, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank,
    )
end


# a map that converts degrees of freedom to a vector of elements
# inputting the assemblydata in this gives the
function build_dof_to_elements(assemblydata_dev::CUDA.CUSPARSE.CuSparseMatrixCSC, ndofs::Int, nshapes::Int)
    assemblydata = SparseMatrixCSC(assemblydata_dev)
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

export AbstractKernelGPU
end # module
