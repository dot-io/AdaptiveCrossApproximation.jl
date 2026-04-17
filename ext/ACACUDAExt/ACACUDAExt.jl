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

include("AbstractKernelGPU.jl")

# --------------------------------------------------------------------------
# Small CUDA helpers
# --------------------------------------------------------------------------

# rowbuffer[npivot, :] .-= sum_{k=1:npivot-1} colbuffer[nextrow, k] * rowbuffer[k, :]
# Deflation operation is parallelized over the columns of the rowbuffer
# function _kernel_deflate_row!(
#     rowbuffer, colbuffer, nextrow::Int32, npivot::Int32, maxcols::Int32
# )
#     j = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
#     if j <= maxcols
#         acc = rowbuffer[npivot, j]
#         k = Int32(1)
#         while k < npivot
#             acc -= colbuffer[nextrow, k] * rowbuffer[k, j]
#             k += Int32(1)
#         end
#         rowbuffer[npivot, j] = acc
#     end
#     return nothing
# end

# colbuffer[:, npivot] .-= sum_{k=1:npivot-1} colbuffer[:, k] * rowbuffer[k, nextcolumn]
# function _kernel_deflate_col!(
#     colbuffer, rowbuffer, nextcolumn::Int32, npivot::Int32, maxrows::Int32
# )
#     i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
#     if i <= maxrows
#         acc = colbuffer[i, npivot]
#         k = Int32(1)
#         while k < npivot
#             acc -= colbuffer[i, k] * rowbuffer[k, nextcolumn]
#             k += Int32(1)
#         end
#         colbuffer[i, npivot] = acc
#     end
#     return nothing
# end

# rowbuffer[npivot, :] ./= pivot
# function _kernel_normalize_row!(rowbuffer, npivot::Int32, pivot, maxcols::Int32)
#     j = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
#     if j <= maxcols
#         rowbuffer[npivot, j] = rowbuffer[npivot, j] / pivot
#     end
#     return nothing
# end

# Compute max abs and argmax in one pass for a vector on device.
# Host-side reduction is acceptable here for first CUDA path.
@inline function _gpu_argmax_abs(v::CUDA.CuVector{T}) where {T}
    # CUDA mapreduce keeps computation on device; moving only scalar result back.
    m = mapreduce(abs, max, v)
    # get first index with abs(v[i]) == m; this keeps logic deterministic with ties by first hit.
    # This fallback uses Array conversion of a tiny boolean vector in worst case, acceptable baseline.
    # For production, replace with custom reduction kernel returning index.
    idx = Int(findfirst(x -> x == m, Array(abs.(v))))
    return idx
end

@inline function has_converged(
    tol::T, rownorm::T, colnorm::T, normF::T
) where {T<:AbstractFloat}
    if isapprox(rownorm, zero(T)) && isapprox(colnorm, zero(T))
        return false
    elseif isapprox(rownorm, zero(T)) || isapprox(colnorm, zero(T))
        return false
    end
    return (rownorm * colnorm) <= tol * sqrt(normF)
end

# Frobenius estimate update mirrored from CPU normF! implementation.
function update_normUV2!(
    normF::Base.RefValue{T},
    rowbuffer::CuMatrix{T},
    colbuffer::CuMatrix{T},
    npivots::Vector{Int},
    maxrows::Int,
    maxcols::Int,
) where {T<:AbstractFloat}
    @views r = rowbuffer[npivot:npivot, 1:maxcols]
    @views c = colbuffer[1:maxrows, npivot:npivot]
    """
    How to define this normalization for the blocked/batched case? TODO

    need to write down proof neatly but:
    ||UV||_F^2 = ||U_prev V_prev||_F^2 + cdotc(herk(V), herk(U)) + 2 * real(dot(herkx(V_prev, V), herkx(U_prev, U)))
    """
    n_cols = length(npivots)
    @views u_new = rowbuffer[npivots, 1:maxcols]
    @views v_new = colbuffer[1:maxrows, npivots]
    VVh = CUBLAS.herk('N', 'N', v_new)
    UhU = CUBLAS.herk('T', 'T', u_new)
    new_norm = CUBLAS.dotc(VVh, UhU) # Squared frobenius norm of the new matrix
    # VpVh = CUBLAS.herk
    normF[] = accumulator
    return rnorm, cnorm
end

# --------------------------------------------------------------------------
# CUDA ACA main routine (CuArray fast path)
# --------------------------------------------------------------------------

"""
    (aca::ACA{RP,CP,C})(A::CuArray{T,2}, colbuffer::CuArray{T,2}, rowbuffer::CuArray{T,2},
                        rows::Vector{Int}, cols::Vector{Int}, rowidcs::Vector{Int},
                        colidcs::Vector{Int}, maxrank::Int)

CUDA fast path for ACA main routine (host-controlled iterative loop + GPU kernels),
currently specialized for `MaximumValue` pivoting and `FNormEstimator` convergence.
"""
function (aca::ACA{RP,CP,C})(
    A::CUDA.CuArray{T,2},
    colbuffer::CUDA.CuArray{T,2},
    rowbuffer::CUDA.CuArray{T,2},
    rows::Vector{Int},
    cols::Vector{Int},
    rowidcs::Vector{Int},
    colidcs::Vector{Int},
    maxrank::Int,
) where {T<:Number,RP<:PivStrat,CP<:PivStrat,C<:ConvCrit}
    # Restrict this CUDA path to MaximumValue + FNormEstimator.
    if !(
        aca.rowpivoting isa MaximumValue &&
        aca.columnpivoting isa MaximumValue &&
        aca.convergence isa FNormEstimator
    )
        throw(
            ArgumentError(
                "CUDA ACA extension currently supports only MaximumValue pivoting with FNormEstimator convergence.",
            ),
        )
    end

    maxrows = size(colbuffer, 1)
    maxcols = size(rowbuffer, 2)

    # Mirror used-index masks on host for deterministic pivot tracking.
    row_used = falses(maxrows)
    col_used = falses(maxcols)

    # Device copies of global index arrays for gathers.
    d_rowidcs = CUDA.CuArray(rowidcs)
    d_colidcs = CUDA.CuArray(colidcs)

    npivot = 1

    # Initial row pivot = 1 (as in MaximumValueFunctor()).
    nextrow = 1
    row_used[nextrow] = true
    rows[1] = rowidcs[nextrow]

    # rowbuffer[1, :] = A[rowidcs[nextrow], colidcs[:]]
    @views rowbuffer[npivot:npivot, 1:maxcols] .= A[
        d_rowidcs[nextrow:nextrow], d_colidcs[1:maxcols]
    ]

    # Next column pivot by max abs on rowbuffer[npivot, :] over unused cols.
    @views rowvec = rowbuffer[npivot, 1:maxcols]
    rowvec_h = Array(rowvec)
    bestval = zero(T)
    nextcolumn = 1
    @inbounds for j in 1:maxcols
        if (!col_used[j]) && abs(rowvec_h[j]) >= bestval
            bestval = abs(rowvec_h[j])
            nextcolumn = j
        end
    end
    col_used[nextcolumn] = true
    cols[npivot] = colidcs[nextcolumn]

    # Normalize row by pivot if nonzero.
    pivotval = rowvec_h[nextcolumn]
    if pivotval != zero(T)
        """
        scal!(
        n: Number of elements in vector
        alpha: Scalar to multiply by
        x: vector
        )

        will use dgmm!(.) to generalize this to the batched case
        """
        CUBLAS.scal!(maxcols, 1 / pivotval, rowbuffer[npivot:npivot, 1:maxcols])
    end

    # colbuffer[:, npivot] = A[rowidcs[:], colidcs[nextcolumn]]
    @views colbuffer[1:maxrows, npivot:npivot] .= A[
        d_rowidcs[1:maxrows], d_colidcs[nextcolumn:nextcolumn]
    ]

    # Convergence state
    normUV2 = Ref{T}(zero(T))
    rnorm, cnorm = update_normUV2(normUV2, rowbuffer, colbuffer, npivot, maxrows, maxcols)
    conv = has_converged(T(aca.convergence.tol), rnorm, cnorm, normUV2[])

    while !conv && npivot < maxrank
        npivot += 1

        # row pivot from previous column colbuffer[:, npivot-1], among unused rows.
        @views prevcol_h = Array(colbuffer[1:maxrows, max(1, npivot - 1)])
        bestval = zero(T)
        nextrow = 1
        @inbounds for i in 1:maxrows
            if (!row_used[i]) && abs(prevcol_h[i]) >= bestval
                bestval = abs(prevcol_h[i])
                nextrow = i
            end
        end
        row_used[nextrow] = true
        rows[npivot] = rowidcs[nextrow]

        # Gather new row
        @views rowbuffer[npivot:npivot, 1:maxcols] .= A[
            d_rowidcs[nextrow:nextrow], d_colidcs[1:maxcols]
        ]

        # Deflate new row
        # threads = 256
        # blocks = cld(maxcols, threads)
        # @cuda threads = threads blocks = blocks _kernel_deflate_row!(
        #     rowbuffer, colbuffer, Int32(nextrow), Int32(npivot), Int32(maxcols)
        # )
        CUBLAS.gemm!(
            'N',
            'T',
            -one(T),
            rowbuffer[1:(npivot - 1), 1:maxcols],
            colbuffer[nextrow:nextrow, 1:(npivot - 1)],
            one(T),
            rowbuffer[npivot:npivot, 1:maxcols],
        )

        # Column pivot from deflated row among unused cols
        @views rowvec = rowbuffer[npivot, 1:maxcols]
        rowvec_h = Array(rowvec)
        bestval = zero(T)
        nextcolumn = 1
        @inbounds for j in 1:maxcols
            if (!col_used[j]) && abs(rowvec_h[j]) >= bestval
                bestval = abs(rowvec_h[j])
                nextcolumn = j
            end
        end
        col_used[nextcolumn] = true
        cols[npivot] = colidcs[nextcolumn]

        pivotval = rowvec_h[nextcolumn]
        if pivotval != zero(T)
            # blocks = cld(maxcols, threads)
            # @cuda threads = threads blocks = blocks _kernel_normalize_row!(
            #     rowbuffer, Int32(npivot), pivotval, Int32(maxcols)
            # )
            CUBLAS.scal!(
                maxcols, 1 / pivotval, rowbuffer[npivot:npivot, 1:maxcols], maxcols
            )

            # Gather column for selected pivot
            @views colbuffer[1:maxrows, npivot:npivot] .= A[
                d_rowidcs[1:maxrows], d_colidcs[nextcolumn:nextcolumn]
            ]

            # Deflate new column
            """
            from https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api :
            y = α op(A) x + β y

            gemv!(
            trans: op(.) --> 'N' or 'T' to show whether or not to transpose the matrix
            Note from me on CUDA.jl's column major storage (cf. https://discourse.julialang.org/t/cuarray-is-row-major-or-column-major/7402) BLAS can be optimized in native C++ thanks to its 'incx' property which allows strided accesses, which is as far as I know not supported in CUDA.jl. It may be worth looking at an alternative (TODO)

            alpha: alpha scalar as in above formula;
            A: Matrix, in this case 'colbuffer';
            ......
            )

            """
            CUBLAS.gemm!(
                'N',
                'N' - one(T),
                colbuffer[1:maxrows, 1:(npivot - 1)],
                rowbuffer[1:(npivot - 1), nextcolumn],
                one(T),
                colbuffer[1:maxrows, npivot:npivot],
            )

            # blocks = cld(maxrows, threads)
            # @cuda threads = threads blocks = blocks _kernel_deflate_col!(
            #   colbuffer, rowbuffer, Int32(nextcolumn), Int32(npivot), Int32(maxrows)
            # )
        end

        rnorm, cnorm = update_normUV2!(
            normUV2, rowbuffer, colbuffer, npivot, maxrows, maxcols
        )
        conv = _should_continue(T(aca.convergence.tol), rnorm, cnorm, normUV2[])
    end

    return npivot
end

"""
    (aca::ACA{RP,CP,C})(A::CuArray{T,2}, colbuffer::CuArray{T,2}, rowbuffer::CuArray{T,2}, maxrank::Int; kwargs...)

CUDA convenience overload corresponding to the CPU method:
initializes rows/cols and delegates to the CUDA main routine.
"""
function (aca::ACA{RP,CP,C})(
    A::CUDA.CuArray{T,2},
    colbuffer::CUDA.CuArray{T,2},
    rowbuffer::CUDA.CuArray{T,2},
    maxrank::Int;
    rows    = zeros(Int, maxrank),
    cols    = zeros(Int, maxrank),
    rowidcs = Vector(1:size(colbuffer, 1)),
    colidcs = Vector(1:size(rowbuffer, 2)),
) where {T<:AbstractFloat,RP<:PivStrat,CP<:PivStrat,C<:ConvCrit}
    return aca(A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank)
end

"""
    aca(M::CuArray{T,2}; kwargs...)

CUDA overload of high-level `aca` convenience function for dense CuMatrix input,
supporting `MaximumValue` + `FNormEstimator`.
"""
function AdaptiveCrossApproximation.aca(
    M::CUDA.CuArray{T,2};
    tol=1e-4,
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(tol),
    maxrank=40,
    svdrecompress=false,
) where {T<:AbstractFloat}
    compressor = ACA(rowpivoting, columnpivoting, convergence)
    rowbuffer = CUDA.zeros(T, maxrank, size(M, 2))
    colbuffer = CUDA.zeros(T, size(M, 1), maxrank)

    npivots = compressor(M, colbuffer, rowbuffer, maxrank)

    if svdrecompress
        @views Q, R = qr(colbuffer[1:size(M, 1), 1:npivots])
        @views U, s, V = svd(R * rowbuffer[1:npivots, 1:size(M, 2)])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tolerance(convergence) * s[1]
                opt_r = i
                break
            end
        end

        A = (Q * U)[1:size(M, 1), 1:opt_r]
        B = (diagm(s) * V')[1:opt_r, 1:size(M, 2)]
        return A, B
    else
        return colbuffer[1:size(M, 1), 1:npivots], rowbuffer[1:npivots, 1:size(M, 2)]
    end
end

end # module
