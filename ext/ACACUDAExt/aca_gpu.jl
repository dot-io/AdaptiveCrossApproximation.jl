"""
    aca_gpu!(A, U, V, rowidcs, colidcs, maxrank;
             rowpivoting, columnpivoting, tol)
"""
function aca_gpu!(
    A::CuMatrix{K},
    U::CuMatrix{K},
    V::CuMatrix{K},
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int},
    maxrank::Int;
    rowpivoting,
    columnpivoting,
    tol::Real=1e-4,
) where {K<:Number}
    m = length(rowidcs)
    n = length(colidcs)
    @assert size(A) == (m, n)
    @assert size(U, 1) == m && size(U, 2) >= maxrank
    @assert size(V, 1) >= maxrank && size(V, 2) == n

    b_row = rowpivoting isa BatchedPivStratFunctor ? batchsize(rowpivoting) : 1
    b_col = columnpivoting isa BatchedPivStratFunctor ? batchsize(columnpivoting) : 1
    b_max = max(b_row, b_col)

    if b_max > 1
        rowpivoting isa BatchedPivStratFunctor || throw(
            ArgumentError(
                "rowpivoting must be a BatchedPivStratFunctor when batch size > 1, " *
                "got $(typeof(rowpivoting))",
            ),
        )
        columnpivoting isa BatchedPivStratFunctor || throw(
            ArgumentError(
                "columnpivoting must be a BatchedPivStratFunctor when batch size > 1, " *
                "got $(typeof(columnpivoting))",
            ),
        )
    end

    fill!(U, zero(K))
    fill!(V, zero(K))

    row_resid = CUDA.zeros(K, b_max, n)
    col_resid = CUDA.zeros(K, m, b_max)
    U_I_buf = CUDA.zeros(K, b_max, maxrank)
    V_J_buf = CUDA.zeros(K, maxrank, b_max)
    R_IJ_buf = CUDA.zeros(K, b_max, b_max)

    npivot = 0
    normUV = zero(real(K))
    converged = false

    while npivot < maxrank && !converged
        I_local = _select_batch(rowpivoting, npivot)
        J_local = _select_batch(columnpivoting, npivot)

        (isempty(I_local) || isempty(J_local)) && break

        b = min(length(I_local), length(J_local), maxrank - npivot)
        b == 0 && break
        I_local = I_local[1:b]
        J_local = J_local[1:b]

        row_view = view(row_resid, 1:b, 1:n)
        col_view = view(col_resid, 1:m, 1:b)

        _gather_rows!(row_view, A, I_local)
        _gather_cols!(col_view, A, J_local)

        if npivot > 0
            UI_view = view(U_I_buf, 1:b, 1:npivot)
            _gather_rows!(UI_view, U, I_local, npivot)
            CUBLAS.gemm!(
                'N', 'N', -one(K), UI_view, view(V, 1:npivot, 1:n), one(K), row_view
            )

            VJ_view = view(V_J_buf, 1:npivot, 1:b)
            _gather_cols!(VJ_view, V, J_local, npivot)
            CUBLAS.gemm!(
                'N', 'N', -one(K), view(U, 1:m, 1:npivot), VJ_view, one(K), col_view
            )
        end

        R_IJ_view = view(R_IJ_buf, 1:b, 1:b)
        _gather_cols!(R_IJ_view, row_view, J_local, b)

        R_IJ_cpu = Array(R_IJ_view)
        F = svd(R_IJ_cpu)
        sigma = F.S
        sigma_1 = sigma[1]

        if isapprox(sigma_1, zero(real(K)))
            break
        end

        k_trunc = findfirst(s -> s <= tol * sigma_1, sigma)

        #TODO remove logging
        @show b, k_trunc

        k_trunc = k_trunc === nothing ? length(sigma) : k_trunc - 1
        k_trunc = max(k_trunc, 1)
        k_trunc = min(k_trunc, maxrank - npivot)

        sqrt_sigma_inv_h = [one(real(K)) / sqrt(sigma[i]) for i in 1:k_trunc]
        sqrt_sigma_inv = CuArray(K.(sqrt_sigma_inv_h))

        Vsvd_k = CuMatrix{K}(F.V[:, 1:k_trunc])
        Usvd_k = CuMatrix{K}(F.U[:, 1:k_trunc])

        U_target = view(U, 1:m, (npivot + 1):(npivot + k_trunc))
        CUBLAS.gemm!('N', 'N', one(K), col_view, Vsvd_k, zero(K), U_target)
        U_target .*= reshape(sqrt_sigma_inv, 1, k_trunc)

        V_target = view(V, (npivot + 1):(npivot + k_trunc), 1:n)
        CUBLAS.gemm!('C', 'N', one(K), Usvd_k, row_view, zero(K), V_target)
        V_target .*= reshape(sqrt_sigma_inv, k_trunc, 1)

        batch_contribution = zero(real(K))
        @inbounds for i in 1:k_trunc
            p = npivot + i
            u_col = view(U, 1:m, p)
            v_row = view(V, p, 1:n)
            rnorm = CUBLAS.nrm2(u_col)
            cnorm = CUBLAS.nrm2(v_row)
            if !isapprox(rnorm, zero(real(K))) && !isapprox(cnorm, zero(real(K)))
                normUV += (rnorm * cnorm)^2
                for j in 1:(p - 1)
                    cdot = _cublas_dot(u_col, view(U, 1:m, j))
                    rdot = _cublas_dot(v_row, view(V, j, 1:n))
                    normUV += 2 * real(cdot * rdot)
                end
            end
            batch_contribution += abs2(sigma[i])
        end

        npivot += k_trunc

        if normUV > zero(real(K))
            converged = sqrt(batch_contribution) <= tol * sqrt(normUV)
        end
    end

    return npivot
end

"""
    _select_batch(pivoting, npivot)
"""
function _select_batch(pivoting::BatchedPivStratFunctor, npivot::Int)
    return npivot == 0 ? pivoting() : pivoting(npivot)
end

function _select_batch(pivoting, npivot::Int)
    idx = npivot == 0 ? pivoting() : pivoting(npivot)
    return idx isa AbstractVector ? collect(Int, idx) : Int[idx]
end

"""
    _gather_rows!(dest, src, rows[, ncols])

Copy `src[rows[i], 1:ncols]` into `dest[i, 1:ncols]` on the device for each `i`.
Defaults `ncols` to `size(src, 2)`. One async copy per row.
"""
function _gather_rows!(dest::AbstractMatrix, src::AbstractMatrix, rows::AbstractVector{Int})
    return _gather_rows!(dest, src, rows, size(src, 2))
end

function _gather_rows!(
    dest::AbstractMatrix, src::AbstractMatrix, rows::AbstractVector{Int}, ncols::Int
)
    @inbounds for (i, r) in enumerate(rows)
        copyto!(view(dest, i:i, 1:ncols), view(src, r:r, 1:ncols))
    end
    return dest
end

"""
    _gather_cols!(dest, src, cols[, nrows])
"""
function _gather_cols!(dest::AbstractMatrix, src::AbstractMatrix, cols::AbstractVector{Int})
    return _gather_cols!(dest, src, cols, size(src, 1))
end

function _gather_cols!(
    dest::AbstractMatrix, src::AbstractMatrix, cols::AbstractVector{Int}, nrows::Int
)
    @inbounds for (j, c) in enumerate(cols)
        copyto!(view(dest, 1:nrows, j:j), view(src, 1:nrows, c:c))
    end
    return dest
end

"""
    _cublas_dot(x, y)
"""
_cublas_dot(x::StridedCuVector{T}, y::StridedCuVector{T}) where {T<:Real} =
    CUBLAS.dot(length(x), x, y)
_cublas_dot(x::StridedCuVector{T}, y::StridedCuVector{T}) where {T<:Complex} =
    CUBLAS.dotc(length(x), x, y)
