"""
    aca_gpu!(assembler, U, V, rowidcs, colidcs, maxrank;
             rowpivoting, columnpivoting, tol)
"""
function aca_gpu!(
    assembler::GPUBlockAssembler{K},
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

    U_I_buf = CUDA.zeros(K, b_max, maxrank)
    V_J_buf = CUDA.zeros(K, maxrank, b_max)
    R_IJ_buf = CUDA.zeros(K, b_max, b_max)

    # F-norm update scratch: per-pivot dot products with all earlier pivots
    # are computed as two single GEMVs instead of 2·(p-1) CUBLAS.dot calls,
    # which is the difference between O(maxrank²) and O(maxrank) host syncs.
    ucdots_buf = CUDA.zeros(K, maxrank)
    vrdots_buf = CUDA.zeros(K, maxrank)
    vrow_conj_buf = K <: Complex ? CUDA.zeros(K, n) : nothing

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

        # i accidentally assembled entire block here
        selected_test  = [Int(rowidcs[i]) for i in I_local]
        selected_trial = [Int(colidcs[j]) for j in J_local]
        row_view = CUDA.zeros(K, b, n)
        col_view = CUDA.zeros(K, m, b)
        _assemble_rows_gpu!(row_view, assembler, selected_test, colidcs)
        _assemble_cols_gpu!(col_view, assembler, rowidcs, selected_trial)

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
                if p > 1
                    # 2 gemv more efficient than the many dots from before i think
                    U_prev = view(U, 1:m, 1:(p - 1))
                    V_prev = view(V, 1:(p - 1), 1:n)
                    ucdots = view(ucdots_buf, 1:(p - 1))
                    vrdots = view(vrdots_buf, 1:(p - 1))

                    # ucdots = U_prev† * u_p
                    CUBLAS.gemv!('C', one(K), U_prev, u_col, zero(K), ucdots)

                    # vrdots[j]  = V_prev * (v_p)†
                    # maybe dont have to allocate vrow_conj_buf and just do inplace?
                    # but is conj transpose so wrong dims
                    if K <: Complex
                        vrow_conj = view(vrow_conj_buf, 1:n)
                        vrow_conj .= conj.(v_row)
                        CUBLAS.gemv!('N', one(K), V_prev, vrow_conj, zero(K), vrdots)
                    else
                        CUBLAS.gemv!('N', one(K), V_prev, v_row, zero(K), vrdots)
                    end

                    cross = 2 * real(sum(ucdots .* vrdots)) #crossterm justlike in aca
                    normUV += cross
                end
            end
            batch_contribution += abs2(sigma[i])
        end

        npivot += k_trunc

        if normUV > zero(real(K)) # how can i move it on-device
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
