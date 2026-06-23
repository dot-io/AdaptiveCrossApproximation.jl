"""
    aca_gpu!(assembler, U, V, rowidcs, colidcs, maxrank;
             rowpivoting, columnpivoting, tol,
             svd_compression, fnorm_iteration, svd_backend)

# Keyword arguments

  - `svd_compression::Bool` (default `false`): remove rows / columns
    whose singular values in the SVD is near-zero: this SHOULD improve numerical
    accuracy but remains to be tested
  - `fnorm_iteration::Bool` (default `false`): controls the F-norm accumulation (1-by-1 or batched).
  - `svd_backend::Symbol` (default `:cusolver`)
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
    svd_compression::Bool=false,
    fnorm_iteration::Bool=false,
    svd_backend::Symbol=:cusolver,
    svd_algorithm::CUDA.CUSOLVER.SVDAlgorithm=CUDA.CUSOLVER.JacobiAlgorithm(),
) where {K<:Number}
    svd_backend in (:cusolver, :cpu) ||
        throw(ArgumentError("svd_backend must be :cusolver or :cpu, got :$svd_backend"))

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

    # F-norm GEMV scratch: replaces O(p) dot-call syncs with two GEMVs per pivot.
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

        # On-demand assembly via the shared nextrc! interface (GPU specialisation
        # in assembly_gpu.jl), mirroring the CPU ACA driver in src/aca.jl.
        selected_test  = [Int(rowidcs[i]) for i in I_local]
        selected_trial = [Int(colidcs[j]) for j in J_local]
        row_view       = CUDA.zeros(K, b, n)
        col_view       = CUDA.zeros(K, m, b)
        nextrc!(row_view, assembler, selected_test, colidcs)
        nextrc!(col_view, assembler, rowidcs, selected_trial)

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

        # SVD of the b×b intersection residual.
        # Pull sigma to the host in both paths so threshold comparisons and
        # findfirst run without scalar-indexing a CuArray.
        local sigma_h::Vector{real(K)}, Usvd_k::CuMatrix{K}, Vsvd_k::CuMatrix{K}
        if svd_backend === :cusolver
            G = copy(R_IJ_view)
            U_s, S_s, V_s = CUDA.CUSOLVER._svd!(G, false, svd_algorithm)
            sigma_h = Array(S_s)
            Usvd_k = CuMatrix{K}(U_s)    # already on device
            Vsvd_k = CuMatrix{K}(V_s)
        else
            F = svd!(Array(R_IJ_view))
            sigma_h = F.S
            Usvd_k = CuMatrix{K}(F.U)    # tfr back to device
            Vsvd_k = CuMatrix{K}(F.V)
        end

        sigma_1 = sigma_h[1]
        if isapprox(sigma_1, zero(real(K)))
            break
        end

        b_full = length(sigma_h)
        b_full = min(b_full, maxrank - npivot)
        if b_full == 0
            converged = true
            break
        end

        sqrt_sigma_inv = CuArray(K.([one(real(K)) / sqrt(sigma_h[i]) for i in 1:b_full]))

        U_full = view(U, 1:m, (npivot + 1):(npivot + b_full))
        CUBLAS.gemm!('N', 'N', one(K), col_view, view(Vsvd_k, :, 1:b_full), zero(K), U_full)
        U_full .*= reshape(sqrt_sigma_inv, 1, b_full)

        V_full = view(V, (npivot + 1):(npivot + b_full), 1:n)
        CUBLAS.gemm!('C', 'N', one(K), view(Usvd_k, :, 1:b_full), row_view, zero(K), V_full)
        V_full .*= reshape(sqrt_sigma_inv, b_full, 1)

        u_norms = Array(vec(sqrt.(sum(abs2, U_full; dims=1))))
        v_norms = Array(vec(sqrt.(sum(abs2, V_full; dims=2))))

        per_pivot = false
        if fnorm_iteration
            # Per-pivot trim: stop inside the batch as soon as the new pivot's
            # F-norm² contribution is negligible relative to the running ‖UV‖_F².
            k_trunc = 0
            running = normUV
            @inbounds for i in 1:b_full
                contrib_i = abs2(u_norms[i]) * abs2(v_norms[i]) #works for cpx type
                k_trunc += 1
                running += contrib_i
                if running > zero(real(K)) && contrib_i <= tol^2 * running
                    per_pivot = true
                    break
                end
            end
        else
            k_trunc = b_full
        end

        if svd_compression && k_trunc > 0
            kk = findfirst(s -> s <= tol * sigma_1, view(sigma_h, 1:k_trunc))
            if kk !== nothing
                k_trunc = max(kk, 1)
                per_pivot = true
            end
        end

        batch_contribution = zero(real(K))
        @inbounds for i in 1:k_trunc
            batch_contribution += abs2(u_norms[i]) * abs2(v_norms[i])
        end

        if fnorm_iteration
            # Per-pivot F-norm with cross-terms (expensive: O(p) GEMVs per pivot).
            @inbounds for i in 1:k_trunc
                p = npivot + i
                u_col = view(U, 1:m, p)
                v_row = view(V, p, 1:n)
                rnorm = CUBLAS.nrm2(u_col)
                cnorm = CUBLAS.nrm2(v_row)
                if !isapprox(rnorm, zero(real(K))) && !isapprox(cnorm, zero(real(K)))
                    normUV += (rnorm * cnorm)^2
                    if p > 1
                        U_prev = view(U, 1:m, 1:(p - 1))
                        V_prev = view(V, 1:(p - 1), 1:n)
                        ucdots = view(ucdots_buf, 1:(p - 1))
                        vrdots = view(vrdots_buf, 1:(p - 1))

                        CUBLAS.gemv!('C', one(K), U_prev, u_col, zero(K), ucdots)

                        if K <: Complex
                            vrow_conj = view(vrow_conj_buf, 1:n)
                            vrow_conj .= conj.(v_row)
                            CUBLAS.gemv!('N', one(K), V_prev, vrow_conj, zero(K), vrdots)
                        else
                            CUBLAS.gemv!('N', one(K), V_prev, v_row, zero(K), vrdots)
                        end

                        normUV += 2 * real(sum(ucdots .* vrdots))
                    end
                end
            end
        else
            # Cheap per-batch accumulator: no kernel launches, no syncs.
            # Treats batches as orthogonal contributions to ‖UV‖_F².
            normUV += batch_contribution
        end

        npivot += k_trunc

        if per_pivot
            converged = true
        elseif normUV > zero(real(K))
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
