"""
    BatchedACA{RowPivType,ColPivType,ConvCritType}
broad outline:

 1. Select index subsets I (rows) and J (columns) using a batched pivoting strategy
 2. Assemble A[I,:] and A[:,J] via the kernel matrix interface
 3. Compute residuals R[I,:] = A[I,:] - U[I,:]*V and R[:,J] = A[:,J] - U*V[:,J]
 4. Compute the SVD of the intersection residual: R[I,J] = W * Σ * X'
 5. Truncate based on singular values: keep only σ_k > ε * σ_1
 6. Update U and V factors: U_new = R[:,J] * X[:,J'] * √Σ†, V_new = √Σ† * W[I',:] * R[I,:]
 7. Append U_new, V_new to the existing U, V factors
 8. Check F-norm convergence criterion

inspired by "A GPU parallel randomized CUR compression method for the
Method of Moments", uses fill-distance pivoting instead of random sampling, and
maintains the incremental F-norm convergence criterion from standard ACA.

See also [`BatchedFillDistance`](@ref), [`FNormEstimator`](@ref), [`ACA`](@ref)
"""
struct BatchedACA{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function BatchedACA(rowpivoting, columnpivoting, convergence)
        # Enforce that both pivoting strategies are batched
        if !(rowpivoting isa BatchedPivStrat)
            throw(
                ArgumentError(
                    "rowpivoting must be a BatchedPivStrat, got $(typeof(rowpivoting))"
                ),
            )
        end
        if !(columnpivoting isa BatchedPivStrat)
            throw(
                ArgumentError(
                    "columnpivoting must be a BatchedPivStrat, got $(typeof(columnpivoting))",
                ),
            )
        end
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

"""
    BatchedACA(; tol=1e-4, rowpivoting, columnpivoting, convergence, batchsize=4)

Construct a BatchedACA compressor with keyword arguments.

# Example

```julia
using AdaptiveCrossApproximation

pos = [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(0.0, -1.0)]
row_piv = BatchedFillDistance(pos; batchsize=4)
col_piv = BatchedFillDistance(pos; batchsize=4)

compressor = BatchedACA(; rowpivoting=row_piv, columnpivoting=col_piv, tol=1e-4)
```
"""
function BatchedACA(;
    tol=1e-4,
    rowpivoting::BatchedPivStrat,
    columnpivoting::BatchedPivStrat,
    convergence=FNormEstimator(tol),
    batchsize::Int=4,
)
    return BatchedACA(rowpivoting, columnpivoting, convergence)
end

# TODO document constructor
function (baca::BatchedACA{RP,CP,C})(
    A, rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {RP<:BatchedPivStrat,CP<:BatchedPivStrat,C<:ConvCrit}
    convcrit = _buildconvcrit(baca.convergence, A, rowidcs, colidcs, maxrank)
    rowpiv = baca.rowpivoting(rowidcs, batchsize(baca.rowpivoting))
    colpiv = baca.columnpivoting(colidcs, batchsize(baca.columnpivoting))

    return BatchedACA{typeof(rowpiv),typeof(colpiv),typeof(convcrit)}(
        rowpiv, colpiv, convcrit
    )
end

#TODO document batchsize getter
function batchsize(strat::BatchedPivStrat)
    return strat.b
end

#TODO doc reset fn
function reset!(
    baca::BatchedACA{RP,CP,C}, rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}
) where {RP<:BatchedPivStratFunctor,CP<:BatchedPivStratFunctor,C<:ConvCritFunctor}
    reset!(baca.rowpivoting, rowidcs)
    reset!(baca.columnpivoting, colidcs)
    reset!(baca.convergence)
    return nothing
end

"""
    (baca::BatchedACA)(A, U, V, maxrank; rowidcs, colidcs)

Compute batched ACA approximation with preallocated buffers.
Fill `U` and `V` with low-rank factors such that `A[rowidcs, colidcs] ≈ U * V`.
Use batched pivot selection and SVD-based residual compression.
"""
function (baca::BatchedACA{RP,CP,C})(
    A,
    U::AbstractMatrix{K},
    V::AbstractMatrix{K},
    maxrank::Int;
    rowidcs::AbstractVector{Int}=Vector(1:size(U, 1)),
    colidcs::AbstractVector{Int}=Vector(1:size(V, 2)),
) where {K,RP<:BatchedPivStratFunctor,CP<:BatchedPivStratFunctor,C<:ConvCritFunctor}
    reset!(baca, rowidcs, colidcs)
    return baca(A, U, V, rowidcs, colidcs, maxrank)
end

"""
    (baca::BatchedACA)(A, U, V, rowidcs, colidcs, maxrank)

Execute the algorithm outlined above TODO put here?
Return `npivot`: number of pivots (rank of approximation)
"""
function (baca::BatchedACA{RP,CP,C})(
    A,
    U::AbstractMatrix{K},
    V::AbstractMatrix{K},
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int},
    maxrank::Int,
) where {K,RP<:BatchedPivStratFunctor,CP<:BatchedPivStratFunctor,C<:ConvCritFunctor}
    m = length(rowidcs)  # number of rows in block
    n = length(colidcs)  # number of columns in block

    fill!(U, zero(K))
    fill!(V, zero(K))

    npivot = 0
    conv = true  # continue iterating
    normUV² = zero(real(K))

    # Temporary storage for assembled rows/columns
    row_batch = Matrix{K}(undef, m, n)   # A[I,:] for current batch I
    col_batch = Matrix{K}(undef, m, n)   # A[:,J] for current batch J

    while conv && npivot < maxrank
        # ── Step 1: Select batch of row indices I ──────────────────────────
        I_local = if npivot == 0
            baca.rowpivoting()
        else
            baca.rowpivoting(npivot)
        end

        isempty(I_local) && break
        b_r = min(length(I_local), maxrank - npivot) # limit index batch size if > maxrank
        I_local = I_local[1:b_r]

        # ── Step 2: Select batch of column indices J ───────────────────────
        J_local = if npivot == 0
            baca.columnpivoting()
        else
            baca.columnpivoting(npivot)
        end

        isempty(J_local) && break
        b_c = min(length(J_local), maxrank - npivot)
        J_local = J_local[1:b_c]

        # Use the minimum batch size to keep the intersection square
        b = min(b_r, b_c)
        I_local = I_local[1:b]
        J_local = J_local[1:b]

        # A[I,:] row_batch_view (b x n)
        row_batch_view = view(row_batch, 1:b, 1:n)
        fill!(row_batch_view, zero(K))
        for (li, gi) in enumerate(I_local)
            nextrc!(
                view(row_batch_view, li:li, 1:n),
                A,
                view(rowidcs, gi:gi),
                view(colidcs, 1:n),
            )
        end

        # A[:,J] col_batch_view (m x b)
        col_batch_view = view(col_batch, 1:m, 1:b)
        fill!(col_batch_view, zero(K))
        for (lj, gj) in enumerate(J_local)
            nextrc!(
                view(col_batch_view, 1:m, lj:lj),
                A,
                view(rowidcs, 1:m),
                view(colidcs, gj:gj),
            )
        end

        # ── Step 4: Compute residuals ─────────────────────────────────────
        # R[I,:] = A[I,:] - U[I,:] * V  (b × n)
        # R[:,J] = A[:,J] - U * V[:,J]  (m × b)
        if npivot > 0
            U_I = view(U, I_local, 1:npivot)   # (b × npivot)
            V_full = view(V, 1:npivot, 1:n)      # (npivot × n)
            R_row = copy(row_batch_view)
            mul!(R_row, U_I, V_full, -one(K), one(K))

            U_full = view(U, 1:m, 1:npivot)     # (m × npivot)
            V_J = view(V, 1:npivot, J_local)    # (npivot × b)
            R_col = copy(col_batch_view)
            mul!(R_col, U_full, V_J, -one(K), one(K))
        else
            R_row = copy(row_batch_view)
            R_col = copy(col_batch_view)
        end

        # ── Step 5: SVD of intersection residual R[I,J] ────────────────────
        R_IJ = view(R_row, 1:b, J_local)  # (b × b)

        F = svd(R_IJ)
        sigma = F.S

        # ── Step 6: Truncate based on singular values ───────────────────────
        # Keep only σ_k > ε * σ_1
        tol = tolerance(baca.convergence)
        sigma_1 = sigma[1]

        # Handle zero block
        if isapprox(sigma_1, 0.0)
            conv = false
            break
        end

        k = findfirst(s -> s <= tol * sigma_1, sigma)
        k = k === nothing ? length(sigma) : k - 1
        k = max(k, 1)  # always keep at least one

        # ── Step 7: Compute new U and V factors ─────────────────────────────
        # SVD: R[I,J] = U_svd * Σ * V_svd'
        # Pseudoinverse: R[I,J]† = V_svd * Σ† * U_svd'
        #
        # CUR decomposition: A ≈ C * R[I,J]† * R
        # where C = A[:,J] and R = A[I,:]
        #
        # Using residuals instead of raw rows/columns:
        #   U_new = R[:,J] * V_svd[:,1:k] * diag(1/√σ_i)   → (m × k)
        #   V_new = diag(1/√σ_i) * U_svd[:,1:k]' * R[I,:]  → (k × n)
        #
        # This ensures U_new * V_new ≈ R[:,J] * R[I,J]† * R[I,:]
        # which is the rank-k truncated pseudoinverse contribution.

        sqrt_sigma_inv = similar(sigma, k)
        @inbounds for i in 1:k
            sqrt_sigma_inv[i] = 1 / sqrt(sigma[i])
        end

        # U_new = R_col * V_svd[:,1:k] * diag(1/√σ)
        V_k = F.V[:, 1:k]  # (b × k)
        U_new = R_col * V_k  # (m × k)
        @inbounds for i in 1:k
            U_new[:, i] .*= sqrt_sigma_inv[i]
        end

        # V_new = diag(1/√σ) * U_svd[:,1:k]' * R_row
        U_k = F.U[:, 1:k]  # (b × k)
        V_new = U_k' * R_row  # (k × n)
        @inbounds for i in 1:k
            V_new[i, :] .*= sqrt_sigma_inv[i]
        end

        # ── Step 8: Append to U, V and update F-norm ────────────────────────
        for i in 1:k
            npivot += 1
            if npivot > maxrank
                npivot -= 1
                break
            end
            U[1:m, npivot] = U_new[:, i]
            V[npivot, 1:n] = V_new[i, :]

            # Incremental F-norm² update (mirrors normF! from standard ACA)
            u_col = view(U, 1:m, npivot)
            v_row = view(V, npivot, 1:n)
            rnorm = norm(u_col)
            cnorm = norm(v_row)

            if !isapprox(rnorm, zero(real(K))) && !isapprox(cnorm, zero(real(K)))
                normUV² += (rnorm * cnorm)^2

                # Cross terms with all previous pivots
                for j in 1:(npivot - 1)
                    normUV² +=
                        2 * real(
                            dot(view(U, 1:m, npivot), view(U, 1:m, j)) *
                            dot(view(V, npivot, 1:n), view(V, j, 1:n)),
                        )
                end
            end
        end

        # ── Step 9: Check convergence ───────────────────────────────────────
        # F-norm criterion: continue while the last batch contribution
        # exceeds tolerance relative to the total F-norm estimate.
        batch_contribution = zero(real(K))
        @inbounds for i in 1:k
            batch_contribution += abs2(sigma[i])
        end

        if normUV² > zero(real(K))
            conv = sqrt(batch_contribution) > tol * sqrt(normUV²)
        else
            conv = true
        end
    end

    return npivot
end

# Convergence checking is now integrated into the main loop via
# incremental normUV² updates, matching the standard ACA pattern.

# tolerance() is defined in convergence/estimation.jl and convergence/extrapolation.jl

"""
    batched_aca(M; tol=1e-4, rowpivoting, colpivoting, maxrank=40, batchsize=4)

Compute batched adaptive cross approximation of matrix `M` returning low-rank factors.
High-level convenience function that automatically allocates buffers and returns
`U, V` such that `M ≈ U * V`.
"""
function batched_aca(
    M::AbstractMatrix{K};
    tol=1e-4,
    rowpivoting::BatchedPivStrat,
    colpivoting::BatchedPivStrat,
    maxrank::Int=40,
    batchsize::Int=4,
) where {K}
    compressor = BatchedACA(rowpivoting, colpivoting, FNormEstimator(tol))
    rowbuffer = zeros(K, maxrank, size(M, 2))
    colbuffer = zeros(K, size(M, 1), maxrank)
    npivots = compressor(M, colbuffer, rowbuffer, maxrank)
    return colbuffer[1:size(M, 1), 1:npivots], rowbuffer[1:npivots, 1:size(M, 2)]
end
