using Test
using CUDA
using SparseArrays

# Include the kernel definitions from the extension
include("../../ext/ACACUDAExt/index_sparse.jl")

@testset "GPU Sparse Matrix Indexing" begin
    if !CUDA.functional()
        @warn "CUDA is not functional on this machine; skipping GPU sparse index tests."
        return nothing
    end

    # ── Build a small sparse matrix with known structure ──
    # Matrix:
    #   [1.0  0.0  2.0  0.0]
    #   [0.0  3.0  0.0  4.0]
    #   [5.0  0.0  6.0  0.0]
    #   [0.0  7.0  0.0  8.0]
    rows_cpu = [1, 1, 2, 2, 3, 3, 4, 4]
    cols_cpu = [1, 3, 2, 4, 1, 3, 2, 4]
    vals_cpu = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sparse_csc_cpu = sparse(rows_cpu, cols_cpu, vals_cpu, 4, 4)

    # ── CSC format test ──
    @testset "CSC kernel_index!" begin
        # Convert to GPU CSC format (CUDA convention: rowInd 0-based, colPtr 1-based)
        sparse_csc_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse_csc_cpu)

        # Extract internal arrays from the GPU sparse matrix
        # CuSparseMatrixCSC stores: rowInd (0-based), colPtr (1-based), nzVal
        d_row_indices = sparse_csc_gpu.rowInd  # 0-based row indices
        d_col_ptrs    = sparse_csc_gpu.colPtr  # 1-based column pointers
        d_val         = sparse_csc_gpu.nzVal   # non-zero values

        # Index arrays: we want rows [1, 3] and columns [2, 4] (1-based Julia indices)
        # On GPU, a[i] is 1-based row index (kernel subtracts 1 internally for 0-based comparison)
        # On GPU, b[j] is 1-based column index (used directly to index colPtr)
        a_cpu = [1, 3]   # row indices (1-based)
        b_cpu = [2, 4]   # column indices (1-based)
        na = length(a_cpu)
        nb = length(b_cpu)

        d_a = CUDA.CuArray(a_cpu)
        d_b = CUDA.CuArray(b_cpu)
        d_result = CUDA.CuMatrix{Float64}(undef, na, nb)

        threads = 256
        blocks = cld(na, threads)

        @cuda threads = threads blocks = blocks kernel_index!(
            d_result, d_row_indices, d_col_ptrs, d_val, d_a, d_b, na, nb
        )
        CUDA.synchronize()

        result_gpu = Array(d_result)

        # Expected: result[i,j] = sparse_csc_cpu[a[i], b[j]]
        #   (1,2) -> 0.0,  (1,4) -> 0.0
        #   (3,2) -> 0.0,  (3,4) -> 0.0
        # All entries at these row/col combinations are zero in our matrix
        expected = zeros(Float64, na, nb)
        for i in 1:na, j in 1:nb
            expected[i, j] = sparse_csc_cpu[a_cpu[i], b_cpu[j]]
        end

        @test result_gpu ≈ expected atol = 1e-10

        # Now test with indices that hit non-zero entries
        # rows [1, 2, 3, 4] and columns [1, 3] (1-based)
        a_cpu2 = [1, 2, 3, 4]
        b_cpu2 = [1, 3]
        na2 = length(a_cpu2)
        nb2 = length(b_cpu2)

        d_a2 = CUDA.CuArray(a_cpu2)
        d_b2 = CUDA.CuArray(b_cpu2)
        d_result2 = CUDA.CuMatrix{Float64}(undef, na2, nb2)

        blocks2 = cld(na2, threads)

        @cuda threads = threads blocks = blocks2 kernel_index!(
            d_result2, d_row_indices, d_col_ptrs, d_val, d_a2, d_b2, na2, nb2
        )
        CUDA.synchronize()

        result_gpu2 = Array(d_result2)

        expected2 = zeros(Float64, na2, nb2)
        for i in 1:na2, j in 1:nb2
            expected2[i, j] = sparse_csc_cpu[a_cpu2[i], b_cpu2[j]]
        end

        @test result_gpu2 ≈ expected2 atol = 1e-10
    end

    # ── CSR format test ──
    @testset "CSR kernel_index!" begin
        # Convert to GPU CSR format (CUDA convention: rowPtr 1-based, colInd 0-based)
        sparse_csr_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse_csc_cpu)

        # Extract internal arrays from the GPU sparse matrix
        # CuSparseMatrixCSR stores: rowPtr (1-based), colInd (0-based), nzVal
        d_row_ptrs    = sparse_csr_gpu.rowPtr  # 1-based row pointers
        d_col_indices = sparse_csr_gpu.colInd  # 0-based column indices
        d_val         = sparse_csr_gpu.nzVal   # non-zero values

        # Index arrays: we want rows [1, 3] and columns [2, 4] (1-based Julia indices)
        # On GPU, a[i] is 1-based row index (used directly to index rowPtr)
        # On GPU, b[j] is 1-based column index (kernel subtracts 1 internally for 0-based comparison)
        a_cpu = [1, 3]   # row indices (1-based)
        b_cpu = [2, 4]   # column indices (1-based)
        na = length(a_cpu)
        nb = length(b_cpu)

        d_a = CUDA.CuArray(a_cpu)
        d_b = CUDA.CuArray(b_cpu)
        d_result = CUDA.CuMatrix{Float64}(undef, na, nb)

        threads = 256
        blocks = cld(na, threads)

        @cuda threads = threads blocks = blocks kernel_index!(
            d_result, d_row_ptrs, d_col_indices, d_val, d_a, d_b, na, nb
        )
        CUDA.synchronize()

        result_gpu = Array(d_result)

        # Expected: result[i,j] = sparse_csc_cpu[a[i], b[j]]
        expected = zeros(Float64, na, nb)
        for i in 1:na, j in 1:nb
            expected[i, j] = sparse_csc_cpu[a_cpu[i], b_cpu[j]]
        end

        @test result_gpu ≈ expected atol = 1e-10

        # Now test with indices that hit non-zero entries
        # rows [1, 2, 3, 4] and columns [1, 3] (1-based)
        a_cpu2 = [1, 2, 3, 4]
        b_cpu2 = [1, 3]
        na2 = length(a_cpu2)
        nb2 = length(b_cpu2)

        d_a2 = CUDA.CuArray(a_cpu2)
        d_b2 = CUDA.CuArray(b_cpu2)
        d_result2 = CUDA.CuMatrix{Float64}(undef, na2, nb2)

        blocks2 = cld(na2, threads)

        @cuda threads = threads blocks = blocks2 kernel_index!(
            d_result2, d_row_ptrs, d_col_indices, d_val, d_a2, d_b2, na2, nb2
        )
        CUDA.synchronize()

        result_gpu2 = Array(d_result2)

        expected2 = zeros(Float64, na2, nb2)
        for i in 1:na2, j in 1:nb2
            expected2[i, j] = sparse_csc_cpu[a_cpu2[i], b_cpu2[j]]
        end

        @test result_gpu2 ≈ expected2 atol = 1e-10
    end

    # ── Larger random sparse matrix test ──
    @testset "Random sparse matrix (CSC + CSR)" begin
        n = 64
        density = 0.15
        sparse_rand_cpu = sparse(
            [rand(1:n) for _ in 1:round(Int, n * n * density)],
            [rand(1:n) for _ in 1:round(Int, n * n * density)],
            randn(round(Int, n * n * density)),
            n,
            n,
        )

        # Pick a subset of rows and columns
        a_cpu = sort(unique(rand(1:n, 12)))
        b_cpu = sort(unique(rand(1:n, 10)))
        na = length(a_cpu)
        nb = length(b_cpu)

        sparse_csc_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse_rand_cpu)
        result_dev = CUDA.CuMatrix{Float64}(undef, na, nb)

        threads = 256
        blocks = cld(na, threads)

        @cuda threads = threads blocks = blocks kernel_index!(
            result_dev,
            sparse_csc_gpu.rowInd,
            sparse_csc_gpu.colPtr,
            sparse_csc_gpu.nzVal,
            CUDA.CuArray(a_cpu),
            CUDA.CuArray(b_cpu),
            na,
            nb,
        )
        CUDA.synchronize()

        result_csc = Array(d_result_csc)
        expected_csc = zeros(Float64, na, nb)
        for i in 1:na, j in 1:nb
            expected_csc[i, j] = sparse_rand_cpu[a_cpu[i], b_cpu[j]]
        end
        @test result_csc ≈ expected_csc atol = 1e-10

        # ── CSR ──
        sparse_csr_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse_rand_cpu)
        d_result_csr = CUDA.CuMatrix{Float64}(undef, na, nb)

        @cuda threads = threads blocks = blocks kernel_index!(
            d_result_csr,
            sparse_csr_gpu.rowPtr,
            sparse_csr_gpu.colInd,
            sparse_csr_gpu.nzVal,
            CUDA.CuArray(a_cpu),
            CUDA.CuArray(b_cpu),
            na,
            nb,
        )
        CUDA.synchronize()

        result_csr = Array(d_result_csr)
        expected_csr = zeros(Float64, na, nb)
        for i in 1:na, j in 1:nb
            expected_csr[i, j] = sparse_rand_cpu[a_cpu[i], b_cpu[j]]
        end
        @test result_csr ≈ expected_csr atol = 1e-10
    end
end
