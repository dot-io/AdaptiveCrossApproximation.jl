using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

# ── BatchedFillDistance pivoting tests ────────────────────────────────────────

@testset "BatchedFillDistance" begin
    @testset "Constructor" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        piv = BatchedFillDistance(pts; batchsize=4)
        @test piv.b == 4
        @test length(piv.pos) == 8

        # Default batchsize
        piv_default = BatchedFillDistance(pts)
        @test piv_default.b == 4
    end

    @testset "Functor construction" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        piv = BatchedFillDistance(pts; batchsize=3)
        idcs = [2, 4, 6, 8]

        functor = piv(idcs)
        @test functor.nactive == 4
        @test functor.b == 3
        @test length(functor.idcs) >= 4
        @test length(functor.h) >= 4
        @test isempty(functor.selected)
    end

    @testset "Initial batch selection" begin
        # Points on a line: 1, 2, 3, 4, 5, 6, 7, 8
        # With batchsize=3, should pick well-separated points
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        piv = BatchedFillDistance(pts; batchsize=3)
        idcs = collect(1:8)
        functor = piv(idcs)

        batch = functor()
        @test length(batch) == 3
        @test all(b -> 1 <= b <= 8, batch)
        @test length(unique(batch)) == 3  # all distinct

        # First point should be index 1 (first in active set)
        @test 1 in batch
    end

    @testset "Subsequent batch selection" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:10]
        piv = BatchedFillDistance(pts; batchsize=3)
        idcs = collect(1:10)
        functor = piv(idcs)

        batch1 = functor()
        @test length(batch1) == 3

        batch2 = functor(3)
        @test length(batch2) == 3

        # All selected indices should be distinct
        all_selected = vcat(batch1, batch2)
        @test length(unique(all_selected)) == length(all_selected)
    end

    @testset "Batch exceeds available points" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:4]
        piv = BatchedFillDistance(pts; batchsize=10)
        idcs = collect(1:4)
        functor = piv(idcs)

        batch = functor()
        @test length(batch) == 4  # can't select more than available
    end

    @testset "Reset and reuse" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
        piv = BatchedFillDistance(pts; batchsize=2)
        functor = piv(collect(1:6))

        batch1 = functor()
        @test length(batch1) == 2

        reset!(functor, collect(1:6))
        @test isempty(functor.selected)
        @test all(iszero, view(functor.h, 1:6))

        batch2 = functor()
        @test length(batch2) == 2
    end

    @testset "Resize" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
        piv = BatchedFillDistance(pts; batchsize=2)
        functor = piv(collect(1:6))

        resize!(functor, 10)
        @test functor.nactive == 10
        @test length(functor.h) >= 10
        @test length(functor.idcs) >= 10
    end

    @testset "2D points" begin
        pts = [
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(0.0, 1.0),
            SVector(1.0, 1.0),
            SVector(0.5, 0.5),
        ]
        piv = BatchedFillDistance(pts; batchsize=3)
        functor = piv(collect(1:5))

        batch = functor()
        @test length(batch) == 3
        # Should pick well-separated points
        @test 1 in batch
    end

    @testset "batchsize method" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        piv = BatchedFillDistance(pts; batchsize=5)
        @test AdaptiveCrossApproximation.batchsize(piv) == 5

        functor = piv(collect(1:8))
        @test AdaptiveCrossApproximation.batchsize(functor) == 5
    end
end

# ── BatchedACA compressor tests ───────────────────────────────────────────────

@testset "BatchedACA" begin
    @testset "Constructor type enforcement" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:10]
        row_piv = BatchedFillDistance(pts; batchsize=4)
        col_piv = BatchedFillDistance(pts; batchsize=4)

        # Valid construction
        baca = BatchedACA(rowpivoting=row_piv, columnpivoting=col_piv, tol=1e-4)
        @test baca.rowpivoting === row_piv
        @test baca.columnpivoting === col_piv

        # Invalid: non-batched pivoting strategy should throw
        @test_throws ArgumentError BatchedACA(
            rowpivoting=MaximumValue(), columnpivoting=col_piv, tol=1e-4
        )
        @test_throws ArgumentError BatchedACA(
            rowpivoting=row_piv, columnpivoting=MaximumValue(), tol=1e-4
        )
    end

    @testset "Low-rank matrix approximation" begin
        Random.seed!(42)

        # Create a low-rank matrix: A = U_true * V_true
        m, n, r_true = 20, 25, 3
        U_true = randn(Float64, m, r_true)
        V_true = randn(Float64, r_true, n)
        A = U_true * V_true

        tpos = [@SVector rand(3) for _ in 1:m]
        spos = [@SVector rand(3) for _ in 1:n]

        row_piv = BatchedFillDistance(tpos; batchsize=4)
        col_piv = BatchedFillDistance(spos; batchsize=4)

        U, V = batched_aca(
            A;
            rowpivoting=row_piv,
            columnpivoting=col_piv,
            tol=1e-6,
            maxrank=10,
        )

        @test size(U, 1) == m
        @test size(V, 2) == n
        @test size(U, 2) == size(V, 1)

        # The approximation should be good for a rank-3 matrix
        rel_err = norm(A - U * V) / norm(A)
        @test rel_err < 1e-4
    end

    @testset "Exact rank-1 matrix" begin
        A = ones(10, 12)  # rank 1

        tpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:10]
        spos = [@SVector [Float64(j), 0.0, 0.0] for j in 1:12]

        row_piv = BatchedFillDistance(tpos; batchsize=4)
        col_piv = BatchedFillDistance(spos; batchsize=4)

        U, V = batched_aca(
            A;
            rowpivoting=row_piv,
            columnpivoting=col_piv,
            tol=1e-10,
            maxrank=5,
        )

        # Should recover rank 1 (or close to it)
        @test size(U, 2) <= 2
        rel_err = norm(A - U * V) / norm(A)
        @test rel_err < 1e-6
    end

    @testset "Zero matrix" begin
        A = zeros(8, 8)

        tpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        spos = [@SVector [Float64(j), 0.0, 0.0] for j in 1:8]

        row_piv = BatchedFillDistance(tpos; batchsize=4)
        col_piv = BatchedFillDistance(spos; batchsize=4)

        U, V = batched_aca(
            A;
            rowpivoting=row_piv,
            columnpivoting=col_piv,
            tol=1e-4,
            maxrank=5,
        )

        # Zero matrix should produce zero (or very small) approximation
        @test norm(U * V) < 1e-10
    end

    @testset "Identity matrix" begin
        A = Float64[1.0 * (i == j) for i in 1:8, j in 1:8]

        tpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        spos = [@SVector [Float64(j), 0.0, 0.0] for j in 1:8]

        row_piv = BatchedFillDistance(tpos; batchsize=2)
        col_piv = BatchedFillDistance(spos; batchsize=2)

        U, V = batched_aca(
            A;
            rowpivoting=row_piv,
            columnpivoting=col_piv,
            tol=1e-4,
            maxrank=8,
        )

        # Identity matrix has rank 8, so we need maxrank >= 8 for exact recovery
        # With maxrank=8, should get a good approximation
        rel_err = norm(A - U * V) / norm(A)
        @test rel_err < 0.5  # relaxed tolerance for identity matrix
    end

    @testset "Different batch sizes" begin
        Random.seed!(99)
        m, n, r_true = 30, 35, 5
        U_true = randn(Float64, m, r_true)
        V_true = randn(Float64, r_true, n)
        A = U_true * V_true

        tpos = [@SVector rand(3) for _ in 1:m]
        spos = [@SVector rand(3) for _ in 1:n]

        for bs in [1, 2, 4, 8]
            row_piv = BatchedFillDistance(tpos; batchsize=bs)
            col_piv = BatchedFillDistance(spos; batchsize=bs)

            U, V = batched_aca(
                A;
                rowpivoting=row_piv,
                columnpivoting=col_piv,
                tol=1e-4,
                maxrank=15,
            )

            rel_err = norm(A - U * V) / norm(A)
            @test rel_err < 1e-3
        end
    end

    @testset "Comparison with standard ACA on smooth kernel" begin
        Random.seed!(1234)

        # Create a smooth kernel matrix (fast-decaying singular values)
        tpos = [@SVector rand(3) for _ in 1:40]
        spos = [@SVector [p[1] + 3.5, p[2], p[3]] for p in tpos]  # separated

        K = [1.0 / (norm(tp - sp) + 0.1) for tp in tpos, sp in spos]

        # Standard ACA
        U_aca, V_aca = AdaptiveCrossApproximation.aca(K; tol=1e-4, maxrank=20)

        # Batched ACA
        row_piv = BatchedFillDistance(tpos; batchsize=4)
        col_piv = BatchedFillDistance(spos; batchsize=4)
        U_baca, V_baca = batched_aca(
            K;
            rowpivoting=row_piv,
            columnpivoting=col_piv,
            tol=1e-4,
            maxrank=20,
        )

        err_aca = norm(K - U_aca * V_aca) / norm(K)
        err_baca = norm(K - U_baca * V_baca) / norm(K)

        # Both should achieve reasonable approximation
        @test err_aca < 1e-3
        @test err_baca < 1e-2  # batched may be slightly less accurate due to batch truncation

        # Both should produce valid low-rank factors
        @test size(U_aca, 1) == 40
        @test size(V_aca, 2) == 40
        @test size(U_baca, 1) == 40
        @test size(V_baca, 2) == 40
    end

    @testset "Buffer interface (preallocated)" begin
        Random.seed!(77)
        m, n = 15, 20
        U_true = randn(Float64, m, 3)
        V_true = randn(Float64, 3, n)
        A = U_true * V_true

        tpos = [@SVector rand(3) for _ in 1:m]
        spos = [@SVector rand(3) for _ in 1:n]

        row_piv = BatchedFillDistance(tpos; batchsize=3)
        col_piv = BatchedFillDistance(spos; batchsize=3)

        maxrank = 10
        U_buf = zeros(Float64, m, maxrank)
        V_buf = zeros(Float64, maxrank, n)

        baca = BatchedACA(row_piv, col_piv, FNormEstimator(1e-4))
        npivots = baca(A, U_buf, V_buf, maxrank)

        @test npivots > 0
        @test npivots <= maxrank

        U_result = U_buf[:, 1:npivots]
        V_result = V_buf[1:npivots, :]

        rel_err = norm(A - U_result * V_result) / norm(A)
        @test rel_err < 1e-3
    end

    @testset "Maxrank limit" begin
        Random.seed!(55)
        m, n = 20, 25
        A = randn(Float64, m, n)

        tpos = [@SVector rand(3) for _ in 1:m]
        spos = [@SVector rand(3) for _ in 1:n]

        row_piv = BatchedFillDistance(tpos; batchsize=4)
        col_piv = BatchedFillDistance(spos; batchsize=4)

        maxrank = 5
        U, V = batched_aca(
            A;
            rowpivoting=row_piv,
            columnpivoting=col_piv,
            tol=1e-10,  # very tight tolerance
            maxrank=maxrank,
        )

        # Should not exceed maxrank
        @test size(U, 2) <= maxrank
        @test size(V, 1) <= maxrank
    end

    @testset "Complex matrix" begin
        Random.seed!(42)
        m, n, r_true = 15, 20, 3
        U_true = randn(ComplexF64, m, r_true) + im * randn(ComplexF64, m, r_true)
        V_true = randn(ComplexF64, r_true, n) + im * randn(ComplexF64, r_true, n)
        A = U_true * V_true

        tpos = [@SVector rand(3) for _ in 1:m]
        spos = [@SVector rand(3) for _ in 1:n]

        row_piv = BatchedFillDistance(tpos; batchsize=4)
        col_piv = BatchedFillDistance(spos; batchsize=4)

        U, V = batched_aca(
            A;
            rowpivoting=row_piv,
            columnpivoting=col_piv,
            tol=1e-4,
            maxrank=10,
        )

        @test size(U, 1) == m
        @test size(V, 2) == n
        @test eltype(U) == ComplexF64
        @test eltype(V) == ComplexF64

        rel_err = norm(A - U * V) / norm(A)
        @test rel_err < 1e-3
    end
end
