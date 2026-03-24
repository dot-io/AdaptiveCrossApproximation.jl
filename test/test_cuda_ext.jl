using Test
using AdaptiveCrossApproximation
using CUDA
using LinearAlgebra

@testset "CUDA extension: ACA on CuArray" begin
    if !CUDA.functional()
        @info "CUDA is not functional on this machine; skipping CUDA extension tests."
        return nothing
    end

    Tlist = (Float32, Float64)

    for T in Tlist
        @testset "CuArray ACA ($T)" begin
            m, n = 48, 40
            rtrue = 4

            U0 = rand(T, m, rtrue)
            V0 = rand(T, rtrue, n)
            M_cpu = U0 * V0
            M_gpu = CuArray(M_cpu)

            tol = T(1e-6)
            maxrank = 10

            A_gpu, B_gpu = AdaptiveCrossApproximation.aca(
                M_gpu;
                tol=tol,
                rowpivoting=MaximumValue(),
                columnpivoting=MaximumValue(),
                convergence=FNormEstimator(tol),
                maxrank=maxrank,
                svdrecompress=false,
            )

            A = Array(A_gpu)
            B = Array(B_gpu)

            @test size(A, 1) == m
            @test size(B, 2) == n
            @test size(A, 2) == size(B, 1)
            @test size(A, 2) <= maxrank
            @test size(A, 2) >= 1

            relerr = norm(M_cpu - A * B) / max(norm(M_cpu), eps(T))
            @test relerr <= T(5e-3)
        end
    end
end

@testset "CUDA extension: unsupported strategy errors on CuArray path" begin
    if !CUDA.functional()
        @info "CUDA is not functional on this machine; skipping CUDA extension tests."
        return nothing
    end

    T = Float32
    m, n = 16, 12
    M = CuArray(rand(T, m, n))

    @test_throws ArgumentError aca(
        M;
        tol=T(1e-4),
        rowpivoting=MaximumValue(),
        columnpivoting=MaximumValue(),
        convergence=AdaptiveCrossApproximation.FNormExtrapolator(T(1e-4)),
        maxrank=6,
        svdrecompress=false,
    )
end
