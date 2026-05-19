using Test
using CUDA
using LinearAlgebra
using StaticArrays: SVector
using BEAST: Helmholtz3D, lagrangec0d1, scalartype
using CompScienceMeshes: meshsphere, translate
using H2Trees
using OhMyThreads: SerialScheduler
using AdaptiveCrossApproximation

@testset "HMatrix with GPU compressor" begin
    if !CUDA.functional()
        @warn "CUDA not functional; skipping HMatrix GPU test."
        return nothing
    end
    ext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDAExt)
    if ext === nothing
        @warn "ACACUDAExt not loaded; skipping HMatrix GPU test."
        return nothing
    end

    res = 0.3
    sphere = meshsphere(1.0, res)
    op = Helmholtz3D.singlelayer(; wavenumber=1.0)
    X = lagrangec0d1(sphere)

    T = scalartype(op)
    m = length(X)
    @info "HMatrix GPU test: $m DOFs, res=$res, op=$(typeof(op))"

    K = AdaptiveCrossApproximation.AbstractKernelMatrix(op, X, X)
    A_dense = zeros(T, m, m)
    for i in 1:m, j in 1:m
        K(view(A_dense, i:i, j:j), [i], [j])
    end
    A_norm = norm(A_dense)
    @test A_norm > 0

    tree = TwoNTree(X.pos, X.pos, 1 / 2^10; testminvalues=40, trialminvalues=40)
    tol = 1e-4
    maxrank = 40
    ordering = AdaptiveCrossApproximation.PreserveSpaceOrder()
    scheduler = SerialScheduler()

    @testset "CPU HMatrix baseline" begin
        hmat_cpu = HMatrix(
            op,
            X,
            X,
            tree;
            tol=tol,
            maxrank=maxrank,
            compressor=ACA(; tol=tol),
            spaceordering=ordering, #required
            scheduler=scheduler,
        )
        err_dense = norm(Matrix(hmat_cpu) - A_dense) / A_norm
        @info "  CPU HMatrix vs dense rel_err = $(round(err_dense; digits=6))"
        @test err_dense < 1e-2

        x = rand(T, m)
        y_cpu = hmat_cpu * x
        y_ref = A_dense * x
        @test norm(y_cpu - y_ref) / norm(y_ref) < 1e-2
    end

    for bs in (1, 5)
        @testset "GPU HMatrix batchsize=$(bs)" begin
            gpu_comp = ext.GPUCompressor(
                op, X, X; tol=tol, maxrank=maxrank, batchsize=bs, kernel=:pair_scatter
            )

            hmat_gpu = HMatrix(
                op,
                X,
                X,
                tree;
                tol=tol,
                maxrank=maxrank,
                compressor=gpu_comp,
                spaceordering=ordering,
                scheduler=scheduler,
            )

            err_dense = norm(Matrix(hmat_gpu) - A_dense) / A_norm
            @info "  GPU HMatrix bs=$bs vs dense rel_err = $(round(err_dense; digits=6))"
            @test err_dense < 1e-2

            x = rand(T, m)
            y_gpu = hmat_gpu * x
            y_ref = A_dense * x
            @test norm(y_gpu - y_ref) / norm(y_ref) < 1e-2
        end
    end
end
