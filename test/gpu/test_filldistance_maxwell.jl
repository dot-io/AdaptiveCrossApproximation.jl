using Test
using LinearAlgebra
using StaticArrays: SVector
using BEAST: Helmholtz3D, Maxwell3D, lagrangec0d1, raviartthomas, scalartype
using CompScienceMeshes: meshsphere, translate
using AdaptiveCrossApproximation
using AdaptiveCrossApproximation: MaximumValue, FillDistance, FNormEstimator
"""
i wrote this test to check whether the filldistance criterion leads to a
large error for raviart-thomas bases. i first believed that my gpu implemen-
tation provided a wrong result for this basis specifically so testing it
on cpu allows to check for this.
turns out that this error is normal.
"""
const FAR_OFFSET = SVector(4.0, 0.0, 0.0)
const RES = 0.3
const TOL = 1e-4
const MAXRANK = 40

function dense_block(op, X, Y)
    T = scalartype(op)
    m = length(X)
    n = length(Y)
    K = AdaptiveCrossApproximation.AbstractKernelMatrix(op, X, Y)
    A = zeros(T, m, n)
    for i in 1:m, j in 1:n
        K(view(A, i:i, j:j), [i], [j])
    end
    return A
end

function run_aca(A, rowpiv, colpiv)
    U, V = AdaptiveCrossApproximation.aca(
        A;
        rowpivoting=rowpiv,
        columnpivoting=colpiv,
        convergence=FNormEstimator(TOL),
        maxrank=MAXRANK,
    )
    rel = norm(A - U * V) / norm(A)
    return rel, size(U, 2)
end

@testset "FillDistance pivoting on RT basis (CPU diagnostic)" begin
    sphere       = meshsphere(1.0, RES)
    sphere_far   = translate(sphere, FAR_OFFSET)

    @testset "Helmholtz3D / lagrangec0d1 (control)" begin
        op = Helmholtz3D.singlelayer(; wavenumber=1.0)
        X  = lagrangec0d1(sphere)
        Y  = lagrangec0d1(sphere_far)
        A  = dense_block(op, X, Y)

        err_mv, r_mv = run_aca(A, MaximumValue(), MaximumValue())
        err_fd, r_fd = run_aca(A, FillDistance(X.pos), FillDistance(Y.pos))

        @info "  Helmholtz/Lagrange  MV: rel_err=$(round(err_mv; sigdigits=4))  r=$r_mv"
        @info "  Helmholtz/Lagrange  FD: rel_err=$(round(err_fd; sigdigits=4))  r=$r_fd"

        # Both strategies should converge well for a scalar smooth kernel.
        @test err_mv < 1e-2
        @test err_fd < 1e-2
    end

    @testset "Maxwell3D / raviartthomas (suspect)" begin
        op = Maxwell3D.singlelayer(; wavenumber=5.0)
        X  = raviartthomas(sphere)
        Y  = raviartthomas(sphere_far)
        A  = dense_block(op, X, Y)

        err_mv, r_mv = run_aca(A, MaximumValue(), MaximumValue())
        err_fd, r_fd = run_aca(A, FillDistance(X.pos), FillDistance(Y.pos))

        @info "  Maxwell/RT          MV: rel_err=$(round(err_mv; sigdigits=4))  r=$r_mv"
        @info "  Maxwell/RT          FD: rel_err=$(round(err_fd; sigdigits=4))  r=$r_fd"

        @test err_mv < 1e-2

        @test err_fd > 2 * err_mv
    end
end
