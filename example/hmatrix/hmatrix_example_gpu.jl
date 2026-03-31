using BEAST: DoubleNumSauterQstrat, DoubleNumQStrat
using AdaptiveCrossApproximation
using BEAST
using CUDA
using BlockSparseMatrices
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using LinearAlgebra
using LinearMaps
using OhMyThreads

include("skeletons.jl")
include("hmatrix.jl")

##

λ = 2.0
k = 2π / λ
Γ = meshsphere(1.0, 0.1)

op = Maxwell3D.singlelayer(; wavenumber=k)
X = raviartthomas(Γ)

ttree = H2Trees.KMeansTree(X.pos, 2; minvalues=50)
tree = BlockTree(ttree, ttree)

hmat = HMatrix(
    op,
    X,
    X,
    tree;
    nearquadstrat=DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6),
    farquadstrat=DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1),
    gpu=false,
    compressor=ACA(),
)

hmat_gpu = HMatrix(
    op,
    X,
    X,
    tree;
    nearquadstrat=DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6),
    farquadstrat=DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1),
    gpu=true,
)

##
x = rand(ComplexF64, length(X))

println("\n=== Matrix-vector product timings ===")
hmat * x  # warmup
hmat_gpu * x  # warmup

print("CPU hmat:  ");
@time y_cpu = hmat * x;
print("GPU hmat:  ");
@time y_gpu = hmat_gpu * x;

println("\n=== Accuracy vs full assembly ===")
A = assemble(op, X, X)
y_ref = A * x

println("CPU hmat error:  ", norm(y_cpu - y_ref) / norm(y_ref))
println("GPU hmat error:  ", norm(y_gpu - y_ref) / norm(y_ref))
println("CPU vs GPU diff: ", norm(y_cpu - y_gpu) / norm(y_ref))

println(first(hmat.nearinteractions), last(hmat.nearinteractions))
println(first(hmat_gpu.nearinteractions), last(hmat_gpu.nearinteractions))
println(first(hmat.farinteractions), last(hmat.farinteractions))
println(first(hmat_gpu.farinteractions), last(hmat_gpu.farinteractions))
