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
using StaticArrays: SVector

include("skeletons.jl")
include("hmatrix.jl")
include("calculate_error.jl")

##

λ = 0.5
k = 2π / λ
Γ = meshsphere(1.0, 1.0)
Γ2 = translate(Γ, SVector{3}(0.0, 0.0, 4.0))

op = Maxwell3D.singlelayer(; wavenumber=k)
X = raviartthomas(Γ)

ttree = H2Trees.KMeansTree(X.pos, 2; minvalues=5)
tree = BlockTree(ttree, ttree)

hmat = HMatrix(
    op,
    X,
    X,
    tree;
    nearquadstrat=DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6),
    farquadstrat=DoubleNumSauterQstrat(2, 3, 1, 1, 1, 1),
    gpu=false,
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

println("Matrix-vector product timings")
hmat * x  # warmup
hmat_gpu * x  # warmup

print("CPU hmat:  ");
@time y_cpu = hmat * x;
print("GPU hmat:  ");
@time y_gpu = hmat_gpu * x;

println("Accuracy vs full assembly")
A = assemble(op, X, X)
y_ref = A * x

println("CPU hmat error:  ", norm(y_cpu - y_ref) / norm(y_ref))
println("GPU hmat error:  ", norm(y_gpu - y_ref) / norm(y_ref))
println("CPU vs GPU diff: ", norm(y_cpu - y_gpu) / norm(y_ref))

# println(first(hmat.nearinteractions.blocks), last(hmat.nearinteractions.blocks))
# println(first(hmat_gpu.nearinteractions.blocks), last(hmat_gpu.nearinteractions.blocks))
# println(hmat.nearinteractions, hmat_gpu.nearinteractions)
# println(first(first(hmat.farinteractions)), last(last(hmat.farinteractions)))
# println(first(first(hmat_gpu.farinteractions)), last(last(hmat_gpu.farinteractions)))
#estimate_norm(hmat.nearinteractions - hmat_gpu.nearinteractions)

for (level_cpu, level_gpu) in zip(hmat.farinteractions, hmat_gpu.farinteractions)
    for (tg_cpu, tg_gpu) in zip(level_cpu, level_gpu)
        for (blk_cpu, blk_gpu) in zip(tg_cpu, tg_gpu)
            println(
                "Block error (1 is best): ", estimate_reldifference(blk_cpu.M, blk_gpu.M)
            )
        end
    end
end
