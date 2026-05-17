"""
    BatchedPivStrat <: PivStrat

Abstract base type for batched pivoting strategies that select multiple indices at once.
Subtypes should implement
(strat::ExampleBatchedPivStrat)(idcs::AbstractVector{Int}, batchsize::Int)
which creates a [`BatchedPivStratFunctor`](@ref) for the given index set and batch size.
"""
abstract type BatchedPivStrat <: PivStrat end

"""
    BatchedPivStratFunctor <: PivStratFunctor

Abstract base type for batched functors
subtypes should implement:
(functor::MyBatchedPivStratFunctor)() which selects an initial batch of indices
(functor::MyBatchedPivStratFunctor)(npivot::Int) which selects next batch of indices given
already picked ones
reset!(functor, idcs)
resize!(functor, nactive)
"""
abstract type BatchedPivStratFunctor <: PivStratFunctor end

"""
    batchsize(functor::BatchedPivStratFunctor)

Return the current batch size for functor
"""
function batchsize end
