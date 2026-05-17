"""
    BatchedFillDistance{D,F<:Real} <: BatchedPivStrat

Select batches of `b` indices at once by iteratively choosing the point that
maximizes the minimum distance to all previously selected points.
batched approach enables gpu-friendly processing because multiple rows/cols are
assembled in 1 iteration which improves throughput.

# Notes

Fill distance pivoting can be seenaas a heuristic for maximizing the volume of the intersection
submatrix in CUR decomposition. By uniformly covering the geometric domain, the
selected rows/columns tend to capture the dominant singular vectors of the block.
"""
struct BatchedFillDistance{D,F<:Real} <: BatchedPivStrat
    pos::Vector{SVector{D,F}}
    b::Int
end

"""
    BatchedFillDistance(pos; batchsize=4)

Construct a BatchedFillDistance strategy from position data.
"""
function BatchedFillDistance(pos::Vector{SVector{D,F}}; batchsize::Int=4) where {D,F<:Real}
    return BatchedFillDistance{D,F}(pos, batchsize)
end
#TODO doc
mutable struct BatchedFillDistanceFunctor{D,F<:Real} <: BatchedPivStratFunctor
    pivoting::BatchedFillDistance{D,F}
    nactive::Int
    b::Int                # current batch size (can be adapted)
    idcs::Vector{Int}     # active index set (local → global mapping)
    selected::Vector{Int} # indices already selected (local into idcs)
    h::Vector{F}          # fill distance for each active point
end
#TODO docj
function (pivstrat::BatchedFillDistance{D,F})(
    idcs::AbstractVector{<:Integer}, batchsize::Int=pivstrat.b
) where {D,F<:Real}
    nactive = length(idcs)
    return BatchedFillDistanceFunctor{D,F}(
        pivstrat, nactive, batchsize, collect(Int, idcs), Int[], zeros(F, nactive)
    )
end
#TODO doc
function Base.resize!(
    pivstrat::BatchedFillDistanceFunctor{D,F}, nactive::Int
) where {D,F<:Real}
    length(pivstrat.h) < nactive && resize!(pivstrat.h, nactive)
    length(pivstrat.idcs) < nactive && resize!(pivstrat.idcs, nactive)
    pivstrat.nactive = nactive
    return nothing
end
#TODO doc
function reset!(
    pivstrat::BatchedFillDistanceFunctor{D,F}, idcs::AbstractVector{<:Integer}
) where {D,F<:Real}
    nactive = length(idcs)
    resize!(pivstrat, nactive)
    @inbounds for i in 1:nactive
        pivstrat.idcs[i] = Int(idcs[i])
    end
    empty!(pivstrat.selected)
    fill!(view(pivstrat.h, 1:nactive), zero(F))
    return nothing
end
#TODO doc
function batchsize(pivstrat::BatchedFillDistanceFunctor)
    return pivstrat.b
end

"""
    (pivstrat::BatchedFillDistanceFunctor)()

Select the initial batch of indices.
"""
function (pivstrat::BatchedFillDistanceFunctor{D,F})() where {D,F<:Real}
    nactive = pivstrat.nactive
    b = min(pivstrat.b, nactive)
    pos = pivstrat.pivoting.pos

    # Start with the first index
    first_idx = 1
    push!(pivstrat.selected, first_idx)

    # Initialize fill distances from all points to the first selected point
    @inbounds for i in 1:nactive
        pivstrat.h[i] = norm(pos[pivstrat.idcs[i]] - pos[pivstrat.idcs[first_idx]])
    end

    # Greedily select remaining b-1 points
    batch = Int[first_idx]
    for _ in 2:b
        # Find the point with maximum fill distance (farthest from all selected)
        nextidx = 1
        maxdist = zero(F)
        @inbounds for i in 1:nactive
            if pivstrat.h[i] > maxdist
                maxdist = pivstrat.h[i]
                nextidx = i
            end
        end

        push!(batch, nextidx)
        push!(pivstrat.selected, nextidx)

        # Update fill distances
        @inbounds for i in 1:nactive
            d = norm(pos[pivstrat.idcs[i]] - pos[pivstrat.idcs[nextidx]])
            if d < pivstrat.h[i]
                pivstrat.h[i] = d
            end
        end
    end

    return batch
end

"""
    (pivstrat::BatchedFillDistanceFunctor)(npivot::Int)

Select the next batch of indices, given that `npivot` pivots have already been
selected in total.
"""
function (pivstrat::BatchedFillDistanceFunctor{D,F})(npivot::Int) where {D,F<:Real}
    nactive = pivstrat.nactive
    # Remaining indices to select from
    remaining = nactive - length(pivstrat.selected)
    b = min(pivstrat.b, remaining)

    if b <= 0
        return Int[]
    end

    pos = pivstrat.pivoting.pos

    # If no points selected yet (shouldn't happen since () handles init),
    # fall back to initial selection
    if isempty(pivstrat.selected)
        return pivstrat()
    end

    # The fill distances h[] are already maintained from previous selections.
    # Just greedily pick the next b points.
    batch = Int[]
    sizehint!(batch, b)

    for _ in 1:b
        nextidx = 1
        maxdist = zero(F)
        @inbounds for i in 1:nactive
            if pivstrat.h[i] > maxdist
                maxdist = pivstrat.h[i]
                nextidx = i
            end
        end

        # Safety: if all distances are zero, we can't select more
        maxdist == zero(F) && break

        push!(batch, nextidx)
        push!(pivstrat.selected, nextidx)

        # Update fill distances
        @inbounds for i in 1:nactive
            d = norm(pos[pivstrat.idcs[i]] - pos[pivstrat.idcs[nextidx]])
            if d < pivstrat.h[i]
                pivstrat.h[i] = d
            end
        end
    end

    return batch
end
