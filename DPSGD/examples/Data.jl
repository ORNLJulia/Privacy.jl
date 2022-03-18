# Adapted from Knet's src/data.jl (author: Deniz Yuret)
module Data

export
    DataLoader,
    VariableBatchSizeUniformSampleWithReplacement

using
    Random

using Base: @propagate_inbounds
using Random: GLOBAL_RNG
struct DataLoader{S,D,R<:AbstractRNG}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle::Bool
    rng::R
    datasampler::S
end

function UniformSampleWithoutReplacement(d::DataLoader, i::Integer)
    if d.shuffle && i == 0
        shuffle!(d.rng, d.indices)
        @info "Using UniformSampleWithoutReplacement"
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i + 1:nexti]
    return (ids, nexti)
end

function UniformSampleWithReplacement(d::DataLoader, i::Integer)
    if d.shuffle && i == 0
        # only thing different from above... should probably combine
        rand!(d.rng, d.indices, [1:d.nobs;]) 
        @info "Using UniformSampleWithReplacement"
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i + 1:nexti]
    return (ids, nexti)
end

function VariableBatchSizeUniformSampleWithReplacement(d::DataLoader, i::Integer)
    if i == 0
        @info "Using VariableBatchSizeUniformSampleWithReplacement"
    end

    total_batches = d.nobs / d.batchsize
    sample_rate = float(d.batchsize) / float(d.nobs)

    ids = 0 # hack for scope, probably not Julian
    while true
        mask = rand(d.rng, Float32, size(d.indices)) .< sample_rate
        ids = d.indices[mask]
        length(ids) > 0 && break # ensure we have at least one observation per batch
    end
    # @show length(ids)
    nexti = i + 1

    if nexti >= total_batches
        nexti = d.imax # hacky
    end

    return (ids, nexti)
end

"""
    DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)

An object that iterates over mini-batches of `data`, each mini-batch containing
`batchsize` observations (except possibly the last one).

Takes as input a single data tensor, or a tuple (or a named tuple) of tensors.
The last dimension in each tensor is considered to be the observation dimension.

If `shuffle=true`, shuffles the observations each time iterations are re-started.
If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.

The original data is preserved in the `data` field of the DataLoader.

Usage example:

    Xtrain = rand(10, 100)
    train_loader = DataLoader(Xtrain, batchsize=2)
    # iterate over 50 mini-batches of size 2
    for x in train_loader
        @assert size(x) == (10, 2)
        ...
    end

    train_loader.data   # original dataset

    # similar, but yielding tuples
    train_loader = DataLoader((Xtrain,), batchsize=2)
    for (x,) in train_loader
        @assert size(x) == (10, 2)
        ...
    end

    Xtrain = rand(10, 100)
    Ytrain = rand(100)
    train_loader = DataLoader((Xtrain, Ytrain), batchsize=2, shuffle=true)
    for epoch in 1:100
        for (x, y) in train_loader
            @assert size(x) == (10, 2)
            @assert size(y) == (2,)
            ...
        end
    end

    # train for 10 epochs
    using IterTools: ncycle
    Flux.train!(loss, ps, ncycle(train_loader, 10), opt)

    # can use NamedTuple to name tensors
    train_loader = DataLoader((images=Xtrain, labels=Ytrain), batchsize=2, shuffle=true)
    for datum in train_loader
        @assert size(datum.images) == (10, 2)
        @assert size(datum.labels) == (2,)
    end
"""
function DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG,
                    datasampler=UniformSampleWithoutReplacement)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    n = _nobs(data)
    if n < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batchsize = n
    end
    imax = partial ? n : n - batchsize + 1
    DataLoader(data, batchsize, n, partial, imax, [1:n;], shuffle, rng, datasampler)
end

# Return data in d.indices[i+1:i+batchsize]
@propagate_inbounds function Base.iterate(d::DataLoader, i=0)
    i >= d.imax && return nothing
    # if d.shuffle && i == 0
    #     shuffle!(d.rng, d.indices)
    # end
    # nexti = min(i + d.batchsize, d.nobs)
    # ids = d.indices[i + 1:nexti]
    ids, nexti = d.datasampler(d, i)
    batch = _getobs(d.data, ids)
    return (batch, nexti)
end

function Base.length(d::DataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int, n) : floor(Int, n)
end

_nobs(data::AbstractArray) = size(data)[end]

function _nobs(data::Union{Tuple,NamedTuple})
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    n = _nobs(data[1])
    if !all(x -> _nobs(x) == n, Base.tail(data))
        throw(DimensionMismatch("All data should contain same number of observations"))
    end
    return n
end

_getobs(data::AbstractArray, i) = data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., i]
_getobs(data::Union{Tuple,NamedTuple}, i) = map(Base.Fix2(_getobs, i), data)

Base.eltype(::DataLoader{D}) where D = D

end
