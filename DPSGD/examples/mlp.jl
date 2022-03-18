using
    CUDA,
    Flux,
    MLDatasets,
    Statistics

using Flux: onehotbatch, onecold, logitcrossentropy, logitcrossentropy

using DPSGD

include("Data.jl")
using .Data

if has_cuda()
    @info "Training on CUDA GPU"
    CUDA.allowscalar(false)
end

function mnist(; batchsize = 32)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Flatten each image into a linear array.
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    ytrain = onehotbatch(ytrain, 0:9)
    ytest = onehotbatch(ytest, 0:9)

    train_data = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true,
        datasampler = VariableBatchSizeUniformSampleWithReplacement)
    test_data = DataLoader((xtest, ytest), batchsize = batchsize)

    return train_data, test_data
end

function make_model(; imgsize = (28, 28, 1), nclasses = 10)
    return Chain(
        Dense(prod(imgsize), 64, relu),
        Dense(64, nclasses))
end

function accuracy(model, data_loader)
    acc = 0
    num = 0
    for (x, y) in data_loader
        x, y = gpu(x), y
        acc += sum(onecold(cpu(model(x))) .== onecold(y))
        num += size(x, 2)
    end
    return acc / num
end

function train(; η = 3e-4, nepochs = 1000)
    train_data, test_data = mnist()

    mlp = make_model() |> gpu

    # Use `agg=identity` to not reduce losses over examples
    loss(x, y) = logitcrossentropy(mlp(x), y; agg = identity)

    target_delta = 1e-5
    nexamples = train_data.nobs
    privacy_opt = DifferentialPrivacy(nexamples, lotsize = train_data.batchsize)
    opt = Flux.Optimiser(privacy_opt, Descent(η))

    for epoch in 1:nepochs
        Flux.train!(loss, params(mlp), ((gpu(x), gpu(y)) for (x, y) in train_data), opt)
        @show epoch
        @show privacy_opt.niterations
        @show accuracy(mlp, test_data)
        @show privacy_spent(privacy_opt, target_delta)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
