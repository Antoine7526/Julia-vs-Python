using FileIO
using ImageTransformations, TestImages
using IterTools
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using MLDataUtils
using Flux: @epochs
using Printf


begin
    rug_dir = readdir("./data/rugby")
    soc_dir = readdir("./data/soccer")
end;

begin
    # we load the pre-proccessed images
    rug1 = load.("./data/rugby/" .* rug_dir)
    soc1 = load.("./data/soccer/" .* soc_dir)
end;

dataall = vcat(soc1, rug1);

begin
    labels = vcat([0 for _ in 1:length(soc1)], [1 for _ in 1:length(rug1)])
    (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((dataall, labels)), at = 0.7)
end;

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:1)
    return (X_batch, Y_batch)
end

begin
    # here we define the train and test sets.
    batchsize = 128
    mb_idxs = Iterators.partition(1:length(x_train), batchsize)
    train_set = [make_minibatch(x_train, y_train, i) for i in mb_idxs]
    test_set = make_minibatch(x_test, y_test, 1:length(x_test));
end;

begin
    model = Chain(
        Conv((3, 3), 1=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        Flux.flatten, 
        Dense(15488, 2),
        softmax)
end;

begin
    train_loss = Float64[]
    test_loss = Float64[]
    acc = Float64[]
    ps = Flux.params(model)
    opt = ADAM(0.00001)
    L(x, y) = Flux.crossentropy(model(x), y)
    L((x,y)) = Flux.crossentropy(model(x), y)
    accuracy(x, y, f) = mean(Flux.onecold(f(x)) .== Flux.onecold(y))
    
    function update_loss!()
        push!(train_loss, mean(L.(train_set)))
        push!(test_loss, mean(L(test_set)))
        push!(acc, accuracy(test_set..., model))
        @printf("train loss = %.2f, test loss = %.2f, accuracy = %.2f\n", train_loss[end], test_loss[end], acc[end])
    end
end

begin
@epochs 10 Flux.train!(L, ps, train_set, opt;
               cb = Flux.throttle(update_loss!, 8))
end

begin
    plot(train_loss, xlabel="Iterations", title="Model Training", label="Train loss", lw=2, alpha=0.9)
    plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
    plot!(acc, label="Accuracy", lw=2, alpha=0.9)
end
