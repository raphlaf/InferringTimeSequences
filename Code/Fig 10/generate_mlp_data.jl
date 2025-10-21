# Figure 10

using Random
using Lux, LuxCUDA, Optimisers, Zygote
using JLD2
using FileIO
include("../AdaptiveRate.jl")

dev = gpu_device()
cpu = cpu_device()
rng = Random.default_rng()
# Random.seed!(rng, 0)

function generate_input_output(nsamples, n_intervals, n_values, a_values,
    c_values, β_values, τ_values, x0_values, Tmin, Tmax)
    time_intervals_set = rand(n_intervals, nsamples).*(Tmax - Tmin) .+ Tmin
    N_input = length(a_values)
    response_input = zeros(Float32, N_input, nsamples)
    time_intervals_output = zeros(Float32, length(n_values), nsamples)
    ars = [AdaptiveRate(a_values[i], c_values[i], β_values[i], τ_values[i])
        for i=1:N_input]
    for i=1:nsamples
        dists = [Poisson(firing_parameters(ars[j], time_intervals_set[:, i],
            x0_values[j])[end]) for j=1:N_input]
        response_input[:, i] = rand.(dists)
        for (j, n) in enumerate(n_values)
            time_intervals_output[j, i] = time_intervals_set[end-n, i]
        end
    end
    return response_input, time_intervals_output, time_intervals_set
end

function train_model!(model, ps, st, x_data, y_data, noutput, abs_err, rel_err, max_epochs)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(learning_rate))
    last_training_loss = zeros(noutput)
    training_loss = zeros(max_epochs)

    for iter in 1:max_epochs
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoZygote(), MSELoss(),
            (x_data, y_data), train_state
        )
        Δ = abs.(loss .- last_training_loss)
        training_loss[iter] = loss
        last_training_loss = loss
        if prod(Δ .<= abs_err) && prod(Δ ./ loss .<= rel_err) break end
    end

    return model, ps, st, training_loss
end


N_input = 1000
training_samples = 5000
testing_samples = 1000
max_epochs = 10000

r = 0.1
a_values = ones(N_input)*10.0
c_values = ones(N_input)*0.0
# β_values = ones(N_input)*0.0
# β_values[1:Int64(floor(r*N_input))] .= 0.44
# τ_values = ones(N_input)*15.55
# τ_values[1:Int64(floor(r*N_input))] .= 30.0
x0_values = ones(N_input)*1.0
n_intervals = 10
Tmin, Tmax = 0.1, 30.0

res = 11
ratio_range = LinRange(0.0, 1.0, res)
β_range = LinRange(0.0, 1.0, res)

rmse_values = zeros(res, res, 2)
total_samples = zeros(res, res)
batch_samples = 1

n_values = [0, 1]
noutput = length(n_values)
N_output = length(n_values)
ntransient = 5

learning_rate = 0.001f0

model = Chain(
    Dense(N_input, N_input ÷ 8, relu),
    Dense(N_input ÷ 8, N_input, relu),
    Dense(N_input, N_output)
)

abs_err = 1e-5
rel_err = 1e-4

training_losses = zeros(res, res, max_epochs)

jld_filename = "Code/Fig 10/data_mlp.jld2"
if isfile(jld_filename)
    data = load(jld_filename)
    rmse_values = data["rmse_values"]
    if haskey(data, "total_samples")
        total_samples = data["total_samples"]
    end
else
    save(jld_filename,
        "beta_range", collect(β_range),
        "ratio_range", collect(ratio_range),
        "rmse_values", rmse_values, "total_samples", total_samples,
        "training_losses", training_losses)
end


for s=1:batch_samples
    for i=1:res
        for j=1:res
            β_values = ones(N_input)*0.0
            β_values[1:Int64(floor(ratio_range[j]*N_input))] .= β_range[i]
            τ_values = ones(N_input)*34.2
            τ_values[1:Int64(floor(ratio_range[j]*N_input))] .= 68.4

            # ars = [AdaptiveRate(a_values[k], c_values[k], β_values[k], τ_values[k]) for k=1:N_input]
            # T_transient = rand(ntransient).*(Tmax - Tmin) .+ Tmin
            # x0_values = [resource_variables(ar, T_transient, 1.0)[end] for ar in ars]

            training_input, training_output, training_intervals = generate_input_output(
                training_samples, n_intervals, n_values, a_values, c_values, β_values, τ_values,
                x0_values, Tmin, Tmax
            )
            training_input = training_input |> dev
            training_output = training_output |> dev

            testing_input, testing_output, testing_intervals = generate_input_output(
                testing_samples, n_intervals, n_values, a_values, c_values, β_values, τ_values,
                x0_values, Tmin, Tmax
            )
            testing_input = testing_input |> dev
            testing_output = testing_output |> dev

            ps, st = Lux.setup(rng, model) |> dev

            _, ps, st, training_loss = train_model!(model, ps, st, training_input, training_output, noutput, abs_err, rel_err, max_epochs)
            training_losses[i, j, :] = training_loss

            # Evaluate model on testing data
            y_pred, _ = model(testing_input, ps, st)
            rand_pred = randn(N_output, testing_samples).*5.5 .+ ((Tmax - Tmin)/2 + Tmin) |> dev
            cons_pred = ones(N_output, testing_samples)*(Tmax + Tmin)/2 |> dev
            test_loss1 = MSELoss()(y_pred[1, :], testing_output[1, :])
            rand_loss1 = MSELoss()(rand_pred[1, :], testing_output[1, :])
            cons_loss1 = MSELoss()(cons_pred[1, :], testing_output[1, :])
            println("Test RMSE T_{n} loss: ", sqrt.(test_loss1))
            println("Rand RMSE T_{n} loss: ", sqrt.(rand_loss1))
            println("Cons RMSE T_{n} loss: ", sqrt.(cons_loss1))
            test_loss2 = MSELoss()(y_pred[2, :], testing_output[2, :])
            rand_loss2 = MSELoss()(rand_pred[2, :], testing_output[2, :])
            cons_loss2 = MSELoss()(cons_pred[2, :], testing_output[2, :])
            println("Test RMSE T_{n-1} loss: ", sqrt.(test_loss2))
            println("Rand RMSE T_{n-1} loss: ", sqrt.(rand_loss2))
            println("Cons RMSE T_{n-1} loss: ", sqrt.(cons_loss2))


            rmse_values[i, j, 1] = (rmse_values[i, j, 1] * total_samples[i, j] + sqrt.(test_loss1))/(total_samples[i, j] + 1)
            rmse_values[i, j, 2] = (rmse_values[i, j, 2] * total_samples[i, j] + sqrt.(test_loss2))/(total_samples[i, j] + 1)
            total_samples[i, j] += 1
            save(jld_filename, "beta_range", collect(β_range),
                "ratio_range", collect(ratio_range),
                "rmse_values", rmse_values, "total_samples", total_samples,
                "training_losses", training_losses)
        end
    end
end
