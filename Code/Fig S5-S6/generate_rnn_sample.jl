# Figure S5-S6

#   CREATES LARGE SAVE FILE
#   LONG COMPUTATION TIME

using Random, Distributions
using LinearAlgebra
using Statistics
using JLD2
using Dates

Random.seed!(1234)
include("../AdaptiveRate.jl")
include("../DecayingRate.jl")
include("../RNN.jl")


function generate_input_output(n::Int64, Tmin::Float64, Tmax::Float64,
    dt::Float64, τ_s::Float64, N_input::Int64, N_output::Int64,
    a_values::Array{Float64, 1}, c_values::Array{Float64, 1},
    β_values::Array{Float64, 1}, τ_values::Array{Float64, 1},
    ntransient::Int64)

    @assert ntransient >= N_output
    time_intervals = rand(n+ntransient).*(Tmax - Tmin) .+ Tmin
    events = [sum(time_intervals[1:i]) for i=1:n+ntransient]
    ars = [AdaptiveRate(a_values[i], c_values[i], β_values[i], τ_values[i]) 
            for i=1:N_input]
    
    drs = [DecayingRate(ar, τ_s, time_intervals, 1.0) for ar in ars]

    total_time = events[end]
    simulation_steps = Int64(floor(total_time/dt))
    time_range = collect(LinRange(0.0, total_time, simulation_steps))


    event_indices = [Int(floor(ev/dt))+1 for ev in events]
    event_ranges = Array{UnitRange{Int64}, 1}(undef, length(event_indices))
    event_ranges[1] = 1:event_indices[1]
    for i=1:length(event_indices)-1
        event_ranges[i+1] = (event_indices[i]+1):min((event_indices[i+1]), simulation_steps)
    end

    input_data = zeros(Float64, N_input, simulation_steps)
    for i=1:N_input
        input_data[i, :] = trace(drs[i], time_range)
    end

    output_data = zeros(Float64, N_output, n)
    for i=1:N_output
        output_data[i, :] = time_intervals[ntransient+1-i:end-i]
    end

    # sim_start_index = searchsortedlast(input_trange, events[ntransient])
    return input_data, output_data, time_intervals, events, event_ranges
end


function exp_pool(data::CuArray{Float64, 1}, λ::Float64)
    e = CuArray([exp(-λ*(i-1)) for i=1:length(data)])
    return sum(data .* e)
end

function mean_pool(data::CuArray{Float64, 1})
    return mean(data)
end


@time begin
dt = 0.01  # integration time for reservoir
N_input = 1000  # number of adaptive cells as input to reservoir
N_output = 2  # number of most recent intervals to predict

# Generate adaptive input and corresponding output data

memory_ratio = 0.5
memory_value = 0.5
memory_τ_value = 30.0
a_values = ones(N_input)*10.0
c_values = ones(N_input)*0.0
β_values = ones(N_input)*0.0
β_values[1:Int64(floor(N_input*memory_ratio))] .= memory_value
τ_values = ones(N_input)*15.55
τ_values[1:Int64(floor(N_input*memory_ratio))] .= memory_τ_value
τ_s = 0.1  # (s) synaptic decay rate

Tmin, Tmax = 0.1, 30.0
nfull = 1000  # number of total time intervals
ntransient = 5  # number of transient time intervals
ntrain = Int64(floor(nfull*3/4))
ntest  = nfull - ntrain


input_data, output_data, time_intervals, events, event_ranges = generate_input_output(
    nfull, Tmin, Tmax, dt, τ_s, N_input, N_output, a_values, c_values, β_values,
    τ_values, ntransient
)

# create reservoir

N = 250
spectral_radius = 1.05

p = 0.1
W = randn(Float64, N, N)
mask = rand(Float64, N, N) .< p
W[mask] .= 0.0
λ_max = maximum(real(eigvals(W)))
W = W / abs(λ_max) * spectral_radius

input_scale = 0.1  # 3.0 fudge factor to manually scale the intensity of RNN drive
# Make Win block diagonal with two blocks
# (different parts of RNN are driven by memory cells vs non-memory cells)
block1_size = (Int64(floor(N * memory_ratio)), Int64(floor(N_input * memory_ratio)))
block2_size = (N, N_input) .- block1_size


Win = rand(Float64, N, N_input)
Win[1:block1_size[1], 1:block1_size[2]] ./= sqrt(prod(block1_size)) / input_scale
Win[block1_size[1]+1:end, block1_size[2]+1:end] ./= sqrt(prod(block2_size)) / input_scale
Win[block1_size[1]+1:end, 1:block1_size[2]] .= 0.0
Win[1:block1_size[1], block1_size[2]+1:end] .= 0.0
Win = CuArray{Float64}(Win)

bias = randn(Float64, N)*0.0
g = 1.0
τ_res = CUDA.ones(Float64, N)*1.0
activation = tanh
σ = 0.0  # noise input
α = 5e-3  # ridge regression regularizer
x0 = randn(Float64, N)

rnn = RNN(N, CuArray{Float64}(x0), CuArray{Float64}(W), CuArray{Float64}(bias), g, τ_res, activation)

for i=1:ntransient
    evolve_state!(rnn, Win*CuArray{Float64}(input_data[:, event_ranges[i]]), dt, σ, false)
end

ntrain_steps = event_ranges[ntransient+ntrain][end] - event_ranges[ntransient][end]

λ = dt/τ_s  # decay of exponential pool
train_rnn_states = CUDA.zeros(Float64, N, ntrain_steps)
train_pooled_states = zeros(N, ntrain)

for i=1:ntrain
    saved_states = evolve_state!(rnn, Win*CuArray{Float64}(input_data[:, event_ranges[ntransient+i]]), dt, σ, true)
    train_rnn_states[:, event_ranges[ntransient+i] .- event_ranges[ntransient][end]] = saved_states
    for j=1:N
        train_pooled_states[j, i] = exp_pool(saved_states[j, :], λ)
        # train_pooled_states[j, i-ntransient] = mean_pool(Array(saved_states[j, :]))
    end
end

Wout = RR_weights(train_pooled_states, output_data[:, 1:ntrain], α)

ntest_steps = event_ranges[ntransient+ntrain+ntest][end] - event_ranges[ntransient+ntrain][end]
offset = event_ranges[ntransient+ntrain][end]

test_rnn_states = CUDA.zeros(Float64, N, ntest_steps)
test_pooled_states = zeros(N, ntest)

for i=1:ntest
    saved_states = evolve_state!(rnn, Win*CuArray{Float64}(input_data[:, event_ranges[i+ntrain+ntransient]]), dt, σ, true)
    test_rnn_states[:, event_ranges[ntransient+ntrain+i] .- event_ranges[ntransient+ntrain][end]] = saved_states
    for j=1:N
        test_pooled_states[j, i] = exp_pool(saved_states[j, :], λ)
        # test_pooled_states[j, i - (ntransient+ntrain)] = mean_pool(Array(saved_states[j, :]))
    end
end

train_output = Wout*train_pooled_states
test_output = Wout*test_pooled_states

train_rmse = sqrt.(dropdims(mean((train_output .- Array(output_data[:, 1:ntrain])).^2, dims=2), dims=2))
test_rmse = sqrt.(dropdims(mean((test_output .- Array(output_data[:, ntrain+1:nfull])).^2, dims=2), dims=2))

println("Training RMSE (s): ", train_rmse)
println("Testing  RMSE (s): ", test_rmse)
end

save("Code/Fig S5-S6/data.jld2",
    "train_output", Array(train_output),
    "test_output", Array(test_output),
    "train_rmse", train_rmse,
    "test_rmse", test_rmse,
    "train_pooled_states", Array(train_pooled_states),
    "test_pooled_states", Array(test_pooled_states),
    "train_rnn_states", Array(train_rnn_states),
    "test_rnn_states", Array(test_rnn_states),
    "output_data", Array(output_data),
    "input_data", Array(input_data),
    "events", Array(events),
    "event_ranges", event_ranges,
    "time_intervals", Array(time_intervals),
    "memory_ratio", memory_ratio,
    "memory_value", memory_value,
    "Win", Array(Win),
    "Wout", Array(Wout),
)

