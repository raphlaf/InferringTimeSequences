using LinearAlgebra, Random
using CUDA

mutable struct RNN
    N::Int64  # number of neurons
    x::CuArray{Float64, 1}  # current hidden state
    W::CuArray{Float64, 2}  # recurrent weight matrix
    bias::CuArray{Float64, 1}  # bias of neurons
    g::Float64  # chaos parameter
    τ::CuArray{Float64, 1}  # state time constant
    activation::Function  # activation function
end

function euler_step!(x::CuArray{Float64, 1}, df::CuArray{Float64, 1}, dt::Float64,
    σ::Float64)
    if (σ > 0.0)
        x .+= dt * df .+ σ * sqrt(dt) * CUDA.randn(Float64, size(x))
    else
        x .+= dt * df
    end
end

function dx_neuron(x::CuArray{Float64, 1}, W::CuArray{Float64, 2},
                   bias::CuArray{Float64, 1}, g::Float64, τ::CuArray{Float64, 1},
                   activation::Function, input::CuArray{Float64, 1})
    y = activation.(x + bias + input)
    return (g*W*y - x)./τ
end

function evolve_state!(rnn::RNN, input_states::CuArray{Float64, 2}, dt::Float64,
    σ::Float64, save_steps::Bool=true)
    nsteps = size(input_states)[2]
    saved_states = CUDA.zeros(Float64, rnn.N, nsteps)
    for i=1:nsteps
        df = dx_neuron(rnn.x, rnn.W, rnn.bias, rnn.g, rnn.τ, rnn.activation, input_states[:, i])
        euler_step!(rnn.x, df, dt, σ)
        if (save_steps)
            saved_states[:, i] .= rnn.activation.(rnn.x)
        end
    end
    if (save_steps)
        return saved_states
    end
end

function RR_weights(states::CuArray{Float64, 2},
    objective::CuArray{Float64, 2}, α::Float64=1e-6)
    # Ridge regression on samples
    # states = N×s
    # objective = n×s
    Wout = objective * states' * inv(states * states' + α*I)
    return Wout
end

function RR_weights(states::Array{Float64, 2},
    objective::Array{Float64, 2}, α::Float64=1e-6)
    # Ridge regression on samples
    # states = N×s
    # objective = n×s
    Wout = objective * states' * inv(states * states' + α*I)
    return Wout
end

function RLS_weights!(W::CuArray{Float64, 2}, P::CuArray{Float64, 1},
    state::CuArray{Float64, 1}, objective::CuArray{Float64, 1}, α::Float64)
    # Recursive Least Squares on one sample
    # W = n×N
    # P = N×N
    # state = N×1
    # objective = n×1
    k = P*state/(α + state' * P * state)
    new_W = W + (objective - W * state)*k'
    new_P = (P - k * r' * P)/α
    W[:] = new_W
    P[:] = new_P
end