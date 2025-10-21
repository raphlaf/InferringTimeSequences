using Distributions

mutable struct AdaptiveRate
    a::Float64
    c::Float64
    β::Float64
    τ::Float64
end

function resource_variables(ar::AdaptiveRate, time_intervals::Array{Float64, 1},
    x0::Float64)
    old_x = x0
    rv_values = zeros(length(time_intervals))
    for (i, T) in enumerate(time_intervals)
        old_x = 1.0 - exp(-T/ar.τ)*(1.0 - ar.β*old_x)
        rv_values[i] = old_x
    end
    return rv_values
end

function firing_parameters(ar::AdaptiveRate, time_intervals::Array{Float64, 1},
    x0::Float64)
    rv_values = resource_variables(ar, time_intervals, x0)
    λ_values = zeros(length(time_intervals))
    for (i, x) in enumerate(rv_values)
        λ_values[i] = max(ar.a*x + ar.c, 0)
    end
    return λ_values
end

function resource_variable(ar::AdaptiveRate, time_intervals::Array{Float64, 1},
    x0::Float64)
    old_x = x0
    for (i, T) in enumerate(time_intervals)
        old_x = 1.0 - exp(-T/ar.τ)*(1.0 - ar.β*old_x)
        rv_values[i] = old_x
    end
    return old_x
end

function firing_parameter(ar::AdaptiveRate, time_intervals::Array{Float64, 1}, 
    x0::Float64)
    rv = resource_variable(ar, time_intervals, x0)
    λ = max(ar.a * rv + ar.c, 0)
    return λ
end