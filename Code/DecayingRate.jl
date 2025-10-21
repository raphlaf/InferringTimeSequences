using Random
include("AdaptiveRate.jl")

mutable struct DecayingRate
    ar::AdaptiveRate
    τ_s::Float64  # synaptic rate
    time_intervals::Array{Float64, 1}  # sequence of time intervals
    x0::Float64
    saved_post_event_activity::Array{Float64, 1}

    function DecayingRate(ar::AdaptiveRate, τ_s::Float64,
        time_intervals::Array{Float64, 1}, x0::Float64=1.0)
        # Constructor
        n = length(time_intervals)
        λ_values = firing_parameters(ar, time_intervals, x0)
        initial_λ = max(ar.a*x0 + ar.c, 0.0)
        saved_post_event_activity = zeros(Float64, n+1)
        saved_post_event_activity[1] = rand(Poisson(initial_λ))
        for i=1:n
            start_I = saved_post_event_activity[i]
            end_I = start_I*exp(-time_intervals[i]/τ_s)
            saved_post_event_activity[i+1] = end_I + rand(Poisson(λ_values[i]))
        end
        return new(ar, τ_s, time_intervals, x0, saved_post_event_activity)
    end
end

function add_interval!(dr::DecayingRate, T::Float64)
    append!(dr.time_intervals, T)

    λ = firing_parameter(dr.ar, time_intervals, dr.x0)
    end_I = dr.saved_post_event_activity[end] * exp(-T/dr.τ_s)
    append!(dr.saved_post_event_activity, end_I + rand(Poisson(λ)))
end

function rate(dr::DecayingRate, t::Float64)
    n = length(dr.time_intervals)
    start_index = 1
    sum_t = 0.0
    while (t > sum_t)
        if (start_index == n+1) break end
        new_sum_t = sum_t + dr.time_intervals[start_index]
        if (new_sum_t > t) break end
        sum_t = new_sum_t
        start_index += 1
    end
    start_I = dr.saved_post_event_activity[start_index]
    return start_I * exp(-(t - sum_t)/dr.τ_s)
end

function trace(dr::DecayingRate, trange::Array{Float64, 1})
    n = length(dr.time_intervals)
    start_index = 1
    sum_t = 0.0
    while (trange[1] > sum_t)
        if (start_index == n+1) break end
        new_sum_t = sum_t + dr.time_intervals[start_index]
        if (new_sum_t > trange[1]) break end
        sum_t = new_sum_t
        start_index += 1
    end
    y = zeros(length(trange))
    y[1] = dr.saved_post_event_activity[start_index]*exp(-(trange[1]-sum_t)/dr.τ_s)
    for (i, t) in enumerate(trange[2:end])
        if (start_index != n+1) && (t > sum_t + dr.time_intervals[start_index])
            sum_t += dr.time_intervals[start_index]
            start_index += 1
        end
        start_I = dr.saved_post_event_activity[start_index]
        y[i] = start_I*exp(-(t - sum_t)/dr.τ_s)
    end
    return y
end

# Tmin, Tmax = 0.1, 30.0
# time_intervals = rand(50)*(Tmax - Tmin) .+ Tmin
# τ_s = 0.8
# ar = AdaptiveRate(10.0, 0.0, 0.5, 15.55)
# dr = DecayingRate(ar, τ_s, time_intervals, 1.0)

# t_end = sum(time_intervals)+Tmax
# dt = 0.01
# res = Int64(floor(t_end/dt))

# time_range = collect(LinRange(0.0, t_end, res))

# @time begin
# y1 = [rate(dr, t) for t=time_range]
# end
# @time begin
# y2 = trace(dr, time_range)
# end

# using Makie, GLMakie

# fig = Figure()

# lines(fig[1, 1], time_range, y1)
# lines!(fig[1, 1], time_range, y2)
# fig