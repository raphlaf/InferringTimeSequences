using Distributions
using SpecialFunctions
using GLMakie, Makie

function resource_variables(t, β, τ, x0=1.0)
    n = length(t)
    ret = zeros(Float64, n)
    ret[1] = 1.0 - exp(-t[1]/τ)*(1.0 - β*x0)
    for i=2:n
        ret[i] = 1.0 - exp(-t[i]/τ)*(1.0 - β*ret[i-1])
    end
    return ret
end

function x(t, β, τ, x0=1.0)
    return 1.0 - exp(-t/τ)*(1.0 - β*x0)
end

function firing_parameter(x, a, c)
    ret = a*x + c
    if (ret < 0.0) return 0.0 end
    return ret
end

function rate_distribution(r, lambda)
    return lambda^(-r) * e^(-lambda) / gamma(r+1)
end

function loglikelihood(t, R_values, a_values, c_values, β_values, τ_values, x0=1.0)
    N = length(R_values)
    @assert N == length(a_values) == length(c_values) == length(β_values) == length(τ_values)
    xn_values = [resource_variables(t, β_values[i], τ_values[i], x0)[end] for i=1:N]
    lamb_values = [firing_parameter(xn_values[i], a_values[i], c_values[i]) for i=1:N]
    ll_values = R_values .* log.(lamb_values) .- lamb_values .- log.(gamma.(R_values .+ 1))
    return ll_values
end

function FI(t, a_values, c_values, β_values, τ_values, x0=1.0)
    N = length(a_values)
    @assert N == length(c_values) == length(β_values) == length(τ_values)
    I_values = [a_values[i]^2*(1 - β_values[i]*x0)^2*exp(-2*t/τ_values[i])/( τ_values[i]^2 * (a_values[i]*(1 - exp(-t/τ_values[i])*(1 - β_values[i]*x0)) + c_values[i]) ) for i=1:N]
    return I_values
end

N = 1000

a = 10.0
tau = 10.0
a_min, a_max = 1.0, 20.0
tau_min, tau_max = 0.1, 20.0


tau_sweep = [2.0, 10.0, 15.53, 20.0, 40.0]
res = 2000

Tmin, Tmax = 0.1, 30.0
Trange = LinRange(Tmin, Tmax, res)

total_nll_values = zeros(length(tau_sweep), res)
total_FI_values = zeros(length(tau_sweep), res)

for (j, tau) in enumerate(tau_sweep)
    τ_values = [tau for _=1:N]
    a_values = [a for _=1:N]
    c_values = [0 for _=1:N]
    β_values = [0 for _=1:N]
    het_a = rand(N).*(a_max - a_min) .+ a_min
    het_tau = rand(N).*(tau_max - tau_min) .+ tau_min

    heterogeneous = false

    if (heterogeneous)
        a_values = het_a
        τ_values = het_tau
    end

    T = 10.0  # time to be estimated
    x0 = 1.0

    xn_values = [resource_variables(T, β_values[i], τ_values[i], x0)[end] for i=1:N]
    lamb_values = [firing_parameter(xn_values[i], a_values[i], c_values[i]) for i=1:N]
    poi_dists = [Poisson(lamb_values[i]) for i=1:N]
    R_values = [rand(poi_dists[i]) for i=1:N]


    ll_values = zeros(N, res)
    FI_values = zeros(N, res)
    for i=1:res
        ll_values[:, i] = loglikelihood([Trange[i],], R_values, a_values, c_values, β_values, τ_values)
        FI_values[:, i] = FI(Trange[i], a_values, c_values, β_values, τ_values, 1.0)
    end

    total_nll_values[j, :] = vec(-sum(ll_values, dims=1))
    total_FI_values[j, :] = vec(sum(FI_values, dims=1))
end

time_intervals = [0.0, 5.0, 1.0, 4.0]

n = length(time_intervals)-1  # number of time intervals
x1 = zeros(res*n)
x2 = zeros(res*n)

beta1 = 0.0
beta2 = 0.5
tau1 = 3.0
tau2 = 3.0

full_trange = zeros(res*n)

for i=1:n
    x0 = 1.0
    x1[1] = x0
    x2[1] = x0
    x10 = x1[(i-1)*res+1]
    x20 = x2[(i-1)*res+1]
    if (i == 0) x10 = x20 = x0 end
    Tint = time_intervals[i]
    trange = LinRange(0.0, Tint, res)
    full_trange[(i-1)*res+1:i*res] = trange .+ sum(time_intervals[1:i+1])
    x1values = x.(trange, beta1, tau1, x10)
    x2values = x.(trange, beta2, tau2, x20)
    x1[(i-1)*res+1:i*res] = x1values
    x2[(i-1)*res+1:i*res] = x2values
end

save("Data/model_loglikelihood.jld", "Trange", [T for T in Trange], "nll", total_nll_values, "FI", total_FI_values, "tau_sweep", tau_sweep)
