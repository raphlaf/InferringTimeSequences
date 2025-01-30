# Long computation time

using JLD2
using Optim
using Distributions

file_path = "Data/RMSE_FI_homogeneous.jld"
# file_path = "Data/RMSE_FI_heterogeneous.jld"

function lamb(t, a, tau)
    return a*(1.0 - exp(-t/tau))
end

function lamb(t, a, c, tau, beta, x0=1.0)
    λ = a*(1.0 - exp(-t/tau)*(1.0 - beta*x0)) + c
    if (λ < 0.0) return 0.0 end
    return λ
end

function fisher(t, a, c, tau, beta, x0=1.0)
    λ = lamb(t, a, c, tau, beta, x0)
    if (λ == 0.0) return 0.0 end
    if (tau == 0.0) return 0.0 end
    return (a + c - λ)^2/(tau^2*λ)
end

function estimate(responses, a_values, tau_values)
    N = length(responses)
    f = x -> -sum([responses[i]*log(lamb(x[1], a_values[i], tau_values[i])) - lamb(x[1], a_values[i], tau_values[i]) for i=1:N])
    res = optimize(f, [0.1,], [150.0,], [15.0,])
    return Optim.minimizer(res)[end]
end

nT = 100
T_min, T_max = 0.1, 30.0
T_values = LinRange(T_min, T_max, nT)


N_values = [1, 10, 100, 1000, 10000]
nN = length(N_values)

mean_results = zeros(nN, nT)
var_results = zero(mean_results)

samples = 100  # sample steps
abs_tol = 1e-6
max_samples = Int(1e8)

tau = 10.0
a = 10.0

# a_min, a_max = 1.0, 20.0
tau_min, tau_max = 0.1, 20.0

het_a = [a for _=1:10000]
het_tau = rand(10000).*(tau_max - tau_min) .+ tau_min

het_a[1] = a
het_tau[1] = tau

indices = CartesianIndices(mean_results)


heterogeneous = false  # change this to have heterogeneous population in tau

Threads.@threads for I in indices
    N = N_values[I[1]]
    T = T_values[I[2]]
    total_samples = 0
    sum = 0.0
    sum2 = 0.0
    mean = 0.0
    var = 0.0
    abs_mean_err = 1.0
    abs_var_err = 1.0
    a_values = [a for _=1:N]
    τ_values  = [tau for _=1:N]
    if (heterogeneous)
        a_values = het_a[1:N]
        τ_values = het_tau[1:N]
    end
    lambda_values = [lamb(T, a_values[i], τ_values[i]) for i=1:N]
    poi_dists = [Poisson(lambda_values[i]) for i=1:N]
    while ((abs_mean_err > abs_tol || abs_var_err > abs_tol) && total_samples < max_samples)
        total_samples += samples
        for i=1:samples
            R = [rand(poi_dists[i]) for i=1:N]
            T_est = estimate(R, a_values, τ_values)
            sum += T_est
            sum2 += T_est^2
        end
        new_mean = sum/total_samples
        new_var = sum2/total_samples - new_mean^2
        abs_mean_err = abs(new_mean - mean)
        abs_var_err = abs(new_var - var)
        mean = new_mean
        var = new_var
    end
    mean_results[I] = mean
    var_results[I] = var
end

a_values = [a for i=1:N_values[end]]
tau_values = [tau for i=1:N_values[end]]

if (heterogeneous)
    a_values = het_a
    tau_values = het_tau
end

save(file_path,
    "T_values", [T for T in T_values],
    "N_values", N_values,
    "mean_results", mean_results,
    "var_results", var_results,
    "a_values", a_values,
    "tau_values", tau_values)