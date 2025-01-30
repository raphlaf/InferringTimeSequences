using JLD2
using QuadGK
using Optim
using FiniteDiff
using LinearAlgebra

file_path = "Data/RMSE_averages.jld"

function lamb(t, a, tau)
    λ = a*(1.0 - exp(-t/tau))
    if (λ < 0.0) return 0.0 end
    return λ
end

function fisher(t, a, tau)
    λ = lamb(t, a, tau)
    if (λ == 0.0) return 0.0 end
    if (tau == 0.0) return 0.0 end
    return (a - λ)^2/(tau^2*λ)
end

function normal(x, μ, σ)
    return exp(-(x-μ)^2/(2*σ^2))/(σ*sqrt(2π))
end

function lognormal(x, μ, σ)
    m = log(μ^2/sqrt(μ^2 + σ^2))
    s2 = log(1 + σ^2/μ^2)
    return exp(-(log(x)-m)^2/(2*s2))/(x*sqrt(s2*2π))
end


function dist(τ, tau_means, tau_sigms, p_values)
    N = length(tau_means)
    if (N != length(tau_sigms) || N != length(p_values)) exit(-1) end
    dists = [p_values[i]*lognormal(τ, tau_means[i], tau_sigms[i]) for i=1:N]
    dist = sum(dists)
    return dist
end

function powerlaw(t, Tmin, Tmax, k)
    a = 1/log(Tmax/Tmin)
    if (k != 1)
        a = (1-k)/(Tmax^(1-k) - Tmin^(1-k))
    end
    return a*t^(-k)
end

function RMSE_total(T, a, tau_dist, tau_min=0.0, tau_max=Inf, return_error=false)
    function integrand(tau)
        return tau_dist(tau)*fisher(T, a, tau)
    end
    integ, err = quadgk(integrand, tau_min, tau_max)
    ret = 1/sqrt(integ)
    if (return_error)
        return ret, err
    else
        return ret
    end
end

function RMSE_avg(a, T_dist, tau_dist, tau_min=0.0, tau_max=Inf, T_min=0.0, T_max=30.0, return_error=false)
    function integrand(T)
        return T_dist(T)*RMSE_total(T, a, tau_dist, tau_min, tau_max)
    end
    integ, err = quadgk(integrand, T_min, T_max)
    ret = integ
    if (return_error)
        return ret, err
    else
        return ret
    end
end

function RMSE_avg_dirac(a, T_dist, tau, T_min=0.0, T_max=30.0, return_error=false)
    function integrand(T)
        if (fisher(T, a, tau) <= 0.0)
            return 0.0
        end
        return T_dist(T)/sqrt(fisher(T, a, tau))
    end
    integ, err = quadgk(integrand, T_min, T_max)
    ret = integ
    if (return_error)
        return ret, err
    else
        return ret
    end
end

function RMSE_dirac(a, T, tau)
    return 1/sqrt(fisher(T, a, tau))
end

res = 200
T_space = LinRange(0.0, 30.0, res)
tau_space = T_space .* 2.0

a = 10.0

RMSE_dirac_values = zeros(res, res)

Threads.@threads for i=1:res
    for j=1:res
        RMSE_dirac_values[i, j] = RMSE_dirac(a, T_space[i], tau_space[j])
    end
end

T_min, T_max = 0.1, 30.0
k_values = [-1.0, 0.0, 1.0]
RMSE_avg_dirac_values = zeros(res, length(k_values))
Threads.@threads for i=1:res
    for (j, k) in enumerate(k_values)
        T_dist = x->powerlaw(x, T_min, T_max, k)
        RMSE_avg_dirac_values[i, j] = RMSE_avg_dirac(a, T_dist, tau_space[i], T_min, T_max)
    end
end


mu_values_mu = [1.0, 10.0, 30.0]  # averages of tau distributions
sigm_values_mu = [1.0, 1.0, 1.0]  # standard deviations of tau distributions
n = length(mu_values_mu)
@assert n==length(sigm_values_mu)
RMSE_total_values_mu = zeros(res, length(mu_values_mu))
Threads.@threads for i=1:res
    for j=1:n
        tau_dist = x->lognormal(x, mu_values_mu[j], sigm_values_mu[j])
        RMSE_total_values_mu[i, j] = RMSE_total(T_space[i], a, tau_dist)
    end
end


mu_values_sigm = [10.0, 10.0, 10.0]  # averages of tau distributions
sigm_values_sigm = [5.0, 15.0, 50.0]  # standard deviations of tau distributions
n = length(mu_values_sigm)
@assert n==length(sigm_values_sigm)
RMSE_total_values_sigm = zeros(res, length(sigm_values_sigm))
Threads.@threads for i=1:res
    for j=1:n
        tau_dist = x->lognormal(x, mu_values_sigm[j], sigm_values_sigm[j])
        RMSE_total_values_sigm[i, j] = RMSE_total(T_space[i], a, tau_dist)
    end
end

save(file_path, "res", res, "tau_space", [tau for tau in tau_space], "T_space", [T for T in T_space], "a", a, 
                "RMSE_dirac_values", RMSE_dirac_values, "T_min", T_min, "T_max", T_max,
                "k_values", k_values, "RMSE_avg_dirac_values", RMSE_avg_dirac_values,
                "mu_values_mu", mu_values_mu, "sigm_values_mu", sigm_values_mu,
                "RMSE_total_values_mu", RMSE_total_values_mu,
                "mu_values_sigm", mu_values_sigm, "sigm_values_sigm", sigm_values_sigm,
                "RMSE_total_values_sigm", RMSE_total_values_sigm)