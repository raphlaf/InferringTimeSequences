using JLD2
using QuadGK
using Optim
using FiniteDiff
using LinearAlgebra

file_path = "Data/power_law_sweep.jld"

function lamb(t, a, tau)
    λ = a*(1.0 - exp(-t/tau))
    if (λ < 0.0) return 0.0 end
    return λ
end

function fisher(t, a, tau)
    λ = lamb(t, a, tau)
    if (λ <= 0.0)
        return 0.0
    end
    if (tau <= 0.0)
        return 0.0
    end
    F = (a - λ)^2/(tau^2*λ)
    return F
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
    return sum(dists)
end


function total_fisher(t, a, tau_means, tau_sigms, p_values, tau_min=0.0, tau_max=Inf)
    integrand = τ -> fisher(t, a, τ)*dist(τ, tau_means, tau_sigms, p_values)
    int, err = quadgk(integrand, tau_min, tau_max)
    return int
end

function total_fisher_uni(t, a, tau_min, tau_max)
    c = 1/(tau_max - tau_min)
    integrand = τ -> fisher(t, a, τ)*c
    int, err = quadgk(integrand, tau_min, tau_max)
    return int
end

function prior_uni(t, Tmin, Tmax)
    return 1/(Tmax - Tmin)  # uniform
end

function prior_power(t, Tmin, Tmax, k)
    a = 1/log(Tmax/Tmin)
    if (k != 1)
        a = (1-k)/(Tmax^(1-k) - Tmin^(1-k))
    end
    return a*t^(-k)
end

function xi(a, n, tau_means, tau_sigms, p_values, Tmin=0.1, Tmax=30.0, k=1, tau_min=0.0, tau_max=Inf, relative=false)
    function integrand(t)
        if (relative)
            return prior_power(t, Tmin, Tmax, k)* (t^2 * total_fisher(t, a, tau_means, tau_sigms, p_values, tau_min, tau_max))^n
        else
            return prior_power(t, Tmin, Tmax, k)*(total_fisher(t, a, tau_means, tau_sigms, p_values, tau_min, tau_max))^n
        end
    end
    int, err = quadgk(integrand, Tmin, Tmax)
    return int
end

function xi_dir(a, n, tau_means, p_values, Tmin=0.1, Tmax=30.0, k=1, relative=false)
    N = length(tau_means)
    if (N != length(p_values)) exit(-1) end
    function integrand(t)
        r = 0.0
        if (relative)
            r = prior_power(t, Tmin, Tmax, k)*( t^2 * sum([p_values[i]*fisher(t, a, tau_means[i]) for i=1:N]) )^n
        else
            r = prior_power(t, Tmin, Tmax, k)*( sum([p_values[i]*fisher(t, a, tau_means[i]) for i=1:N]) )^n
        end
        if (r === NaN || abs(r) == Inf)
            r = 0.0
        end
        return r
    end
    int, err = quadgk(integrand, Tmin, Tmax)
    return int
end

function xi_uni(a, n, tau_min, tau_max, Tmin=0.1, Tmax=30.0, k=1, relative=false)
    function integrand(t)
        if (relative)
            return prior_power(t, Tmin, Tmax, k)*( t^2 * total_fisher_uni(t, a, tau_min, tau_max) )^n
        else
            return prior_power(t, Tmin, Tmax, k)*total_fisher_uni(t, a, tau_min, tau_max)^n
        end
    end
    int, err = quadgk(integrand, Tmin, Tmax)
    return int
end

tau_mean_min, tau_mean_max = 0.1, 80.0
a = 10.0
n = -0.5  # -0.5 is RMSE ; -1.0 is MSE ; 1.0 is FI
Tmin, Tmax = 0.1, 30.0

resolution = 100
sigm_values = [4.0, 8.0, 16.0]
k_values = LinRange(-1.0, 1.0, resolution)

N = 1  # number of peaks

mean_tau = zeros(length(sigm_values)+2, resolution)
xi_values = zeros(length(sigm_values)+2, resolution)

lower = [tau_mean_min for _=1:N]
upper = [tau_mean_max for _=1:N]
tau_0 = [15.0 for _=1:N]
p_values = [1.0/N for _=1:N]

opt_relative = true  # optimize relative CRLB
save_relative = true  # save relative CRLB values

Threads.@threads for i=1:resolution
    k = k_values[i]
    for j=1:length(sigm_values)
        sigm = sigm_values[j]
        tau_sigms = [sigm for _=1:N]
        f = x -> (n>0 ? -1 : 1)*xi(a, n, x, tau_sigms, p_values, Tmin, Tmax, k, 0.0, Inf, opt_relative)
        res = optimize(f, lower, upper, tau_0)
        mean_tau[j, i] = Optim.minimizer(res)[end]
        xi_values[j, i] = (n>0 ? -1 : 1)*xi(a, n, Optim.minimizer(res), tau_sigms, p_values, Tmin, Tmax, k, 0.0, Inf, save_relative)
    end
    fd = x -> xi_dir(a, n, x, p_values, Tmin, Tmax, k, opt_relative)
    res = optimize(fd, lower, upper, tau_0)
    mean_tau[end, i] = Optim.minimizer(res)[end]
    xi_values[end, i] = xi_dir(a, n, Optim.minimizer(res), p_values, Tmin, Tmax, k, save_relative)

    tau_min = 1e-6
    tau_max = maximum(sigm_values)*sqrt(12) + tau_min
    mean_tau[end-1, i] = (tau_max - tau_min)/2 + tau_min
    xi_values[end-1, i] = xi_uni(a, n, tau_min, tau_max, Tmin, Tmax, k, save_relative)
end

save(file_path, "k_values", [k for k in k_values], "sigm_values", sigm_values, "mean_tau", mean_tau, "xi_values", xi_values)
