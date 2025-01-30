using Distributions
using Optim

struct AdaptiveRatePopulation
    a::Float64
    c::Float64
    β::Float64
    τ::Float64
    N::Float64
end

function latent_state(population::AdaptiveRatePopulation, T::Vector{Float64},
                      x0::Float64=1.0)
#= Computes the latent state variable x of an adaptive rate population.
    input:
        population (AdaptiveRatePopulation): population whose latent state is
            needed
        T (Vector{Float64}): vector of any size >= 1 representing the time
            intervals between different encounters
        x0 (Float64): initial value of latent state. Defaults to 1.
    returns:
        x (Float64): latent state value at the end of the last encounter.
=#
    n = length(T)
    if (n == 0) return x0 end
    return 1.0 - exp(-T[n]/population.τ)*(1.0 - population.β*latent_state(population, T[1:n-1], x0))
end

function firing_parameter(population::AdaptiveRatePopulation, T::Vector{Float64},
                     x0::Float64=1.0)
#= Computes the firing parameter parameter λ of an adaptive rate population
    input:
        population (AdaptiveRatePopulation): population whose latent state is
            needed
        T (Vector{Float64}): vector of any size >= 1 representing the time
            intervals between different encounters
        x0 (Float64): initial value of latent state. Defaults to 1.
    returns:
        λ (Float64): firing parameter value at the end of the last encounter.
=#
    x = latent_state(population, T, x0)
    λ = (population.a*x + population.c)
    if (λ < 0.0) return 0.0 end
    return λ
end

function response(population::AdaptiveRatePopulation, T::Vector{Float64},
                 x0::Float64=1.0)
#= Generates an average response for a population
    input:
        population (AdaptiveRatePopulation): population whose response is needed
        T (Vector{Float64}): vector of any size >= 1 representing the time
            intervals between different encounters
        x0 (Float64): initial value of latent state. Defaults to 1.
    returns:
        response (Float64): average response of the population
=#
    λ_ = firing_parameter(population, T, x0)
    R = rand(Normal(λ_, sqrt(λ_/population.N)))
    if (R < 0.0) return 0.0 end
    return R
end

function loglikelihood(population::AdaptiveRatePopulation, T::Vector{Float64},
                       response::Float64, x0::Float64=1.0)
#= Computes the loglikelihood of a population given an average spike response at
    time vector T.
    input:
        population (AdaptiveRatePopulation): population whose loglikelihood is
            to be computed
        T (Vector{Float64}): vector of any size >= 1 representing the time
            intervals between different encounters
        response (Float64): average spike response at time vector T
        x0 (Float64): initial value of latent state. Defaults to 1.
    returns:
        loglikelihood (Float64): loglikelihood of the population given the
            average spike response at time vector T. This is not normalized as
            w.r.t. the response.
=#
    λ_ = firing_parameter(population, T, x0)
    if (response <= 0.0) return -population.N*λ_ end
    ll = population.N*(response*log(λ_) - λ_)
    if isnan(ll) return -Inf end
    if isinf(ll) return -Inf end
    return ll
end

function loglikelihood_gradient(population::AdaptiveRatePopulation, T::Vector{Float64},
                                response::Float64, i::Int64=0, x0::Float64=1.0)
#= Computes the gradient of the loglikelihood w.r.t. T_{n-i}
    input:
        population (AdaptiveRatePopulation): population whose loglikelihood is
            to be computed
        T (Vector{Float64}): vector of any size >= 1 representing the time
            intervals between different encounters
        response (Float64): average spike response at time vector T
        i (Int64): index of the time interval before the last time interval 
            w.r.t. which the gradient is computed (T_{n-i}). Defaults to 0,
            which means the gradient is computed w.r.t. the last time interval
            (T_{n-0} = T_{n}).
        x0 (Float64): initial value of latent state. Defaults to 1.
    returns:
        gradient (Float64): gradient of the loglikelihood w.r.t. T_{n-i}
=#
    λ_ = firing_parameter(population, T, x0)
    xni = latent_state(population, T[1:end-i], x0)  # x_{n-i}
    dλ = population.N*population.a*population.β^i*exp(-sum(T[end-i+1:end])/population.τ)*(1.0 - xni)/population.τ
    if (response <= 0.0) return -dλ end
    if (λ_ <= 0.0) return -Inf end
    return (response/λ_ - 1.0)*dλ
end

function fisher_information(population::AdaptiveRatePopulation, T::Vector{Float64},
                            i::Int64=0, x0::Float64=1.0)
#= Computes the Fisher information of a population given a time vector T.
    input:
        population (AdaptiveRatePopulation): population whose Fisher information
            is to be computed
        T (Vector{Float64}): vector of any size >= 1 representing the time
            intervals between different encounters
        i (Int64): index of the time interval before the last time interval 
            w.r.t. which the Fisher information is computed (T_{n-i}). Defaults
            to 0, which means the Fisher information is computed w.r.t. the
            last time interval (T_{n-0} = T_{n}).
        x0 (Float64): initial value of latent state. Defaults to 1.
    returns:
        fisher_information (Float64): Fisher information the population has on
        T_{n-i} at time vector T.
=#
    xni = latent_state(population, T[1:end-i], x0)  # x_{n-i}
    λn = firing_parameter(population, T, x0)  # λ_n
    numerator = population.N*population.a^2*population.β^(2*i)*exp(-2*sum(T[end-i+1:end])/population.τ)*(1.0 - xni)^2
    denominator = (population.τ^2*λn)
    if (denominator <= 0.0) return -Inf end
    return numerator/denominator
end

function optimal_population(T::Vector{Float64}, i::Int64=0, N::Float64=100.0,
                            Λ::Float64=1.0, alpha=1.0, x0::Float64=1.0,
                            beta_min=0.0, beta_max=0.99,
                            tau_min=0.01, tau_max = 100.0)
#= Returns a population whose parameters optimizes the Fisher information on
    interval T_{n-i} at time vector T.
    input:
        T (Vector{Float64}): vector of any size >= 1 representing the time
            intervals between different encounters
        i (Int64): index of the time interval before the last time interval 
            w.r.t. which the Fisher information is computed (T_{n-i}). Defaults
            to 0, which means the Fisher information is computed w.r.t. the
            last time interval (T_{n-0} = T_{n}).
        N (Int64): number of cells in the population. Defaults to 100.
        Λ (Float64): dynamic range remaining after optimized time vector.
            Defaults to 1.0.
        x0 (Float64): initial value of latent state. Defaults to 1.
    returns:
        population (AdaptiveRatePopulation): population whose parameters
            optimizes the Fisher information on interval T_{n-i} at time vector
            T.
=#
    function fi(params::Vector{Float64})
        beta = params[1]
        tau = params[2]
        if (i==0) beta = 0.0 end
        xn = latent_state(AdaptiveRatePopulation(0.0, 0.0, beta, tau, N), T, x0)
        population = AdaptiveRatePopulation(1.0, 0.0, beta, tau, N)
        return -fisher_information(population, T, i, x0)
    end
    res = optimize(fi, [beta_min, tau_min], [beta_max, tau_max], [0.5, 15.0])
    params = Optim.minimizer(res)
    optimal_β = params[1]
    if (i==0) optimal_β = 0.0 end
    optimal_τ = params[2]
    xn = latent_state(AdaptiveRatePopulation(10.0, 0.0, optimal_β, optimal_τ, N), T, x0)
    if (xn >= 1.0) println(optimal_β, "\t", optimal_τ) end
    optimal_a = Λ/(1.0 - xn)
    optimal_c = -alpha*optimal_a*xn
    return AdaptiveRatePopulation(optimal_a, optimal_c, optimal_β, optimal_τ, N)
end