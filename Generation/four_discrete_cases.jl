using JLD2
using QuadGK
using Optim
using FiniteDiff
using LinearAlgebra
using GLMakie, Makie

file_path = "Data/four_discrete_cases_2.jld"

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

a1 = 5.0
a2 = 15.0
c = 0.0
beta = 0.0
# T1, T2 = 10.0, 15.0  # first case
T1, T2 = 2.0, 20.0  # 2nd case

res = 1000
tau_range = LinRange(0.1, 50.0, res)

FI = zeros(2, res)
for i=1:res
    tau = tau_range[i]
    F1 = fisher(T1, a1, c, tau, beta)
    F2 = fisher(T2, a2, c, tau, beta)
    FI[1, i] = F1
    FI[2, i] = F2
end

save(file_path, 
    "tau_range", [tau for tau in tau_range],
    "FI", FI)