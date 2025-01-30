using JLD2
using QuadGK
using Optim
using FiniteDiff
using LinearAlgebra
using GLMakie, Makie

file_path = "Data/FI_params.jld"

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

res = 100

adef = 10.0
cdef = 5.0
taudef = 10.0
betadef = 0.5
x0 = 1.0

amin, amax = 0.0, 10.0
arange = LinRange(amin, amax, res)
cmin, cmax = 0.0, 10.0
crange = LinRange(cmin, cmax, res)
taumin, taumax = 0.1, 30.0
taurange = LinRange(taumin, taumax, res)
betamin, betamax = 0.0, 1.0
betarange = LinRange(betamin, betamax, res)

T = 10.0

FI = zeros(4, res)

FI[1, :] = fisher.(T, arange, cdef, taudef, betadef, x0)
FI[2, :] = fisher.(T, adef, crange, taudef, betadef, x0)
FI[3, :] = fisher.(T, adef, cdef, taurange, betadef, x0)
FI[4, :] = fisher.(T, adef, cdef, taudef, betarange, x0)

save(file_path, 
    "arange", [a for a in arange],
    "crange", [c for c in crange],
    "taurange", [tau for tau in taurange],
    "betarange", [beta for beta in betarange],
    "FI", FI)