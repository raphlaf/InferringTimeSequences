# Figure 9

using LinearAlgebra, JLD2


function resource_variable(T::Array{Float64, 1}, β::Float64, τ::Float64, 
    x0::Float64)
    old_x = x0
    for t in T
        old_x = 1.0 - exp(-t/τ)*(1.0 - β*old_x)
    end
    return old_x
end

function firing_parameter(T::Array{Float64, 1}, a::Float64, c::Float64,
    β::Float64, τ::Float64, x0::Float64)
    x = resource_variable(T, β, τ, x0)
    return max(0.0, a*x + c)
end

function dxdt(T::Array{Float64, 1}, i::Int64, β::Float64, τ::Float64,
    x0::Float64)
    n = length(T)
    @assert i <= n && i > 0
    prev_x = resource_variable(T[1:i-1], β, τ, x0)
    return β^(n-i)*(1.0 - β*prev_x)*exp(-sum(T[i:n])/τ)/τ
end

function fisher_information(T::Array{Float64, 1}, a::Float64, c::Float64,
    β::Float64, τ::Float64, x0::Float64)
    n = length(T)
    FI = zeros(n, n)
    for i=1:n
        for j=i:n
            λ = firing_parameter(T, a, c, β, τ, x0)
            dxdti = dxdt(T, i, β, τ, x0)
            dxdtj = dxdti
            if (i != j)
                dxdtj = dxdt(T, j, β, τ, x0)
            end
            FI[i, j] = a^2*dxdti*dxdtj/λ
            if (i != j)
                FI[j, i] = FI[i, j]
            end
        end
    end
    return FI
end

T1_degenerate = 10.0
T2_degenerate = 15.0

N = 1  # number of cells
res = 200
a = 10.0
c = 0.0
x0 = 1.0
β0 = 0.0
β1 = 0.4
β2 = 0.7
β3 = 0.2
τ0 = 15.0
τ1 = 15.0
τ2 = T1_degenerate/(T1_degenerate/τ1 - log(β1/β2))
τ3 = T2_degenerate/(T2_degenerate/τ2 - log(β2/β3))


# β0 = 0.0
# β1 = 0.424
# β2 = 0.699
# β3 = 0.169
# τ0 = 7.84
# τ1 = 13.2
# τ2 = 23.7
# τ3 = 9.68


Tmin, Tmax = 0.1, 30.0
Trange = LinRange(Tmin, Tmax, res)

detFIT2 = zeros(3, res, res)
CRLBT2 = zeros(3, res, res)

@time begin
for i=1:res
    Threads.@threads for j=1:res
        T = [Trange[i], Trange[j]]
        FI0 = fisher_information(T, a, c, β0, τ0, x0)
        FI1 = fisher_information(T, a, c, β1, τ1, x0)
        FI2 = fisher_information(T, a, c, β2, τ2, x0)
        FI3 = fisher_information(T, a, c, β1, τ2, x0)

        detFIT2[1, i, j] = det((FI0+FI1)/2)
        if (detFIT2[1, i, j] > 1e-30)
            CRLBT2[1, i, j] = sum(inv((FI0+FI1)/2*N))
        else
            CRLBT2[1, i, j] = 1e10
        end

        detFIT2[2, i, j] = det((FI1+FI3)/2)
        if (detFIT2[2, i, j] > 1e-30)
            CRLBT2[2, i, j] = sum(inv((FI1+FI3)/2*N))
        else
            CRLBT2[2, i, j] = 1e10
        end

        detFIT2[3, i, j] = det((FI1+FI2)/2)
        if (detFIT2[3, i, j] > 1e-30)
            CRLBT2[3, i, j] = sum(inv((FI1+FI2)/2*N))
        else
            CRLBT2[3, i, j] = 1e10
        end

    end
end
end

zind = detFIT2 .<= 1e-20
detFIT2[zind] .= 1e-20


detFIT3 = zeros(res, res, res)
CRLBT3 = zeros(res, res, res)
@time begin
for i=1:res
    for j=1:res
        Threads.@threads for k=1:res
            T = [Trange[i], Trange[j], Trange[k]]
            FI0 = fisher_information(T, a, c, β1, τ1, x0)
            FI1 = fisher_information(T, a, c, β2, τ2, x0)
            FI2 = fisher_information(T, a, c, β3, τ3, x0)

            # FI0 = fisher_information(T, a, c, β0, τ0, x0)
            # FI1 = fisher_information(T, a, c, β1, τ1, x0)
            # FI2 = fisher_information(T, a, c, β2, τ2, x0)

            detFIT3[i, j, k] = det((FI0+FI1+FI2)/3)
            if (detFIT3[i, j, k] > 1e-30)
                CRLBT3[i, j, k] = sum(inv((FI0+FI1+FI2)/3*N))
            else
                CRLBT3[i, j, k] = 1e10
            end
        end
    end
end
end

zind = detFIT3 .<= 1e-20
detFIT3[zind] .= 1e-20


save("Code/Fig 9/data.jld2", "detFIT2", detFIT2,
    "detFIT3", detFIT3, "CRLBT2", CRLBT2, "CRLBT3", CRLBT3,
    "Trange", collect(Trange),
    "beta0", β0, "beta1", β1, "beta2", β2, "beta3", β3,
    "tau0", τ0, "tau1", τ1, "tau2", τ2, "tau3", τ3)
