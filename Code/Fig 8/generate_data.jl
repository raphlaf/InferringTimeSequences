# Figure 8

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

res = 600
βrange = LinRange(0.0, 1.0, res)
τrange = LinRange(0.0, 100.0, res)
a = 10.0
c = 0.0
x0 = 1.0

β_values_t1 = Float64[]
τ_values_t1 = Float64[]
β_values_t2 = Float64[]
τ_values_t2 = Float64[]
nmax = 6

det_values_t1 = zeros(nmax, res, res)
det_values_t2 = zeros(nmax, res, res)
crlb_values_t1 = zeros(nmax, res, res)
crlb_values_t2 = zeros(nmax, res, res)
t1 = 5.0
t2 = 10.0


for n=1:nmax
    T = [t1 for _=1:n]
    for i=1:res
        Threads.@threads for j=1:res
            FI_sum = zeros(n, n)
            for k=1:n-1
                FI_sum += fisher_information(T, a, c, β_values_t1[k], τ_values_t1[k], x0)
            end
            FI_sum += fisher_information(T, a, c, βrange[i], τrange[j], x0)
            det_values_t1[n, i, j] = det(FI_sum)
            if (det_values_t1[n, i, j] > 1e-30)
                crlb_values_t1[n, i, j] = sum(inv(FI_sum/n))
            else
                crlb_values_t1[n, i, j] = 1e10
            end
        end
    end
    det_values_t1 .= replace!(det_values_t1, NaN => 0.0)
    indices = argmax(det_values_t1[n, :, :])
    push!(β_values_t1, βrange[indices[1]])
    push!(τ_values_t1, τrange[indices[2]])
end


for n=1:nmax
    T = [t2 for _=1:n]
    for i=1:res
        for j=1:res
            FI_sum = zeros(n, n)
            for k=1:n-1
                FI_sum += fisher_information(T, a, c, β_values_t2[k], τ_values_t2[k], x0)
            end
            FI_sum += fisher_information(T, a, c, βrange[i], τrange[j], x0)
            det_values_t2[n, i, j] = det(FI_sum)
            if (det_values_t2[n, i, j] > 1e-30)
                crlb_values_t2[n, i, j] = sum(inv(FI_sum/n))
            else
                crlb_values_t2[n, i, j] = 1e10
            end
        end
    end
    det_values_t2 .= replace!(det_values_t2, NaN => 0.0)
    indices = argmax(det_values_t2[n, :, :])
    push!(β_values_t2, βrange[indices[1]])
    push!(τ_values_t2, τrange[indices[2]])
end



save("Code/Fig 8/data.jld2", "det_values_t1", det_values_t1,
    "beta_values_t1", β_values_t1, "tau_values_t1", τ_values_t1,
    "det_values_t2", det_values_t2,
    "crlb_values_t1", crlb_values_t1,
    "crlb_values_t2", crlb_values_t2,
    "beta_values_t2", β_values_t2, "tau_values_t2", τ_values_t2,
    "beta_range", collect(βrange), "tau_range", collect(τrange),
    "t1", t1, "t2", t2)
