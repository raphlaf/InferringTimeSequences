include("AdaptiveRatePopulation.jl")

file_path = "Data/data_example_llikelihood.jld"

T = [10.0, 15.0]
Tmin, Tmax = 0.1, 30.0
res = 1000
Trange = LinRange(Tmin, Tmax, res)

pop0 = AdaptiveRatePopulation(10.0, 0.0, 0.0, 15.0, 1000)
pop1 = AdaptiveRatePopulation(10.0, 0.0, 0.3, 15.0, 1000)
pop2 = AdaptiveRatePopulation(10.0, 0.0, 0.5, 15.0, 1000)

R01 = response(pop0, T)
R02 = response(pop0, T)
R11 = response(pop1, T)
R12 = response(pop1, T)
R21 = response(pop2, T)
R22 = response(pop2, T)

llikelihood = zeros(4, res, res)

for i=1:res
    for j=1:res
        llikelihood[1, i, j] = loglikelihood(pop0, [Trange[i], Trange[j]], R01) + loglikelihood(pop0, [Trange[i], Trange[j]], R02)
        llikelihood[2, i, j] = loglikelihood(pop1, [Trange[i], Trange[j]], R11) + loglikelihood(pop1, [Trange[i], Trange[j]], R12)
        llikelihood[3, i, j] = loglikelihood(pop0, [Trange[i], Trange[j]], R01) + loglikelihood(pop1, [Trange[i], Trange[j]], R11)
        llikelihood[4, i, j] = loglikelihood(pop1, [Trange[i], Trange[j]], R11) + loglikelihood(pop2, [Trange[i], Trange[j]], R21)
    end
end

save(file_path,
            "T", T,
            "Tmin", Tmin,
            "Tmax", Tmax,
            "res", res,
            "llikelihood", llikelihood,
            "pop0", pop0,
            "pop1", pop1,
            "pop2", pop2)