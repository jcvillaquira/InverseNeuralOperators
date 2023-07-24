# Data Imports.
using NeuralOperators
using Plots
using LinearAlgebra
using Interpolations

# Flux Imports
using MLUtils
using Flux
using FluxTraining

model = FourierNeuralOperator(
    ch = (2, 64, 64, 64, 64, 64, 128, 10),
    modes = (8,),
    σ = gelu)

n_samples = 100
n_functions = 2
nt = 10
nx = 21

xdata = rand(n_functions, nx, n_samples);
ydata = rand(nt, nx, n_samples);

λ = 1.0f-4
η = 1.0f-3
ratio = 0.8
optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))

data_train, data_test = splitobs((Float32.(xdata), Float32.(ydata)), at = ratio)
loader_train = DataLoader(data_train, batchsize = 50, shuffle = false)
loader_test = DataLoader(data_test, batchsize = 50, shuffle = false)
data = collect.((loader_train, loader_test))

learner = Learner(model, data, optimiser, l₂loss)
fit!(learner, 1)
