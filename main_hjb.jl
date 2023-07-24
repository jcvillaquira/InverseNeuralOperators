## Imports
# Data Imports.
using NeuralOperators
using Plots
using LinearAlgebra
using Interpolations

# Flux Imports
using MLUtils
using Flux
using FluxTraining

# Import own modules and structs.
include("src/HJBDataGenerator.jl")

## Prepare inputs for model
# Create domain and input functions.
domain = Data.Domain(tsteps = 15)
D, B = Data.generate_basis(10, 1.0)
N = 1_000
xdata = Sampler.sample_function(D, B, domain, 3, N)
xdata[3, domain.xsteps, :] .= 0.0
ydata = mapslices(x -> ValueFunction.apply_value_function(x, domain), xdata; dims = [1, 2])
# TODO: Continuity at the corners.

# Create DataLoaders.
data_train, data_test = splitobs((Float32.(xdata), Float32.(ydata)), at = 0.9)
loader_train, loader_test = DataLoader(data_train), DataLoader(data_test)
data = collect.((loader_train, loader_test))

# Hyper params
channels = (size(xdata, 1), 16, 16, 16, 16, 16, 32, size(ydata, 1))
models = (8,)
λ = 1.0f-4
η = 1.0f-3

# Create and train model.
model = FourierNeuralOperator(ch = channels, modes = models, σ = gelu)
optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))
learner = Learner(model, data, optimiser, l₂loss)
fit!(learner, 200)

