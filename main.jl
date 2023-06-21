# Imports.
using NeuralOperators
using Distributions
using LinearAlgebra
using Plots
using MLUtils
using Flux
using FluxTraining

include("src/DataGenerator.jl")

## Preliminary.
# Creating objects.
domain = DifferentialOperator.Domain(nsteps = 50)
sampler = DataGenerator.Sampler()

# Generate maps.
forcing = DifferentialOperator.forcing_term(domain)
forward_realization = a -> DifferentialOperator.forward_operator(domain, forcing, a)

# Generate data.
A = DataGenerator.generate_permeability(sampler, domain.x, 1000)
U = mapslices(forward_realization, A; dims = 1)

# Reshape data [x or y, node, value]
a_data = reshape(A, 1, size(A)...)
u_data = reshape(U, 1, size(U)...)

## Learning forward operator.
# Split data.
ratio = 0.9
data_train, data_test = splitobs((a_data, u_data), at = ratio)
model = FourierNeuralOperator(ch = (1, 64, 64, 64, 64, 64, 128, 1), modes = (16,),
                              σ = :gelu)
λ = 1.0f-4
η = 1.0f-3
optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))
loss_func = l₂loss
learner = Learner(model, (data_train, data_test), optimiser, loss_func)

# Train model.
epochs = 2
fit!(learner, epochs)

