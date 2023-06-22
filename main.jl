# Imports.
using NeuralOperators
using Distributions
using LinearAlgebra
using Plots
using MLUtils
using Flux
using FluxTraining
using BSON
using BSON: @save, @load

include("src/DataGenerator.jl")

## Preliminary.
# Discretization
N = 1000
nsteps = 30

# Creating objects.
domain = DifferentialOperator.Domain(nsteps = nsteps)
sampler = DataGenerator.Sampler()

# Generate maps.
forcing = DifferentialOperator.forcing_term(domain)
forward_realization = a -> DifferentialOperator.forward_operator(domain, forcing, a)

# Generate data.
A = DataGenerator.generate_permeability(sampler, domain.x, N)
U = mapslices(forward_realization, A; dims = 1)

# Create model.
x_data = reshape(A, 1, size(A)...)
y_data = reshape(U, 1, size(U)...)

## Learning forward operator.
# Split data.
ratio = 0.9
λ = 1.0f-4
η = 1.0f-3
optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))
loss_func = l₂loss

# Dataloaders.
data_train, data_test = splitobs((Float32.(x_data), Float32.(y_data)), at = ratio)
loader_train = DataLoader(data_train, batchsize = 10, shuffle = false)
loader_test = DataLoader(data_test, batchsize = 10, shuffle = false)
data = collect.((loader_train, loader_test))

# Create and train model.
model = FourierNeuralOperator(ch = (1, 32, 32, 32, 32, 32, 64, 1), σ = gelu)
learner = Learner(model, data, optimiser, loss_func)

# Train model.
epochs = 100
fit!(learner, epochs)

# data[train or test][batch][x or y data][variable, mesh, observation]
model( data[1][1][1] )

# Plotting.
j = 1
a_to_plot = a_input[2, :, j]
u_to_plot = u_output[1, :, j]

# Save and load.
model_to_save = learner.model |> cpu
@save "models/forward.bson" model_to_save
BSON.load("models/forward.bson")[:model_to_save]
