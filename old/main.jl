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
N = 70_000
nsteps = 80

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
data_f_train, data_f_test = splitobs((Float32.(x_data), Float32.(y_data)), at = ratio)
loader_f_train = DataLoader(data_f_train, batchsize = 500, shuffle = false)
loader_f_test = DataLoader(data_f_test, batchsize = 500, shuffle = false)
data_f = collect.((loader_f_train, loader_f_test))

# Create and train model.
model_f = FourierNeuralOperator(ch = (1, 64, 64, 64, 64, 64, 128, 1), σ = gelu)
learner_f = Learner(model_f, data_f, optimiser, loss_func)

# Train forward model.
epochs = 1
fit!(learner_f, epochs)

# Inverse data loaders.
data_i_train, data_i_test = splitobs((Float32.(y_data), Float32.(x_data)), at = ratio)
loader_i_train = DataLoader(data_i_train, batchsize = 500, shuffle = false)
loader_i_test = DataLoader(data_i_test, batchsize = 500, shuffle = false)
data_i = collect.((loader_i_train, loader_i_test))

# Create inverse model.
model_i = FourierNeuralOperator(ch = (1, 64, 64, 64, 64, 64, 128, 1), σ = gelu)
learner_i = Learner(model_i, data_i, optimiser, loss_func)

# Train inverse model.
epochs = 1
fit!(learner_i, epochs)
forward_model = learner_f.model |> cpu
inverse_model = learner_i.model |> cpu
@save "models/forward.bson" forward_model
@save "models/inverse.bson" inverse_model
# exit()

# data[train or test][batch][x or y data][variable, mesh, observation]
# f : A -> U
# i : U -> A

batch = 2
a_real = data_f[2][batch][1] 
u_real = data_f[2][batch][2] 

# Plots for forward operator.
j = 17
u_to_plot = u_real[1, :, j]
plot(domain.x, u_to_plot)
plot!(domain.x, model_f(a_real)[1, :, j])

# Plots for inverse operator.
j = 17

a_to_plot = a_real[1, :, j]
plot(domain.x, a_to_plot)
plot!(domain.x, model_i(u_real)[1, :, j])

# Save and load.
forward_model = learner_f.model |> cpu
inverse_model = learner_i.model |> cpu
@save "models/forward.bson" forward_model
@save "models/inverse.bson" inverse_model

model_f = BSON.load("models/forward.bson")[:forward_model]
model_i = BSON.load("models/inverse.bson")[:inverse_model]
