# Imports
## Data Imports.
using NeuralOperators
using GLMakie
using LinearAlgebra
using Interpolations

## Flux Imports
using MLUtils
using Flux
using FluxTraining

## Save and load models.
using BSON
using BSON: @save, @load

## Import own modules and structs.
include("src/HJBDataGenerator.jl")

# Prepare inputs for model
## Create domain and input functions.
domain = Data.Domain(tsteps = 22)
D, B = Data.generate_basis(12, 1.0)
N = 20_000

## Generate data.
xdata = Sampler.sample_function(D, B, domain, 3, N)
xdata[3, domain.xsteps, :] .= 0.0
ydata = mapslices(x -> ValueFunction.apply_value_function(x, domain), xdata; dims = [1, 2])

## Load data if necessary.
data = BSON.load("hjb/20230729/data.bson")
forward = BSON.load("hjb/20230729/forward.bson")
inverse = BSON.load("hjb/20230729/inverse.bson")

## Hyper parameter space.
possible_channels = [(16, 16, 16, 16, 16, 32),
                     (32, 32, 32, 32, 32, 64)]
                     # (64, 64, 64, 64, 64, 128)]
possible_modes = [(8,), (16,)]
Λ = [1.0f-4, 2.0f-4]
Η = [1.0f-3, 2.0f-3]
iterator_params = Iterators.product(Λ, Η, possible_modes, possible_channels)
iterator_params = reshape(collect(iterator_params), :)
errors = (true, true, false)
epochs = 100

## Create data_loader.
NN = 2_000
data_loader = Model.Loader(Float32.(data[:ydata][:, :, 1:NN]), Float32.(data[:xdata][:, :, 1:NN]))

## Create Fourier Neural Oprator model.
models = Array{Any}(undef, length(iterator_params))
for (j, (λ, η, modes, channels)) in enumerate(iterator_params)
  models[j] = Model.NOModel(loader = data_loader, inner_channels = channels, λ = λ, η = η, modes = modes)
end
saved_loss = Array{Float32}(undef, length(models), sum(errors), epochs)

for k in 1:epochs
  for (j, model) in enumerate(models)
    saved_loss[j, :, k] = Model.train_epoch!(model, data_loader, errors)
  end
end
@save "hjb/20230730/tuning_.bson" saved_loss models iterator_params

## Definitive model.
NN = 20_000
epochs = 200
data_loader = Model.Loader(Float32.(data[:ydata][:, :, 1:NN]), Float32.(data[:xdata][:, :, 1:NN]))
test_model = Model.NOModel(loader = data_loader, inner_channels = (32, 32, 32, 32, 32, 64), modes = (16,), λ = 0.0001, η = 0.001)
test_params = Dict("channels" => (32, 32, 32, 32, 32, 64), "modes" => (16,), "lambda" => 0.0001, "eta" => 0.001)
errors = (true, true, true)
test_error = Array{Float32}(undef, sum(errors), epochs)
for k in 1:epochs
  test_error[:, k] = Model.train_epoch!(test_model, data_loader, errors)
end
@save "hjb/20230730/inverse_definitive__.bson" test_error test_model test_params

data = BSON.load("hjb/20230729/data.bson")
tuning = BSON.load("hjb/20230730/tuning.bson")

lines( test_error[1, :] )
lines!( test_error[2, :] )

## Plots and validations
x_real = data[:xdata]
y_real = data[:ydata]
x_pred = test_model.model(y_real)
y_pred = mapslices(x -> ValueFunction.apply_value_function(x, domain), x_pred; dims = [1, 2])








k = 25
y_real_ = y_real[:, :, end-k]
y_pred_ = y_pred[:, :, end-k]
T = Float32.( domain.T )
Ω = Float32.( domain.Ω )
fig = Figure()
ax = Axis3(fig[1, 1], xlabel = "Tiempo", ylabel = "y(t)")
lines!(ax, [Point3f(ty[1], 1.0, ty[2]) for ty in zip(T, y_real_[:, end])], color = :blue, linewidth = 3)
lines!(ax, [Point3f(ty[1], -1.0, ty[2]) for ty in zip(T, y_real_[:, 1])], color = :blue, linewidth = 3)
lines!(ax, [Point3f(1.0, xy[1], xy[2]) for xy in zip(Ω, y_real_[end, :])], color = :blue, linewidth = 3)
surface!(ax, [t for t in T, x in Ω], [x for t in T, x in Ω], y_pred_, colormap = (:blackbody, 0.8))
fig

