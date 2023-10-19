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

## Hyper params
modes = (8,)
λ = 1.0f-4
η = 1.0f-3
epochs = 100
optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))

## Forward model.
### DataLoaders.
data_train_f, data_test_f = splitobs((Float32.(xdata), Float32.(ydata)), at = 0.9)
loader_train_f, loader_test_f = DataLoader(data_train_f), DataLoader(data_test_f)
data_f = collect.((loader_train_f, loader_test_f))

### Model.
channels_f = (size(xdata, 1), 16, 16, 16, 16, 16, 32, size(ydata, 1))
model_f = FourierNeuralOperator(ch = channels_f, modes = modes, σ = gelu)
learner_f = Learner(model_f, data_f, optimiser, l₂loss)
errors_f = Array{Float32}(undef, 3, epochs)
for k in 1:epochs
  epoch!(learner_f, TrainingPhase(), learner_f.data.training)
  errors_f[1, k] = l₂loss( learner_f.model( data_train_f[1] ), data_train_f[2] )
  epoch!(learner_f, ValidationPhase(), learner_f.data.validation)
  errors_f[2, k] = l₂loss( learner_f.model( data_test_f[1] ), data_test_f[2] )
  errors_f[3, k] = l₂loss( learner_f.model( xdata ), ydata )
end

## Inverse model.
### DataLoaders.
data_train_i, data_test_i = splitobs((Float64.(ydata), Float64.(xdata)), at = 0.9)
loader_train_i, loader_test_i = DataLoader(data_train_i), DataLoader(data_test_i)
data_i = collect.((loader_train_i, loader_test_i))

### Model.
channels_i = (size(ydata, 1), 16, 16, 16, 16, 16, 32, size(xdata, 1))
model_i = FourierNeuralOperator(ch = channels_i, modes = modes, σ = gelu)
learner_i = Learner(model_i, data_i, optimiser, l₂loss)
errors_i = Array{Float64}(undef, 3, epochs)
for k in 1:epochs
  epoch!(learner_i, TrainingPhase(), learner_i.data.training)
  errors_i[1, k] = l₂loss( learner_i.model( data_train_i[1] ), data_train_i[2] )
  epoch!(learner_i, ValidationPhase(), learner_i.data.validation)
  errors_i[2, k] = l₂loss( learner_i.model( data_test_i[1] ), data_test_i[2] )
  errors_i[3, k] = l₂loss( learner_i.model( ydata ), xdata )
end

## Save output.
model_ff = learner_f.model |> cpu
model_ii = learner_i.model |> cpu

@save "hjb/20230729/data.bson" xdata ydata
@save "hjb/20230729/forward.bson" errors_f model_ff
@save "hjb/20230729/inverse.bson" errors_i model_ii

## Load output.
data = BSON.load("hjb/20230729/data.bson" )
forward = BSON.load("hjb/20230729/forward.bson" )
inverse = BSON.load("hjb/20230729/inverse.bson" )

lines(forward[:errors_f][2, :])

# Plots and validations
y_pred = forward[:model_ff]( data[:xdata] )
y_real = data[:ydata]

colormap = :blackbody


k = 1
y_pred_ = y_pred[:, :, end-k]
y_real_ = y_real[:, :, end-k]
T = Float32.( domain.T )
Ω = Float32.( domain.Ω )
fig = Figure()
ax = Axis3(fig[1, 1], xlabel = "Tiempo", ylabel = "y(t)")
lines!(ax, [Point3f(ty[1], 1.0, ty[2]) for ty in zip(T, y_real_[:, end])], color = :blue, linewidth = 3)
lines!(ax, [Point3f(ty[1], -1.0, ty[2]) for ty in zip(T, y_real_[:, 1])], color = :blue, linewidth = 3)
lines!(ax, [Point3f(1.0, xy[1], xy[2]) for xy in zip(Ω, y_real_[end, :])], color = :blue, linewidth = 3)
surface!(ax, [t for t in T, x in Ω], [x for t in T, x in Ω], y_pred_, colormap = (:blackbody, 0.8))
fig

compare_boundary_conditions(y_pred, y_real_, domain)

@save "hjb/20230729/tuning.bson" errors_i model_ii
















## Hyper params
modes = (8,)
λ = 1.0f-4
η = 1.0f-3
epochs = 100
optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))
xdata = data[:xdata]
ydata = data[:ydata]

### Model.

possible_channels = [(size(ydata, 1), 32, 32, 32, 32, 32, 64, size(xdata, 1)), (size(ydata, 1), 64, 64, 64, 64, 64, 128, size(xdata, 1))]
possible_modes = [(8,), (16,)]
Λ = [1.0f-4, 2.0f-4]
Η = [1.0f-3, 2.0f-3]
epochs = 100

iterator_params = Iterators.product(Λ, Η, possible_modes, possible_channels)
iterator_params = reshape(collect(iterator_params), :)
models = Array{Any}(undef, length(iterator_params))
errors = Array{Float64}(undef, length(iterator_params), epochs)

data_train_i, data_test_i = splitobs((Float64.(ydata), Float64.(xdata)), at = 0.9)
loader_train_i, loader_test_i = DataLoader(data_train_i), DataLoader(data_test_i)
data_i = collect.((loader_train_i, loader_test_i))

for (j, params) in enumerate(iterator_params)
  λ, η, modes, channel = params
  model = FourierNeuralOperator(ch = channel, modes = modes, σ = gelu)
  optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))
  learner = Learner(model, data_i, optimiser, l₂loss)
  for k in 1:epochs
    epoch!(learner, TrainingPhase(), learner.data.training)
    epoch!(learner, ValidationPhase(), learner.data.validation)
    errors[j, k] = l₂loss( learner.model( ydata ), xdata )
  end
  models[j] = learner.model
end

k = 4

iterator_params = iterator_params[1:5]
models = models[1:5]
errors = errors[1:5, :]

@save "hjb/20230729/tuning.bson" iterator_params models errors



channels_i = (size(ydata, 1), 32, 32, 32, 32, 32, 64, size(xdata, 1))
model_i = FourierNeuralOperator(ch = channels_i, modes = (8,), σ = gelu)
learner_i = Learner(model_i, data_i, optimiser, l₂loss)
errors_i = Array{Float64}(undef, 3, epochs)

model_i(ydata)

## Inverse model.
### DataLoaders.
data_train_i, data_test_i = splitobs((Float64.(ydata), Float64.(xdata)), at = 0.9)
loader_train_i, loader_test_i = DataLoader(data_train_i), DataLoader(data_test_i)
data_i = collect.((loader_train_i, loader_test_i))


for k in 1:epochs
  epoch!(learner_i, TrainingPhase(), learner_i.data.training)
  errors_i[1, k] = l₂loss( learner_i.model( data_train_i[1] ), data_train_i[2] )
  epoch!(learner_i, ValidationPhase(), learner_i.data.validation)
  errors_i[2, k] = l₂loss( learner_i.model( data_test_i[1] ), data_test_i[2] )
  errors_i[3, k] = l₂loss( learner_i.model( ydata ), xdata )
end
















































################################################################################
# Explicit formula #############################################################
################################################################################

# Populate data.
for k in range(1, n)
  ℓ = Data.generate_function(domain)
  φ = Data.generate_function(domain)
  fℓ = linear_interpolation(domain.Ω, cumsum(ℓ))
  fφ = linear_interpolation(domain.Ω, φ)
  xdata[1, :, k] = ℓ
  xdata[2, :, k] = φ
  ydata[:, :, k] = Operator.cost(domain, fℓ, fφ)
end

ℓ = Data.generate_function(domain)
φ = (rand(domain.xsteps) .- 0.5) .^ 2
∫ℓ = linear_interpolation(domain.Ω, cumsum(ℓ) .* domain.dx)
fφ = linear_interpolation(domain.Ω, φ)
φ[1] = 0.0
φ[end] = 0.0
t = 0.5
x = -0.1

if abs(x) >= t
  return fℓ(-abs(x))
end

remaining = 1.0 - t
x₀ = x - remaining
x₁ = x + remaining

projection_interior = domain.Ω[abs.(domain.Ω .- x) .< remaining]
projection = vcat([x₀], projection_interior, [x₁])

# Supposing that x < 0.
y = -0.1

function arrival_cost(y, x, x₀, x₁, fφ, ∫ℓ)
  xm₀ = ( y + x₀ ) / 2.0 # Turning point.
  rc₀ = ∫ℓ(y) - ∫ℓ(xm₀) + ∫ℓ(x) - ∫ℓ(xm₀) # Running cost.
  xm₁ = ( y + x₁ ) / 2.0 # Turning point.
  rc₁ = ∫ℓ(xm₁) - ∫ℓ(x) + ∫ℓ(xm₁) - ∫ℓ(y) # Running cost
  return fφ(y) + min(rc₀, rc₁)
end

data_train, data_test = splitobs((Float32.(xdata), Float32.(ydata)), at = 0.9)
loader_train, loader_test = DataLoader(data_train), DataLoader(data_test)
data = collect.((loader_train, loader_test))

# Create model.
model = FourierNeuralOperator(ch = (2, 16, 16, 16, 16, 16, 32, domain.tsteps), modes = (8,), σ = gelu)
optimiser = Flux.Optimiser(WeightDecay(1.0f-4), Flux.Adam(1.0f-3))
learner = Learner(model, data, optimiser, l₂loss)
fit!(learner, 20)

# Validations
model( xdata[:, :, [1]] ) 
ydata[:, :, [1]]
learner.model

################################################################################
# Trying to solve the boundary value problem ###################################
################################################################################

u0 = φ
function difference(f)
  f_left = circshift(f, -1)
  f_right = circshift(f, 1)
  grad = (f_left - f_right) / (2 * dx)
  grad[1] = (f[2] - f[1]) / dx
  grad[end] = (f[end] - f[end-1]) / dx
  return abs.(grad)
end

function hjb!(du, u, p, t)
  du[:] = -( difference(u) .- ℓ )
end

function bc!(residual, u, p, t)
  residual[1] = norm(u[1, :])
  residual[2] = norm(u[end, :])
end

tspan = (0.0, 1.0)
bvp = BVProblem(hjb!, bc!, u0, tspan)
sol = solve(bvp, dt = 0.01);

plot(sol.u)
plot(u0)

