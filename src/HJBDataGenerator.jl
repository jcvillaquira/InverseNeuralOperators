## Data module.
module Data

using Distributions
using LinearAlgebra

Base.@kwdef struct Domain
  # Spatial domain.
  t0::Float64 = 0.0
  t1::Float64 = 1.0
  tsteps::Int64 = 10
  dt::Float64 = (t1 - t0) / (tsteps - 1)
  T = range(t0, t1, length = tsteps)
  x0::Float64 = -1.0
  x1::Float64 = 1.0
  xsteps::Int64 = 2 * tsteps - 1
  dx = (x1 - x0) / (xsteps - 1)
  Ω = range(x0, x1, length = xsteps)
end

function generate_basis(N, τ)
  # Generate a basis of eigenvectors of 1/(-Δ+τ²).
  σΔ = [ π^2 * n^2 / 4.0 for n in range(0, N - 1) ]
  σRΔ = [ 1.0 / (λ + τ^2) for λ in σΔ ]
  D = MvNormal(Diagonal(σRΔ))
  B = [t -> cos(n * π * (t + 1.0) / 2.0) for n in range(0, N - 1)]
  return (D, B)
end

export Domain
export generate_basis

end

## Sampler module.
module Sampler

function base_to_function(c, B)
  # Create function with basis B and coefficents c.
  return t -> sum( k[1] * k[2](t) for k in zip(c, B) )
end

function generate_concave_function(domain)
  # Generate random concave function defined on domain.Ω.
  y₀ = rand( domain.xsteps ) .^ 2
  y₁ = cumsum(abs.([0.0, diff(y₀)...]))
  y₂ = min.(y₁, reverse(y₁))
  return y₂

end

function sample_function(D, B::Array, domain, n::Int64, N::Int64)
  evalΩ = f -> f.(domain.Ω)
  coeffs = rand(D, (n, N))
  funcs = base_to_function.(coeffs, Ref(B))
  funcs_array = permutedims(reshape(funcs, size(funcs)..., 1), [1, 3, 2])
  return mapslices(v -> evalΩ.(v)[1], funcs_array; dims = 2)
end

export generate_concave_function
export base_to_function
export sample_function

end

## Forward operator module.
module ValueFunction

function apply_value_function(sample::Matrix, domain)
  # Compute value function using dynamic programming.
  v = Array{Float64}(undef, domain.tsteps, domain.xsteps)
  v[1:end-1, 1] = sample[3, 1:domain.tsteps - 1]
  v[1:end-1, end] = sample[3, domain.tsteps:domain.xsteps-1]
  v[end, :] = sample[2, :]
  ℓ = sample[1, :]
  for nt in reverse(range(1, domain.tsteps - 1))
    for nx in range(2, domain.xsteps - 1)
      ℓ_cost = v[nt+1, nx-1:nx+1] + domain.dx * ( ℓ[nx-1:nx+1] .+ ℓ[nx] ) ./ 2.0
      v[nt, nx] = min(ℓ_cost...)
    end
  end
  return v
end

export apply_value_function

end

## Module models.
module Model
  using MLUtils
  using Flux
  using FluxTraining
  using NeuralOperators

  Base.@kwdef struct Loader
    # Create dataloader.
    xdata
    ydata
    data_train
    data_test
    data
    function Loader(xdata, ydata, ratio::Float64 = 0.9)
      data_train, data_test = splitobs((Float32.(xdata), Float32.(ydata)), at = ratio)
      loader_train, loader_test = DataLoader(data_train), DataLoader(data_test)
      data = collect.((loader_train, loader_test))
      return new(xdata, ydata, data_train, data_test, data)
    end
  end

  Base.@kwdef struct NOModel
    # Fourier neural operator and learner.
    loader
    inner_channels::Tuple = (16, 16, 16, 16, 16, 32)
    channels::Tuple = (size(loader.xdata, 1), inner_channels... , size(loader.ydata, 1))
    modes::Tuple = (8,)
    λ::Float64 = 1.0f-4
    η::Float64 = 1.0f-3
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η))
    model = FourierNeuralOperator(ch = channels, modes = modes, σ = gelu)
    learner = Learner(model, loader.data, optimiser, l₂loss)
  end

  function train_epoch!(nomodel, loader, errors = (true, true, true))
    # Train model one epoch based on data and return l₂loss.
    error₁ = error₂ = error₃ = NaN
    epoch!(nomodel.learner, TrainingPhase(), nomodel.learner.data.training)
    epoch!(nomodel.learner, ValidationPhase(), nomodel.learner.data.validation)
    if errors[1]
      error₁ = l₂loss( nomodel.learner.model( loader.data_train[1] ), loader.data_train[2] )
    end
    if errors[2]
      error₂ = l₂loss( nomodel.learner.model( loader.data_test[1] ), loader.data_test[2] )
    end
    if errors[3]
      error₃ = l₂loss( nomodel.learner.model( loader.xdata ), loader.ydata )
    end
    return [error₁, error₂, error₃][collect(errors)]
  end

  export DataLoader
  export NOModel
  export train_epoch!

end


































################################################################################
# Abandoned code ###############################################################
################################################################################

## Operator module.
module Operator
# Running cost.
function running_cost(t, x, fℓ)
  x₀ = abs(x)
  return fℓ(-max(t, x₀))
end
# Cost associated to reaching the boundary.
function boundary_cost(t, x, fφ, domain)
  if abs(x) >= t
    return 0.0
  end
  remaining = 1.0 - t
  projection = abs.(domain.Ω .- x) .< remaining
  nodes_projection = vcat([x - remaining, x + remaining], domain.Ω[projection])
  return min(fφ.(nodes_projection)...)
end
# Total cost.
function cost(domain, fℓ, fφ)
  total_cost(tx) = boundary_cost(tx..., fφ, domain) + running_cost(tx..., fℓ)
  return total_cost.(Iterators.product(domain.T, domain.Ω))
end
export running_cost
export boundary_cost
end
