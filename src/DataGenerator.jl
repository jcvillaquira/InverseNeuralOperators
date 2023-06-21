# Generates all the data needed.
module DataGenerator

using Distributions
using LinearAlgebra

Base.@kwdef struct Sampler
  τ::Float64 = 1.0
  N::Int64 = 10
  σ = [1.0 / (π^2 * n^2 + τ^2) for n in range(0, N - 1)]
  D = MvNormal(Diagonal(σ))
  E = [ t -> cos(n * π * t) for n in range(0, N - 1)]
end

function base_to_function(b, E)
  return t -> sum( k[1] * k[2](t) for k in zip(b, E) )
end

function generate_permeability(S, x, data_size)
  Ab = rand(S.D, data_size)
  Ab[1, :] .= 0.1 .+ abs.( Ab[1, :] )
  base_to_function_aux = b -> base_to_function(b, S.E).(x)
  return mapslices(base_to_function_aux, Ab; dims = 1)
end

export Sampler
export base_to_function
export generate_permeability

end

# Differential equation.
module DifferentialOperator
Base.@kwdef struct Domain
  x0::Float64 = 0.0
  x1::Float64 = 1.0
  nsteps::Int64
  dx = (x1 - x0) / nsteps
  x = range(x0, x1, length = nsteps)
end

function forcing_term(domain)
  f = collect(range(1, domain.nsteps))
  return (f .- sum(f) / domain.nsteps) .* domain.dx
end

function forward_operator(domain, f, a)
  ∫f = cumsum(f) * domain.dx
  inv_a = 1.0 ./ a
  return -cumsum( inv_a .* ∫f ) * domain.dx
end

export Domain
export forcing_term
export forward_operator

end
