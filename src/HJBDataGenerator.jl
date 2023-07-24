## Data module.
module Data

using Distributions
using LinearAlgebra

Base.@kwdef struct Domain
  # Spatial domain.
  t0::Float64 = 0.0
  t1::Float64 = 1.0
  tsteps::Int64 = 10
  dt = (t1 - t0) / (tsteps - 1)
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
  v = Array{Float64}(undef, domain.tsteps, domain.xsteps)
  v[end, :] = sample[2, :]
  v[1:end-1, 1] = sample[3, 1:domain.tsteps - 1]
  v[1:end-1, end] = sample[3, domain.tsteps:domain.xsteps-1]
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


