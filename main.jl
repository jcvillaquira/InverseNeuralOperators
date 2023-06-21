# Imports.
using NeuralOperators
using Distributions
using LinearAlgebra
using Plots

include("src/DataGenerator.jl")

# Creating objects.
domain = DifferentialOperator.Domain(nsteps = 50)
sampler = DataGenerator.Sampler()

# Generate maps.
forcing = DifferentialOperator.forcing_term(domain)
forward_realization = a -> DifferentialOperator.forward_operator(domain, forcing, a)

# Generate data.
A = DataGenerator.generate_permeability(sampler, domain.x, 1000)
U = mapslices(forward_realization, A; dims = 1)

