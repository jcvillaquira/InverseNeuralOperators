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
data = BSON.load("hjb/20230729/data.bson" )
forward = BSON.load("hjb/20230729/forward.bson" )
inverse = BSON.load("hjb/20230729/inverse.bson" )

## Create dataloaders.
data_loader = Model.Loader(xdata, ydata)

## Create Fourier Neural Oprator model.
fno_model = Model.NOModel(loader = data_loader)

## Train one epoch.
Model.train_epoch!(fno_model, data_loader)

