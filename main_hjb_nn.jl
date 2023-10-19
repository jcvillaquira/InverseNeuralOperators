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

## Load data.
data = BSON.load("hjb/20230729/data.bson")
xdata = data[:ydata]
ydata = data[:xdata]

## Flatten data
rxdata = reshape(xdata, size(xdata, 1) * size(xdata, 2), :)
rydata = reshape(ydata, size(ydata, 1) * size(ydata, 2), :)

size( rydata ) 

nn = Chain(Dense(size(rxdata, 1) => 400),
           Dense(400 => 200),
           Dense(200 => 200),
           Dense(200 => 200),
           Dense(200 => 200),
           Dense(200 => 100),
           Dense(100 => size(rydata, 1)))

data_train, data_test = splitobs((Float32.(rxdata), Float32.(rydata)), at = 0.9)
loader_train, loader_test = DataLoader(data_train), DataLoader(data_test)
data = collect.((loader_train, loader_test))

optimiser = Flux.Optimiser(WeightDecay(1.0f-4), Flux.Adam(1.0f-3))
learner = Learner(nn, data, optimiser, l₂loss)

learner.model

fit!(learner, 100)


model = BSON.load("hjb/20230730/inverse_definitive.bson" )
Chain(model[:test_model].learner.model)
Chain( OperatorKernel(32 => 32, (16,), FourierTransform, gelu) )

X = Array{Float64}(undef, 22, 5, 3)
X .= 3.0

c( xdata )
c( [3.0 for x in 1:22] )

