using Distributions: Normal
import Base: copy

abstract type Node end

mutable struct Input <: Node
    a::Float64
end

function copy(other::Input)
    Input(other.a)
end

function Input()
    Input(NaN)
end

mutable struct Neuron <: Node
    weights::Vector{Float64}
    bias::Float64
    activation::Function
    derivative::Function
    a::Float64 # set by predict, used by backpropagation
    δ::Float64 # set by backpropagation, used by update
end

function copy(other::Neuron)
    Neuron(copy(other.weights), other.bias, other.activation, other.derivative, other.a, other.δ)
end

function Neuron(input_dimension::Integer, activation::Function,
                weight_distr=Normal(0,.1), bias_distr=Normal(0,1))
    weights = rand(weight_distr, input_dimension)
    bias = rand(bias_distr)
    derivative = get_derivative(activation)
    Neuron(weights, bias, activation, derivative, NaN, NaN)
end

# Models a densely-connected neural network.
mutable struct NeuralNetwork
    input_layer::Vector{Input}
    hidden_layers::Vector{Vector{Neuron}}
    output_layer::Vector{Neuron}
end

function NeuralNetwork(input_dim::Int64, output_dim::Int64, hidden_dims::Vector,
                       activation_functions::Union{Function, Vector{Function}}=sigmoid_activation;
                       weight_distr=Normal(0,.1), bias_distr=Normal(0,1))
    @assert input_dim > 0
    @assert output_dim > 0
    @assert length(hidden_dims) >= 0
    @assert all(d > 0 for d in hidden_dims)
    
    input_layer = [Input() for i in 1:input_dim]
    if activation_functions isa Function
        activation_functions = [activation_functions for _ in 1:length(hidden_dims)+1]
    end
    hidden_layers = Vector{Vector{Neuron}}()
    for l in 1:length(hidden_dims)
        activation = activation_functions[l]
        num_weights = l == 1 ? input_dim : hidden_dims[l-1]
        layer = [Neuron(num_weights, activation, weight_distr, bias_distr) for _ in 1:hidden_dims[l]]
        push!(hidden_layers, layer)
    end
    activation = activation_functions[end]
    if length(hidden_dims) > 0
        num_weights = hidden_dims[end]
    else
        num_weights = input_dim
    end
    output_layer = [Neuron(num_weights, activation, weight_distr, bias_distr) for _ in 1:output_dim]
    
    NeuralNetwork(input_layer, hidden_layers, output_layer)
end

function copy(other::NeuralNetwork)
    input_layer = [copy(node) for node in other.input_layer]
    hidden_layers = [[copy(node) for node in layer] for layer in other.hidden_layers]
    output_layer = [copy(node) for node in other.output_layer]
    NeuralNetwork(input_layer, hidden_layers, output_layer)
end