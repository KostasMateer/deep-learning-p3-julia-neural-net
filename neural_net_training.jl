using Random: shuffle

include("neural_net_model.jl")

# finds the model's output on a signle data point
# input: neural network, point (represented as a vector)
# modifies each node in the network by setting node.a
# no return value (the network's output can be found by inspecting output-layer node.a values)
function predict!(model::NeuralNetwork, data_point::AbstractVector{<:Real})
    if size(model.input_layer, 1) != size(data_point, 1)
        error("data_point dimensions don't match size of input layer")
    end
    # Input layer activation(s)
    for i in 1:size(model.input_layer, 1)
        model.input_layer[i].a = data_point[i]
    end
    
    # Hidden layer(s) activation(s)
    if size(model.hidden_layers, 1) > 0
        # For hidden layer 1:
        for i in 1:size(model.hidden_layers[1], 1)
            weights = model.hidden_layers[1][i].weights
            bias = model.hidden_layers[1][i].bias
            x = 0
            for k in 1:size(model.input_layer, 1)
                x += model.input_layer[k].a * weights[k]
            end
            model.hidden_layers[1][i].a = model.hidden_layers[1][i].activation(x + bias)
        end
    end
    if size(model.hidden_layers, 1) > 1
        # For hidden layers 2 through size(hidden_layers, 1)
        for i in 2:size(model.hidden_layers, 1)
            # For each node in hidden layer i
            for j in 1:size(model.hidden_layers[i], 1)
                weights = model.hidden_layers[i][j].weights
                bias = model.hidden_layers[i][j].bias
                x = 0
                for k in 1:size(model.hidden_layers[i-1], 1)
                    x += model.hidden_layers[i-1][k].a * weights[k]
                end
                model.hidden_layers[i][j].a = model.hidden_layers[i][j].activation(x + bias)
            end
        end
    end

    # Output layer activation(s)
    if size(model.hidden_layers, 1) > 0
        # If hidden layer(s) exists
        n = size(model.hidden_layers, 1)
        # Saving last hidden layer to n
        for i in 1:size(model.output_layer, 1)
            weights = model.output_layer[i].weights
            bias = model.output_layer[i].bias
            x = 0
            for j in 1:size(model.hidden_layers[n], 1)
                x += model.hidden_layers[n][j].a * weights[j]
            end
            model.output_layer[i].a = model.output_layer[i].activation(x + bias)
        end
    else
        # If there DNE hidden layer(s)
        for i in 1:size(model.output_layer, 1)
            weights = model.output_layer[i].weights
            bias = model.output_layer[i].bias
            x = 0
            for j in 1:size(model.input_layer, 1)
                x += model.input_layer[j].a * weights[j]
            end
            model.output_layer[i].a = model.output_layer[i].activation(x + bias)
        end
    end
end

# makes predictions on an entire data set without modifying the neural network
# returns an (N x num_output_neurons) array of predictions, one-per row
# The shape of the data set should be (N x num_input_neurons)
function predict(model::NeuralNetwork, data_set::AbstractMatrix{<:Real})
    N = size(model.output_layer, 1)
    num_input_neurons = size(data_set, 1)
    predictions = zeros(N, num_input_neurons)
    model_copy = copy(model)
    
    for i in 1:size(data_set, 1)
        data_point = data_set[i, :]
        # println(data_point)
        predict!(model_copy, data_point)
        output = []
        for node in model_copy.output_layer
            push!(output, node.a)
        end
        predictions[:, i] = output
    end
    predictions
end

# computes the gradient for a single data point, storing the result in node.δ for each node
# it is assumed that predict!() has just been called on this data point,
# so each node's node.a stores the activation
function gradient!(model::NeuralNetwork, target::AbstractVector{<:Real})
    # Gradients for output_layer
    for i in 1:size(model.output_layer, 1)
        error = target[i] - model.output_layer[i].a
        a_deriv = model.output_layer[i].derivative(model.output_layer[i].a)
        model.output_layer[i].δ = -2 * error * a_deriv
    end
    
    # Gradients for hidden_layer(s)
    if size(model.hidden_layers, 1) > 0
    # Gradients for last hidden_layer
        n = size(model.hidden_layers, 1)
        for i in 1:size(model.hidden_layers[n], 1)
            delta = 0
            activation = model.hidden_layers[n][i].a
            deriv = model.hidden_layers[n][i].derivative(activation)
            for j in 1:size(model.output_layer, 1)
                prev_delta = model.output_layer[j].δ
                out_weight = model.output_layer[j].weights[i]
                delta += prev_delta * deriv * out_weight
            end
            model.hidden_layers[n][i].δ = delta
        end

        if size(model.hidden_layers, 1) > 1
        # Gradients for hidden_layers 1 to n-1 
            for i in reverse(1:n-1)
                for j in 1:size(model.hidden_layers[i], 1)
                    delta = 0
                    activation = model.hidden_layers[i][j].a
                    deriv = model.hidden_layers[i][j].derivative(activation)
                    for k in 1:size(model.hidden_layers[i+1][j].weights, 1)
                        prev_delta = model.hidden_layers[i+1][k].δ
                        out_weight = model.hidden_layers[i+1][j].weights[k]
                        delta += prev_delta * deriv * out_weight
                    end
                    model.hidden_layers[i][j].δ = delta
                end
            end
        end    
    end
end

# updates the weights of a neural network
# it is assumed that gradient!() has just been called to update each node.δ
function update!(model::NeuralNetwork, learning_rate::Real)
    # Updating output_layer weights if hidden_layer(s) DNE
    if size(model.hidden_layers, 1) == 0
        for l in 1:size(model.output_layers, 1)
            delta_l = model.output_layer[l].δ
            for k in 1:size(model.output_layer[l].weights, 1)
                activ_k = model.input_layer[k].a
                # Update w(k -> l)
                model.output_layer[l].weights[k] -= learning_rate * delta_l * activ_k
            end
            # Update bias(l)
            model.output_layer[l].bias -= learning_rate * delta_l
        end
    end
    if size(model.hidden_layers, 1) > 0
        # Updating output_layer parameters if hidden_layer(s) exists
        n = size(model.hidden_layers, 1)
        for l in 1:size(model.output_layer, 1)
            delta_l = model.output_layer[l].δ
            for k in 1:size(model.output_layer[l].weights, 1)
                activ_k = model.hidden_layers[n][k].a
                # Update w(k -> l)
                model.output_layer[l].weights[k] -= learning_rate * delta_l * activ_k
            end
            # Update bias(l)
            model.output_layer[l].bias -= learning_rate * delta_l
        end
    
        # Updating first hidden_layer using activations from input_layer
        for l in 1:size(model.hidden_layers[1], 1)
            delta_l = model.hidden_layers[1][l].δ
            for k in 1:size(model.hidden_layers[1][l].weights, 1)
                activ_k = model.input_layer[k].a
                # Update w(k -> l)
                model.hidden_layers[1][l].weights[k] -= learning_rate * delta_l * activ_k
            end
            # Update bias(l)
            model.hidden_layers[1][l].bias -= learning_rate * delta_l
        end
    end
    
    # Updating remaining hidden_layers' parameters if layer(s) exists
    if size(model.hidden_layers, 1) > 1
        for n in 2:size(model.hidden_layers, 1)
            for l in 1:size(model.hidden_layers[n], 1)
                delta_l = model.hidden_layers[n][l].δ
                for k in 1:size(model.hidden_layers[n][l].weights, 1)
                    activ_k = model.hidden_layers[n-1][k].a
                    # Update w(k -> l)
                    model.hidden_layers[n][l].weights[k] -= learning_rate * delta_l * activ_k
                end
                # Update bias(l)
                model.hidden_layers[n][l].bias -= learning_rate * delta_l
            end
        end
    end
end

# trains a neural network using backpropagation
# runs through the data epochs number of times
# each epoch the data is shuffled, then predict!, gradient!, and update! are called on each batch
# learning_rate specifies the gradient-descent step-size
# if losses isa Vector, MSE loss should be logged for the entire data set at the start and after each epoch
function train!(model::NeuralNetwork, inputs::AbstractMatrix{<:Real},
                targets::AbstractMatrix{<:Real}, learning_rate::Real,
                epochs::Integer, batch_size::Integer=1,
                losses::Union{Vector,Nothing}=nothing)
    for _ in 1:epochs
        if batch_size == 1
            for i in 1:size(inputs, 1)
                predict!(model, inputs[i,:])
                gradient!(model, [targets[i]])
                update!(model, learning_rate)
            end
        else
            error("batch size > 1 unimplemented")
        end
        if losses isa Vector
            push!(losses, MSE(predict(model, inputs), targets'))
        end
    end
    losses
end

# mean SSE over all all outputs in a data set
function MSE(predictions::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real})
    mse = 0
    N = size(predictions, 1)
    for i in 1:N
        p, t = predictions[i, :], targets[i, :]
        mse += sum((t .- p).^2)
    end
    (1/N) * mse
end

# finds the fraction of points classified correctly if predictions are rounded to 0/1
# only applicable for classification models WITH 1D OUTPUT
# input: vector of predictions, vector of targets
# output: number
function accuracy(predictions::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real})
    sum(round.(predictions) .== targets) / size(targets, 2)
end