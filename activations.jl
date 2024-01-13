function linear_activation(input)
    return input
end

function linear_derivative(activation)
    return 1
end

function sigmoid_activation(input)
    1 / (1 + exp(-input))
end

function sigmoid_derivative(activation)
    activation * (1 - activation)
end

function tanh_activation(input)
    (1 + exp(-2*input))/(1 - exp(-2*input))
end

function tanh_derivative(activation)
    1 - activation^2
end

function ReLU_activation(input)
    max(0, input)
end

function ReLU_derivative(activation)
    if activation < 0
        return 0
    end
    return 1
end

# example usage: get_derivative(ReLU_activation) returns ReLU_derivative
function get_derivative(activation_function::Function)
    function_string = String(Symbol(activation_function))
    @assert endswith(function_string, "_activation")
    getfield(Main, Symbol(function_string[1:end-10] * "derivative"))
end