using Plots
# plots a ONE DIMENSIONAL regression data set and model
# inputs: model should be a Neuron with linear activation
#         inputs/targets specify a data set
#         resolution is optional and specifies how many
#         points the line should be plotted from
# outputs: a scatter-plot of the data set and a line representing model predictions
function plot_regressor(model::NeuralNetwork, inputs::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real}; resolution::Integer=100)
    @assert length(model.output_layer) == 1
    lb = minimum(inputs)
    ub = maximum(inputs)
    x_range = collect(lb:(ub - lb)/resolution:ub)
    x_range = reshape(x_range, length(x_range), 1)
    plot(x_range, predict(model, x_range)', leg=false)
    scatter!(inputs, targets)
end
# plots a TWO DIMENSIONAL classification data set and model
# inputs: model should be a Neuron with sigmoid activation
#         inputs/targets specify a data set
#         resolution is optional and specifies how many points
#         (along each dimension) the heatmap is generated from.
# outputs: a heatmap of model predictions, overlayed with a scatter-plot of the data set
function plot_classifier(model::NeuralNetwork, inputs::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real};
                         dim::Integer=1, color=:blue, resolution::Integer=100)
    @assert length(model.input_layer) == 2
    lb = minimum(inputs)
    ub = maximum(inputs)
    x_range = lb:(ub - lb)/resolution:ub
    y_range = lb:(ub - lb)/resolution:ub
    data = zeros(length(x_range),length(y_range))
    for (r,x) in enumerate(x_range)
        for (c,y) in enumerate(y_range)
            predict!(model,[y,x]) # why???
            data[r,c] = model.output_layer[dim].a
        end
    end
    heatmap(x_range, y_range, data, c=cgrad([:gray, :white, color]), clim=(0,1))
    for d in 1:size(targets, 2)
        points_labeled_d = inputs[targets[:,d] .== 1, :]
        if d == dim
            scatter!(points_labeled_d[:, 1], points_labeled_d[:, 2], color=color, label="")
        else
            scatter!(points_labeled_d[:, 1], points_labeled_d[:, 2], label="")
        end
    end
    plot!()
end