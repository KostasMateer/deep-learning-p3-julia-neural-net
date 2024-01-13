using Distributions: Uniform, MvNormal
using Random: shuffle
using LinearAlgebra: I

# generates a random data set with a linear trend plus noise
# returns two arrays: (x,y)
function generate_regression_data(num_points, input_dimension, output_dimension; lb=-1, ub=1, var=.1)
    inputs = rand(Uniform(lb,ub), num_points, input_dimension)
    targets = Array{Float64,2}(undef, num_points, output_dimension)
    for d in 1:output_dimension
        coefficients = rand(Uniform(lb/input_dimension, ub/input_dimension), input_dimension)
        constant = rand(Uniform(lb,ub))
        targets[:,d] = inputs * coefficients .+ constant .+ rand(Normal(0,var), num_points)
    end
    return inputs, targets
end

# generates a random data set with clusters of points assigned to some number of labels
# returns two arrays: (x,y)
function generate_classification_data(num_points, input_dimension; num_labels=2, num_clusters=2, lb=-1, ub=1, var=.1, covar=0)
    covar_matrix = zeros(input_dimension, input_dimension)
    covar_matrix[I(input_dimension)] .= var
    covar_matrix[.!I(input_dimension)] .= covar
    points_per_cluster = num_points ÷ num_clusters
    num_points = points_per_cluster * num_clusters
    centroids = [rand(Uniform(lb,ub), input_dimension) for c in 1:num_clusters]
    inputs = Matrix{Float64}(undef, num_points, input_dimension)
    targets = zeros(num_points, num_labels)
    for c in 1:num_clusters
        label = c % num_labels + 1
        targets[points_per_cluster*(c-1)+1 : points_per_cluster*c, label] .= 1
        inputs[points_per_cluster*(c-1)+1 : points_per_cluster*c, :] = rand(MvNormal(centroids[c], covar_matrix), points_per_cluster)'
    end
    random_indices = shuffle(1:num_points)
    return inputs[random_indices,:], targets[random_indices,:]
end
