# lime implementation
using XAIBase
using Random

struct lime_sle{M} <: AbstractXAIMethod
    model::M
end

# function: creating N sample points around x_dash
function sampling_x_dash(x_dash, num_samples=100, std_dev=0.1)
z_dash = [x_dash .+ randn(length(x_dash)) .* std_dev for _ in 1:num_samples]
return z_dash
end

# function calculate the distance of the original x und x_dash
function calc_dist(x, x_dash, std_dist=1.0)
    return exp(-norm(x - x_dash)^2 / (std_dist^2))
end

# function calculate the weights regarding the distance
function calc_weights(x, x_dash, std_dist=1.0)
    weights = [calc_dist(x, x_dash[i,:][1]) for i in 1:size(x_dash, 1)]
    return weights
end


function (method::lime_sle)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)                        # y = f(x)
    output_selection = output_selector(output)          # relevant output
  
#### Compute VJP at the Points of the output_selector
    v = zero(output)                                    # vector with zeros
    v[output_selection] .= 1                            # ones at the relevant indices
    val = only(back(v))                                 # VJP to get the gradient - v*(dy/dx)
###
    return Explanation(val, output, output_selection, :lime_sle, :attribution, nothing)
end
