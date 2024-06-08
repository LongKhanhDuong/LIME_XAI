# lime implementation
using XAIBase

struct lime_sle{M} <: AbstractXAIMethod
    model::M
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
