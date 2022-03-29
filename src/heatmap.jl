# NOTE: Heatmapping assumes Flux's WHCN convention (width, height, color channels, batch size).

const HEATMAPPING_PRESETS = Dict{Symbol,Tuple{ColorScheme,Symbol,Symbol}}(
    # Analyzer => (colorscheme, reduce, normalize)
    :LRP => (ColorSchemes.bwr, :sum, :centered), # attribution
    :InputTimesGradient => (ColorSchemes.bwr, :sum, :centered), # attribution
    :Gradient => (ColorSchemes.grays, :norm, :extrema), # gradient
)

"""
    heatmap(expl::Explanation; kwargs...)
    heatmap(attr::AbstractArray; kwargs...)

    heatmap(input, analyzer::AbstractXAIMethod)
    heatmap(input, analyzer::AbstractXAIMethod, neuron_selection::Int)

Visualize explanation.
Assumes Flux's WHCN convention (width, height, color channels, batch size).

## Keyword arguments
- `cs::ColorScheme`: ColorScheme that is applied.
    When calling `heatmap` with an `Explanation` or analyzer, the method default is selected.
    When calling `heatmap` with an array, the default is `ColorSchemes.bwr`.
- `reduce::Symbol`: How the color channels are reduced to a single number to apply a colorscheme.
    The following methods can be selected, which are then applied over the color channels
    for each "pixel" in the attribution:
    - `:sum`: sum up color channels
    - `:norm`: compute 2-norm over the color channels
    - `:maxabs`: compute `maximum(abs, x)` over the color channels in
    When calling `heatmap` with an `Explanation` or analyzer, the method default is selected.
    When calling `heatmap` with an array, the default is `:sum`.
- `normalize::Symbol`: How the color channel reduced heatmap is normalized before the colorscheme is applied.
    Can be either `:extrema` or `:centered`.
    When calling `heatmap` with an `Explanation` or analyzer, the method default is selected.
    When calling `heatmap` with an array, the default for use with the `bwr` colorscheme is `:centered`.
- `permute::Bool`: Whether to flip W&H input channels. Default is `true`.
- `unpack_singleton::Bool`: When heatmapping a batch with a single sample, setting `unpack_singleton=true`
    will return an image instead of an Vector containing a single image.

**Note:** these keyword arguments can't be used when calling `heatmap` with an analyzer.
"""
function heatmap(
    attr::AbstractArray{T,N};
    cs::ColorScheme=ColorSchemes.bwr,
    reduce::Symbol=:sum,
    normalize::Symbol=:centered,
    permute::Bool=true,
    unpack_singleton::Bool=true,
) where {T,N}
    N != 4 && throw(
        DomainError(
            N,
            """heatmap assumes Flux's WHCN convention (width, height, color channels, batch size) for the input.
            Please reshape your attribution to match this format if your model doesn't adhere to this convention.""",
        ),
    )
    if unpack_singleton && size(attr, 4) == 1
        return _heatmap(attr[:, :, :, 1], cs, reduce, normalize, permute)
    end
    return map(a -> _heatmap(a, cs, reduce, normalize, permute), eachslice(attr; dims=4))
end

# Use HEATMAPPING_PRESETS for default kwargs when dispatching on Explanation
function heatmap(expl::Explanation; permute::Bool=true, kwargs...)
    _cs, _reduce, _normalize = HEATMAPPING_PRESETS[expl.analyzer]
    return heatmap(
        expl.attribution;
        reduce=get(kwargs, :reduce, _reduce),
        normalize=get(kwargs, :normalize, _normalize),
        cs=get(kwargs, :cs, _cs),
        permute=permute,
    )
end
# Analyze & heatmap in one go
function heatmap(input, analyzer::AbstractXAIMethod, args...; kwargs...)
    return heatmap(analyze(input, analyzer, args...; kwargs...))
end

# Lower level function that is mapped along batch dimension
function _heatmap(
    attr::AbstractArray{T,3},
    cs::ColorScheme,
    reduce::Symbol,
    normalize::Symbol,
    permute::Bool,
) where {T<:Real}
    img = _normalize(dropdims(_reduce(attr, reduce); dims=3), normalize)
    permute && (img = permutedims(img))
    return ColorSchemes.get(cs, img)
end

# Normalize activations across pixels
function _normalize(attr, method::Symbol)
    if method == :centered
        min, max = (-1, 1) .* maximum(abs, attr)
    elseif method == :extrema
        min, max = extrema(attr)
    else
        throw(
            ArgumentError(
                "Color scheme normalizer :$method not supported, `normalize` should be :extrema or :centered",
            ),
        )
    end
    return (attr .- min) / (max - min)
end

# Reduce attributions across color channels into a single scalar – assumes WHCN convention
function _reduce(attr::AbstractArray{T,3}, method::Symbol) where {T}
    if size(attr, 3) == 1 # nothing to reduce
        return attr
    elseif method == :sum
        return reduce(+, attr; dims=3)
    elseif method == :maxabs
        return reduce((c...) -> maximum(abs.(c)), attr; dims=3, init=zero(T))
    elseif method == :norm
        return reduce((c...) -> sqrt(sum(c .^ 2)), attr; dims=3, init=zero(T))
    end
    throw(
        ArgumentError(
            "Color channel reducer :$method not supported, `reduce` should be :maxabs, :sum or :norm",
        ),
    )
end
