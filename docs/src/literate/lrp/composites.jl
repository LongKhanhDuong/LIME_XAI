# # Assigning LRP rules to layers
# In this example, we will show how to assign LRP rules to specific layers.
# For this purpose, we first define a small VGG-like convolutional neural network:
using ExplainableAI
using Flux

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1),
        Conv((3, 3), 16 => 16, relu; pad=1),
        MaxPool((2, 2)),
    ),
    Chain(
        Flux.flatten,
        Dense(1024 => 512, relu),
        Dropout(0.5),
        Dense(512 => 100, relu)
    ),
);

# ## [Manually assigning rules](@id manual_rules)
# When creating an LRP-analyzer, we can assign individual rules to each layer.
# As we can see above, our model is a `Chain` of two Flux `Chain`s.
# Using [`flatten_model`](@ref), we can flatten the model into a single `Chain`:
model_flat = flatten_model(model)

# This allows us to define an LRP analyzer using an array of rules
# matching the length of the Flux chain:

rules = [
    FlatRule(),
    ZPlusRule(),
    ZeroRule(),
    ZPlusRule(),
    ZPlusRule(),
    ZeroRule(),
    PassRule(),
    EpsilonRule(),
    PassRule(),
    EpsilonRule(),
];

# The `LRP` analyzer will show a summary of how layers and rules got matched:
LRP(model_flat, rules)

# However, this approach only works for models that can be fully flattened.
# For unflattened models and models containing `Parallel` layers, we can compose rules using
# [`ChainTuple`](@ref)s and [`ParallelTuple`](@ref)s which match the model structure:
rules = ChainTuple(
    ChainTuple(
        FlatRule(),
        ZPlusRule(),
        ZeroRule(),
        ZPlusRule(),
        ZPlusRule(),
        ZeroRule()
    ),
    ChainTuple(
        PassRule(),
        EpsilonRule(),
        PassRule(),
        EpsilonRule(),
    ),
)

analyzer = LRP(model, rules; flatten=false)

#md # !!! note "Keyword argument `flatten`"
#md #
#md #     We used the `LRP` keyword argument `flatten=false` to showcase
#md #     that the structure of the model can be preserved.
#md #     For performance reasons, the default `flatten=true` is recommended.

# ## Custom composites
# Instead of manually defining a list of rules, we can also define a [`Composite`](@ref).
# A composite contructs a list of LRP-rules by sequentially applying
# [Composite primitives](@ref composite_primitive_api) it contains.
#
# To obtain the same set of rules as in the previous example, we can define
composite = Composite(
    GlobalTypeMap(
        Conv                 => ZPlusRule(),   # apply ZPlusRule on all Conv layers
        Dense                => EpsilonRule(), # apply EpsilonRule on all Dense layers
        Dropout              => PassRule(),    # apply PassRule on all Dropout layers
        MaxPool              => ZeroRule(),    # apply ZeroRule on all MaxPool layers
        typeof(Flux.flatten) => PassRule(),    # apply PassRule on all flatten layers
    ),
    FirstLayerMap(
        FlatRule()
    ), # apply FlatRule on the first layer
);

# We now construct an LRP analyzer from `composite`
analyzer = LRP(model, composite; flatten=false)

# As you can see, this analyzer contains the same rules as our previous one.
# To compute rules for a model without creating an analyzer, use [`lrp_rules`](@ref):
lrp_rules(model, composite)

# ## Composite primitives
# The following [Composite primitives](@ref composite_primitive_api) can used to construct a [`Composite`](@ref).
#
# To apply a single rule, use:
# * [`LayerMap`](@ref) to apply a rule to the `n`-th layer of a model
# * [`GlobalMap`](@ref) to apply a rule to all layers
# * [`RangeMap`](@ref) to apply a rule to a positional range of layers
# * [`FirstLayerMap`](@ref) to apply a rule to the first layer
# * [`LastLayerMap`](@ref) to apply a rule to the last layer
#
# To apply a set of rules to layers based on their type, use:
# * [`GlobalTypeMap`](@ref) to apply a dictionary that maps layer types to LRP-rules
# * [`RangeTypeMap`](@ref) for a `TypeMap` on generalized ranges
# * [`FirstLayerTypeMap`](@ref) for a `TypeMap` on the first layer of a model
# * [`LastLayerTypeMap`](@ref) for a `TypeMap` on the last layer
# * [`FirstNTypeMap`](@ref) for a `TypeMap` on the first `n` layers
#
# Primitives are called sequentially in the order the `Composite` was created with
# and overwrite rules specified by previous primitives.

# ## Composite presets
# A list of implemented default composites can be found under
# [composite presets](@ref default_composite_api) in the API reference,
# e.g. the [`EpsilonPlusFlat`](@ref) composite:
composite = EpsilonPlusFlat()
#
analyzer = LRP(model, composite; flatten=false)