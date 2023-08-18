using ExplainableAI
using ExplainableAI: check_lrp_compat
using Suppressor
err = ErrorException("Unknown layer or activation function found in model")

# TODO: test checks on unflattened model

# Flux layers
unknown_function(x) = x
@test check_lrp_compat(Chain(Dense(2, 2, relu)))
@test_throws err check_lrp_compat(Chain(Dense(2, 2, softmax)); verbose=false)
@test_throws err check_lrp_compat(Chain(unknown_function); verbose=false)
@test_throws err @suppress check_lrp_compat(
    Chain(
        unknown_function,
        Chain(unknown_function),
        Parallel(+, unknown_function, unknown_function),
    );
    verbose=false,
)

# Custom layers
## Test using a simple wrapper
struct MyLayer{T}
    x::T
end
TestLayer = MyLayer(Dense(2, 2, relu))
@test_throws err check_lrp_compat(Chain(TestLayer); verbose=false)
@test_throws err LRP(Chain(TestLayer); verbose=false)
@test_nowarn LRP(Chain(TestLayer); skip_checks=true)

## Test should pass after registering the layer
LRP_CONFIG.supports_layer(::MyLayer) = true
@test check_lrp_compat(Chain(TestLayer); verbose=false) == true
@test_nowarn LRP(Chain(TestLayer))

## ...repeat for layers that are functions
@test_throws err check_lrp_compat(Chain(unknown_function); verbose=false)
LRP_CONFIG.supports_layer(::typeof(unknown_function)) = true
@test check_lrp_compat(Chain(unknown_function); verbose=false) == true

## ...repeat for activation functions
@test_throws err check_lrp_compat(Chain(Dense(2, 2, unknown_function)); verbose=false)
LRP_CONFIG.supports_activation(::typeof(unknown_function)) = true
@test check_lrp_compat(Chain(Dense(2, 2, unknown_function)); verbose=false) == true
