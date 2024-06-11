using MLJ
using Plots
using DataFrames
using StatsBase

# 生成一些示例数据
X = rand(100, 5)
y = X * [0.5, -0.2, 0.1, 0.4, -0.3] + 0.1 * randn(100)

# 转换数据为 DataFrame
df = DataFrame(X, :auto)
df[!, :y] = y

# 训练一个线性模型
model = @load LinearRegressor pkg=MLJLinearModels
mach = machine(model, df[:, 1:5], df[:, :y]) |> fit!

# 定义一个预测函数
function predict_fn(model, X)
    X_df = DataFrame(X, :auto)
    return predict(model, X_df).prediction
end

# 定义一个生成邻域样本的函数
function generate_neighborhood(instance, num_samples=500)
    neighborhood = []
    for i in 1:num_samples
        sample = instance .+ 0.01 * randn(length(instance))
        push!(neighborhood, sample)
    end
    return reduce(hcat, neighborhood)'
end

# 选择一个实例进行解释
instance = X[1, :]

# 生成邻域样本
neighborhood = generate_neighborhood(instance)

# 获取邻域样本的预测
predictions = predict_fn(mach, neighborhood)

# 计算邻域样本与原始实例之间的距离
distances = [sum((neighborhood[i, :] .- instance).^2) for i in 1:size(neighborhood, 1)]

# 使用加权线性回归拟合邻域样本
weights = exp.(-distances / 0.2)
WLS = @load WeightedLeastSquaresRegressor pkg=MLJLinearModels
wls_model = machine(WLS(), DataFrame(neighborhood, :auto), predictions, weights) |> fit!

# 获取解释系数
coefs = wls_model.fitresult.model.coefs[2:end]

# 生成热图
heatmap_data = coefs'
heatmap(heatmap_data, c=:viridis, title="LIME Explanation Heatmap", xlabel="Features", ylabel="Instance")
