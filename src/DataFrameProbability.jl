module DataFrameProbability

using DataFrames
using StatsBase
using Discretizers
using Distances
using CategoricalArrays

export calculate_probability_distribution, compare_distributions

"""
    calculate_probability_distribution(df::DataFrame, focal_column::Symbol, condition_columns::Vector{Symbol} = Symbol[]; n_bins::Union{Int, Vector{Int}} = 10, column_types::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}(), column_ranges::Dict{Symbol, Tuple{Real, Real}} = Dict{Symbol, Tuple{Real, Real}}(), all_categories::Dict{Symbol, Vector} = Dict{Symbol, Vector}()) -> Tuple{Union{Weights, Dict}, Dict{Symbol, Vector}}

Calculate the probability distribution of a focal column in a DataFrame, optionally conditioned on other columns.

# Arguments
- `df::DataFrame`: The DataFrame to calculate the probability distribution from.
- `focal_column::Symbol`: The column to calculate the probability distribution for.
- `condition_columns::Vector{Symbol}`: The columns to condition the probability distribution on. If empty, the marginal distribution of the focal column is calculated.
- `n_bins::UUnion{Int,Dict{Symbol,Int}}=5`: The number of bins to use for discretization. If an integer is provided, it is used for all columns. If a dictionary is provided, each element is used for the corresponding column.
- `column_types::Dict{Symbol, Symbol}`: A dictionary mapping column names to their types. The types can be `:continuous` or `:categorical`. If a column is not found in the dictionary, it is considered `:continuous` by default.
- `column_ranges::Dict{Symbol, Tuple{Real, Real}}`: A dictionary mapping column names to their min and max possible value. The ranges are used for discretization. If a column is not found in the dictionary, the minimum and maximum values of the column are used.
- `all_categories::Dict{Symbol, Vector}`: A dictionary mapping column names to all possible categories for the focal column. If not provided, it will be calculated from the data.

# Returns
- `Tuple{Union{Weights, Dict}, Dict{Symbol, Vector}}`: The calculated probability distribution and a dictionary of all categories for each column.
"""
function calculate_probability_distribution(
  df::DataFrame, focal_column::Symbol, condition_columns::Vector{Symbol}=Symbol[];
  n_bins::Union{Int,Dict{Symbol,Int}}= 5,
  column_types::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
  column_ranges::Dict{Symbol,Tuple{Real,Real}}=Dict{Symbol,Tuple{Real,Real}}(),
  all_categories::Dict{Symbol,Vector}=Dict{Symbol,Vector}())

  # 1. Preprocess the data
  processed_df, updated_all_categories = preprocess_data(
    df, [focal_column; condition_columns]; n_bins=n_bins, column_types=column_types,
    column_ranges=column_ranges, all_categories=all_categories)

  # 2. Calculate the distribution
  if isempty(condition_columns)
    dist = calculate_marginal_distribution(
      processed_df, focal_column, updated_all_categories[focal_column])
    return dist, updated_all_categories
  else
    dist = calculate_conditional_distribution(
      processed_df, focal_column, condition_columns, updated_all_categories)
    return dist, updated_all_categories
  end
end

function preprocess_data(
  df::DataFrame, columns::Vector{Symbol}; n_bins::Union{Int,Dict{Symbol,Int}},
  column_types::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
  column_ranges::Dict{Symbol,Tuple{Real,Real}}=Dict{Symbol,Tuple{Real,Real}}(),
  all_categories::Dict{Symbol,Vector}=Dict{Symbol,Vector}())
  processed_df = select(df, columns)
  updated_all_categories = copy(all_categories)

  @assert length(columns) == length(n_bins) || length(n_bins) == 1 "n_bins must be a scalar or a vector of the same length as columns"

  for (i, col) in enumerate(columns)
    bin = isa(n_bins, Int) ? n_bins : get(n_bins, col, 5)  # default to 5 bins if not specified in the dictionary

    if get(column_types, col, :continuous) == :categorical || eltype(df[!, col]) <: Bool
      processed_df[!, col] = categorical(processed_df[!, col])
      if !haskey(updated_all_categories, col)
        updated_all_categories[col] = unique(processed_df[!, col])
      end
    elseif eltype(df[!, col]) <: Number
      min_val, max_val = get(column_ranges, col, (minimum(df[!, col]), maximum(df[!, col])))
      discretizer = LinearDiscretizer(range(min_val, max_val; length=bin + 1))
      processed_df[!, col] = categorical(encode(discretizer, df[!, col]))
      if !haskey(updated_all_categories, col)
        updated_all_categories[col] = unique(processed_df[!, col])
      end
    end
  end

  return processed_df, updated_all_categories
end

function calculate_marginal_distribution(
  df::DataFrame, focal_column::Symbol, all_categories::Vector)
  counts = countmap(df[!, focal_column])
  total = sum(values(counts))

  probabilities = [get(counts, category, 0) / total for category in all_categories]

  return Weights(probabilities)
end

function calculate_conditional_distribution(
  df::DataFrame, focal_column::Symbol, condition_columns::Vector{Symbol},
  all_categories::Dict{Symbol,Vector})
  grouped = groupby(df, condition_columns)
  result = Dict()

  focal_categories = all_categories[focal_column]

  for (key, group) in pairs(grouped)
    counts = countmap(group[!, focal_column])
    total = sum(values(counts))

    probabilities = [get(counts, category, 0) / total for category in focal_categories]

    result[key] = Weights(probabilities)
  end

  return result
end

"""
    compare_distributions(dist1::Union{Weights, Dict}, dist2::Union{Weights, Dict}) -> Union{Float64, Dict}

Compare two probability distributions using Jensen-Shannon Divergence.

# Arguments
- `dist1`, `dist2`: The probability distributions to compare. Can be either Weights (for marginal distributions) or Dict (for conditional distributions).
- `distance`: The distance metric to use. Defaults to Jensen-Shannon Divergence. Can be any distance metric from the Distances.jl package. Note that you can get the JS distance by taking the square root of the JSDivergence. Another good option is `HellingerDist()`.

# Returns
- `Union{Float64, Dict}`: The Jensen-Shannon Divergence between the distributions. Returns a single Float64 for marginal distributions, or a Dict of divergences for conditional distributions.
"""
function compare_distributions(dist1::Union{Weights,Dict}, dist2::Union{Weights,Dict}; distance=JSDivergence())
  if isa(dist1, Weights) && isa(dist2, Weights)
    return evaluate(distance, dist1, dist2)
  elseif isa(dist1, Dict) && isa(dist2, Dict)
    result = Dict()
    all_keys = union(keys(dist1), keys(dist2))
    for key in all_keys
      if haskey(dist1, key) && haskey(dist2, key)
        result[key] = evaluate(distance, dist1[key], dist2[key])
      else
        # Handle missing conditions by using a uniform distribution
        missing_dist = Weights(ones(length(first(dist1).second)) /
                               length(first(dist1).second))
        if haskey(dist1, key)
          result[key] = evaluate(distance, dist1[key], missing_dist)
        else
          result[key] = evaluate(distance, missing_dist, dist2[key])
        end
      end
    end
    return result
  else
    error("Incompatible distribution types")
  end
end

end # module
