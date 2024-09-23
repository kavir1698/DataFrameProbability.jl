module DataFrameProbability

using DataFrames
using StatsBase
using Discretizers
using Distances
using CategoricalArrays

export probability_distribution, marginal_distribution, conditional_distribution, compare_distributions

"""
    probability_distribution(df, focal_column; condition_columns=[], column_info=Dict(), count_occurrences=false)

Calculate the probability distribution of a focal column in a DataFrame, optionally conditioned on other columns.

# Arguments
- `df::DataFrame`: The DataFrame to calculate the probability distribution from.
- `focal_column::Symbol`: The column to calculate the probability distribution for.
- `condition_columns::Vector{Symbol}=[]`: The columns to condition the probability distribution on.
- `column_info::Dict{Symbol, NamedTuple}=Dict()`: A dictionary with column information. Each entry can have fields:
  - `type::Symbol`: `:continuous` or `:categorical` (default: auto-detect)
  - `bins::Union{Int, Vector{<:Real}}`: Number of bins or bin edges for continuous variables (default: auto-bin)
  - `categories::Vector`: All possible categories for categorical variables (default: auto-detect)
- `count_occurrences::Bool=false`: If true, return probabilities based on number of occurrences.

# Returns
- `Tuple{Union{Weights, Dict}, Dict{Symbol, Vector}}`: The calculated probability distribution and a dictionary of all categories for each column.
"""
function probability_distribution(df::DataFrame, focal_column::Symbol;
  condition_columns::Vector{Symbol}=Symbol[],
  column_info::Dict{Symbol,T}=Dict{Symbol,NamedTuple}(),
  count_occurrences::Bool=false) where T<:NamedTuple
  columns = vcat(focal_column, condition_columns)
  processed_df, updated_categories = preprocess_data(df, columns, column_info)

  if isempty(condition_columns)
    dist = calculate_marginal_distribution(processed_df, focal_column, updated_categories[focal_column], count_occurrences)
  else
    dist = calculate_conditional_distribution(processed_df, focal_column, condition_columns, updated_categories, count_occurrences)
  end

  return dist, updated_categories
end

"""
    marginal_distribution(df, column; kwargs...)

Calculate the marginal probability distribution of a column.
"""
marginal_distribution(df::DataFrame, column::Symbol; kwargs...) =
  probability_distribution(df, column; kwargs...)

"""
    conditional_distribution(df, focal_column, condition_columns; kwargs...)

Calculate the conditional probability distribution of a focal column given condition columns.
"""
conditional_distribution(df::DataFrame, focal_column::Symbol, condition_columns::Vector{Symbol}; kwargs...) =
  probability_distribution(df, focal_column; condition_columns=condition_columns, kwargs...)

function preprocess_data(df::DataFrame, columns::Vector{Symbol}, column_info::Dict{Symbol,T}) where {T<:NamedTuple}
  processed_df = select(df, columns)
  updated_categories = Dict{Symbol,Vector}()

  for col in columns
    info = get(column_info, col, (type=:auto,))
    col_type = get(info, :type, :auto)

    if col_type == :auto
      col_type = if eltype(df[!, col]) <: Union{Bool,Missing} || length(unique(df[!, col])) == 1
        :categorical
      elseif eltype(df[!, col]) <: Union{Number,Missing}
        :continuous
      end
    end

    if col_type == :categorical
      processed_df[!, col] = categorical(processed_df[!, col])
      updated_categories[col] = get(info, :categories, unique(processed_df[!, col]))
    elseif col_type == :continuous
      bins = get(info, :bins, nquantile(skipmissing(df[!, col]), min(length(unique(df[!, col])), 30)))
      if isa(bins, Int)
        min_val, max_val = extrema(skipmissing(df[!, col]))
        edges = range(min_val, stop=max_val, length=max(bins + 1, 3))
      else
        # Ensure bin edges are unique and sorted
        edges = sort(unique(bins))
        if length(edges) < 2
          min_val, max_val = extrema(skipmissing(df[!, col]))
          edges = [min(min_val, minimum(edges)), max(max_val, maximum(edges))]
        end
      end
      discretizer = LinearDiscretizer(edges)
      processed_df[!, col] = categorical(encode(discretizer, df[!, col]))
      updated_categories[col] = unique(processed_df[!, col])
    end
  end

  return processed_df, updated_categories
end

function calculate_marginal_distribution(
  df::DataFrame, focal_column::Symbol, categories::Vector, count_occurrences::Bool)
  if count_occurrences && length(categories) == 1
    counts = sum(df[!, focal_column] .== categories[1])
    return Weights([1.0]), counts
  else
    counts = countmap(df[!, focal_column])
    total = sum(values(counts))
    probabilities = [get(counts, category, 0) / total for category in categories]
    return Weights(probabilities)
  end
end

function calculate_conditional_distribution(df::DataFrame, focal_column::Symbol, condition_columns::Vector{Symbol},
  categories::Dict{Symbol,Vector}, count_occurrences::Bool)
  grouped = groupby(df, condition_columns)
  result = Dict()
  focal_categories = categories[focal_column]

  if count_occurrences && length(focal_categories) == 1
    counts = Dict()
    for (key, group) in pairs(grouped)
      counts[key] = sum(group[!, focal_column] .== focal_categories[1])
    end
    total_counts = sum(values(counts))

    # Group by the first condition column
    primary_condition = condition_columns[1]
    for primary_value in categories[primary_condition]
      key = NamedTuple{(primary_condition,)}((primary_value,))
      result[key] = Float64[]
      for (group_key, count) in counts
        if group_key[1] == primary_value
          push!(result[key], count / total_counts)
        end
      end
    end
  else
    # For non-count_occurrences cases, keep the original logic but use NamedTuples for keys
    for (key, group) in pairs(grouped)
      counts = countmap(group[!, focal_column])
      total = sum(values(counts))
      probabilities = [get(counts, category, 0) / total for category in focal_categories]
      named_key = NamedTuple{Tuple(condition_columns)}(key)
      result[named_key] = Weights(probabilities)
    end
  end

  return result
end

"""
    compare_distributions(dist1, dist2; distance=JSDivergence())

Compare two probability distributions using a specified distance metric.

# Arguments
- `dist1`, `dist2`: The probability distributions to compare. Can be either Weights (for marginal distributions) or Dict (for conditional distributions).
- `distance`: The distance metric to use. Defaults to Jensen-Shannon Divergence.

# Returns
- `Union{Float64, Dict}`: The distance between the distributions. Returns a single Float64 for marginal distributions, or a Dict of distances for conditional distributions.
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
        missing_dist = Weights(ones(length(first(values(dist1)))) / length(first(values(dist1))))
        result[key] = haskey(dist1, key) ? evaluate(distance, dist1[key], missing_dist) : evaluate(distance, missing_dist, dist2[key])
      end
    end
    return result
  else
    error("Incompatible distribution types")
  end
end

end # module
