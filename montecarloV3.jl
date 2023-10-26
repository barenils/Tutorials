using MarketData, DataFrames, LibPQ, Statistics, Plots, CUDA, CUDA.CURAND, Random, BenchmarkTools, Distributions, Dates

function dl_marketdata(ticker_list)
    out_put = DataFrame()
    list_of_tickers = ticker_list
    for i in 1:length(list_of_tickers)
        ticker = list_of_tickers[i]
        data_download = yahoo(list_of_tickers[i]) |> DataFrame
        data_download.ticker .= ticker
        out_put = vcat(out_put, data_download)
    end
    return(out_put)
end

ticker_list = (
    "V", "PG", "UNH", "MA", "DIS", "HD", "BAC", "VZ", "ADBE", "CRM",
    "MRK", "KO", "NKE", "PFE", "WMT", "CSCO", "CMCSA", "PEP", "XOM", "ABBV"
)
all_stocks = dl_marketdata(ticker_list)
DataFrames.rename!(all_stocks, lowercase.(names(all_stocks)))

function compute_gbm_params(sub_df::SubDataFrame)
    cleaned_df = dropmissing(sub_df, :adjclose)
    log_returns = diff(log.(Float32.(cleaned_df.adjclose)))
    μ = mean(log_returns) # drift
    σ = std(log_returns) #volatility
    idx = argmax(skipmissing(cleaned_df.timestamp))
    adjclose = cleaned_df.adjclose[idx]
    return (ticker=first(cleaned_df.ticker), μ=μ, σ=σ, adjclose=adjclose)
end

grouped = groupby(all_stocks, :ticker)
gbm_params = combine(grouped, compute_gbm_params)


################ RUN ON SINGLE CPU CORE ########################

function montecarlo(data, stock_select)
    data = data[stock_select,:]

    T = 252  # for one trading year
    n_sims = 10000
    future_prices = zeros(T, n_sims)
    S0 = data.adjclose  # last observed price
    μ = data.μ
    σ = data.σ

    for sim in 1:n_sims
        future_prices[1, sim] = S0
        for t in 2:T
            z = randn()  # random normal value
            future_prices[t, sim] = future_prices[t-1, sim] * exp((μ - (σ^2) / 2) + σ * z)
        end
    end
    return future_prices
end

#out_cpu = montecarlo(gbm_params, 1)

################### CUDA IMPLEMENTATION ###########################
function montecarlo_gpu(data,stock_select)
    data = data[stock_select,:]
    T = 252
    n_sims = 10000

    # Transfer initial data to GPU 
    d_S0 = Float32(data.adjclose)
    d_μ = Float32(data.μ)
    d_σ = Float32(data.σ)

    # Create a matrix for future prices on the GPU
    d_future_prices = CUDA.zeros(Float32, T, n_sims)

    # Generate random numbers directly on GPU
    d_random_nums = CuArray(randn(Float32, T, n_sims))

    # Define the kernel
    function kernel(d_future_prices, d_S0, d_μ, d_σ, d_random_nums)
        sim = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        if sim > n_sims
            return
        end

        d_future_prices[1, sim] = d_S0
        for t in 2:T
            z = d_random_nums[t, sim]
            d_future_prices[t, sim] = d_future_prices[t-1, sim] * exp((d_μ - (d_σ^2) / 2) + d_σ * z)
        end

        return
    end

    # Launch the kernel with 2D blocks and threads
    threads_per_block = 1024  
    blocks_x = ceil(Int, n_sims / threads_per_block)
    blocks_y = 1
    @cuda threads=(threads_per_block, blocks_y) blocks=(blocks_x, blocks_y) kernel(d_future_prices, d_S0, d_μ, d_σ, d_random_nums)

    return Array(d_future_prices)  # Convert back to CPU array before returning
end

out_gpu = montecarlo_gpu(gbm_params, 1)

####################### CROSS CHECKING FOR ZEROS IN DF, INCASE ANY ERRORS ###################
contains_zeros = any(x -> x == 0, out_gpu) # Check if we have any zeros 

##########################  BENCHMARKING  ###################################################

#BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

#@benchmark montecarlo(gbm_params, 1)
#@benchmark montecarlo_gpu(gbm_params, 1)

# As shown we can greatly increase our Monte Carlo simulations using CUDA with aproximatly 10 times the speed compared to single a single CPU core

############################# visualization of simulations #################################################


function plot_matrix(matrix)
    p = plot(legend=false, title="Monte Carlo Simulation Paths", xlabel="Time", ylabel="Value")
    for col in 1:size(matrix, 2)
        plot!(matrix[:, col])
    end
    display(p)
end


#plot_matrix(out_gpu[1:end,300:350])
#plot_matrix(out_cpu[1:end,300:350])

function plot_percentiles(matrix)
    percentiles = 1:1:99  # 1% to 99%
    n_timepoints = size(matrix, 1)
    p = plot(legend=false, title="Monte Carlo Simulation Percentile Paths", xlabel="Time", ylabel="Value")

    # Plot each percentile
    for percentile in percentiles
        path = [quantile(matrix[t, :], percentile/100) for t in 1:n_timepoints]
        plot!(path, alpha=0.5)  # set alpha for better visualization
    end

    # Plot the median
    median_path = [median(matrix[t, :]) for t in 1:n_timepoints]
    plot!(median_path, color=:black, linewidth=2, label="Median")

    display(p)
end

# Assuming your_matrix is the matrix you want to visualize

#plot_percentiles(out_gpu)
#plot_percentiles(out_cpu)

############### Running montecarlo for multiple stocks ###################

function mc_multi_stock(data)
    results_dict = Dict{String, Matrix{Float32}}()
    for i in 1:nrow(data)
        ticker_id = data.ticker[i]  # Assuming you have a column named "ticker_id" in gbm_params
        results_dict[ticker_id] = montecarlo(data,i)
    end 
    return results_dict
end


function mc_multi_stock_gpu(data)
    results_dict = Dict{String, Matrix{Float32}}()
    for i in 1:nrow(data)
        ticker_id = data.ticker[i]  # Assuming you have a column named "ticker_id" in gbm_params
        results_dict[ticker_id] = montecarlo_gpu(data,i)
    end 
    return results_dict
end


########## benchmarking again we se about 20-25 times increase in calculation speeds using the monte carlo gpu model ###############

#@benchmark mc_multi_stock(gbm_params)
#@benchmark mc_multi_stock_gpu(gbm_params)

############
multi_stock = mc_multi_stock_gpu(gbm_params)

#plot_matrix(multi_stock["MSTR"][1:252,1:100])
#plot_matrix(multi_stock["AAPL"][1:252,1:100])



############## Calculating returs with cuda for all stocks ############################

function calculate_returns_kernel(returns, prices)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(prices, 2)
        for t in 2:size(prices, 1)
            returns[t, i] = ((prices[t, i] - prices[t-1, i]) / prices[t-1, i])+1
        end
    end
    return
end

function calculate_returns(data)
    data = data |> CuArray
    returns = CUDA.zeros(Float32, size(data, 1), size(data, 2))
    @cuda threads= 1024 blocks=ceil(Int, size(data, 2) / 1024) calculate_returns_kernel(returns, data)
    return Array(returns)  # Convert back to CPU array before returning
end

function rets_cuda(data) 
    results_dict = Dict{String, Matrix{Float32}}()
    for (ticker, stock_rets) in data
        results_dict[ticker] = calculate_returns(stock_rets)
    end
    return results_dict
end

rets_data = rets_cuda(multi_stock)

################### Sanity check. To investigate if cuda computations where correct ##################################

function sanity_rets(multi_stock, rets_data)
    ticker_ids = keys(multi_stock)
    random_ticker = rand(ticker_ids)
    random_column = rand(1:size(multi_stock[random_ticker], 2))
    
    actual_returns = diff(Float32.(multi_stock[random_ticker][:, random_column])) ./ Float32.(multi_stock[random_ticker][1:end-1, random_column]) .+ 1
    actual_returns = vcat(0,actual_returns)
    expected_returns = rets_data[random_ticker][:, random_column]
    
    is_sane = isapprox(actual_returns, expected_returns, atol=1e-8)
    return is_sane
end

function check_sanity(multi_stock, rets_data, n_checks)
    for y in 1:n_checks
        is_sane = sanity_rets(multi_stock, rets_data)
        if is_sane == false
            print("ERROR! at $y")
        else
        print("$is_sane ")
        end
    end
end

check_sanity(multi_stock, rets_data, 1000)

##### Now we know our return calculations are correct 
##### Setting first line to 1 from 0 in all matrixes

# Sett first row to 1 (easier for calculations)
function set_first_to_one(data)
    results_dict = Dict{String, Matrix{Float32}}()
    for (ticker, stock_rets) in data
        new_matrix = copy(stock_rets)
        new_matrix[1,:] .= 1f0
        results_dict[ticker] = new_matrix
    end
    return results_dict
end

modified_data = set_first_to_one(rets_data)


##### Next we will see our daily returns cumalative

function cumalative_rets(data)
    initial_investment = 100.0  # Initial investment
    investment_values = Vector{Float64}([initial_investment])
    holding = []

    for i in 1:size(data)[2]
        daily_return = data[:, i]  # Assuming returns are in the first row
        next_value = investment_values[end] * (daily_return)
        push!(holding, next_value)
    end

    holding = hcat(holding...)
    return holding
end


function cal_cum_rets(data::Dict{String, Matrix{Float32}}) 
    results_dict = Dict{String, Matrix{Float32}}()
    for (ticker, stock_rets) in data
        results_dict[ticker] = cumalative_rets(stock_rets)
    end
    return results_dict
end


c_rets = cal_cum_rets(modified_data)

######## Now we plot the histograms for the final cumalative return

function plot_hists(data)
    for (ticker, c_rets) in data
        plot_data = c_rets[252,1:end]
        plt = histogram(plot_data, title=ticker, bins=50, alpha=0.6, legend=false)
        display(plt)
    end
end

plot_hists(c_rets)

##### Next step is to calculate expected returns at the end of the year and volatility for each stock 

function over_view_last_day(data)
    holding_frame = DataFrame(
        ticker = String[], 
        expected_returns = Float64[], 
        volatilities = Float64[],
        skewness = Float64[],
        kurtosis = Float64[],
        max_drawdown = Float64[],
        value_at_risk = Float64[],
        Q1 = Float64[],  # First Quartile
        Q2 = Float64[],  # Median or Second Quartile
        Q3 = Float64[]   # Third Quartile
        # Add other standalone indicators as necessary
    )

    for (ticker, c_rets) in data
        rets = c_rets[252,1:end]  # Assuming the 252nd row contains the final returns
        temp_df = DataFrame(
            ticker = ticker,
            expected_returns = mean(rets),
            volatilities = std(rets),
            skewness = skewness(rets),
            kurtosis = kurtosis(rets),
            max_drawdown = calculate_max_drawdown(rets),
            value_at_risk = calculate_var(rets),
            Q1 = quantile(rets, 0.25),
            Q2 = median(rets),
            Q3 = quantile(rets, 0.75)
        )
        holding_frame = vcat(holding_frame, temp_df)
    end
    
    return holding_frame
end

function calculate_var(returns, confidence_level=0.05)
    sorted_returns = sort(returns)
    var_index = ceil(Int, confidence_level * length(sorted_returns))
    return -sorted_returns[var_index]  # Return as positive value for loss
end

function calculate_max_drawdown(rets)
    peak = rets[1]
    max_drawdown = 0.0
    for ret in rets
        peak = max(peak, ret)
        max_drawdown = min(max_drawdown, ret - peak)
    end
    return -max_drawdown
end


over_view_last_day(c_rets)