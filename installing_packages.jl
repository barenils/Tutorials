# Created becouse i reinstalled Julia becouse of a bug. 
# 
using Pkg

function manage_packages(packages; update=false, precompile=true)
    for pkg in packages
        Pkg.add(pkg)
    end

    if update
        Pkg.update()
    end

    if precompile
        Pkg.precompile()
    end
end

manage_packages(["DataFrames", "CUDA", "Statistics", "LinearAlgebra", "LibPQ", "BenchmarkTools", "Random", "Distributions", "Dates", "Plots", "Flux", "JSON", "HTTP"], update=true)   

using LinearAlgebra

## Just checking if Threads are working propperly again

function create_and_multiply_matrices(n, size)
    matrices = [rand(Float32, size, size) for _ in 1:n] # Creating matrixes 
    results = Vector{Matrix{Float32}}(undef, n-1) # Storing matrix
    Threads.@threads for i in 1:(n-1)
        results[i] = matrices[i] * matrices[i+1] # Perform matrix multiplication in parallel to check if multi threading works again
    end

    return results
end

result_matrices = create_and_multiply_matrices(1000, 1000)