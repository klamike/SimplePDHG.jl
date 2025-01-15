
# contains the problem setting, i.e., the parameter γ as set by the user

mutable struct PDHG_settings{T<:Real}

    # user settings to solve the problem using PDHG 
    # =============================================
    maxit::Int64 # maximum number of iteration
    tol::T # tolerance, i.e., if |||| ≤ tol, we take x to be an optimal solution
    verbose::Bool # whether to print information about the iterates
    freq::Int64 # how often print information about the iterates

    # constructor for the structure, so if user does not specify any particular values, 
    # then we create a setting object with default values
    function PDHG_settings{T}(;maxit, tol, verbose, freq)  where {T<:Real}
        new(maxit, tol, verbose, freq)
    end

    function PDHG_settings(;maxit, tol, verbose, freq)
        new{Float64}(maxit, tol, verbose, freq)
    end
    
end

# default setting
setting = PDHG_settings{Float64}(maxit=100000, tol=1e-4, verbose=false, freq=1000)
