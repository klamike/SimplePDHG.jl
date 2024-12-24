## Module SimplePDHG

module SimplePDHG

import MathOptInterface as MOI
import SparseArrays
import LinearAlgebra

include("settings.jl")
include("initialization.jl")

# Given a vector x in R^n, write code to project on to the positive orthant {x ∣ x ≥ 0}. The projection should be done in place, i.e. the function should modify the input vector x. The function should return the number of negative entries in x before projection.
function project_nonnegative!(x::AbstractVector{T}) where T<:Real
    for i in eachindex(x)
        x[i] = max(zero(T), x[i])
    end
end

# Write the same function as project_nonnegative, but this time the solution will be assigned to a new variable
function project_nonnegative(x::AbstractVector{T}) where T<:Real
    y = zeros(length(x))
    for i in eachindex(y)
        y[i] = max(zero(T), y[i])
    end
    return y
end

# Create the termination function
# which will take input A, b, c, x, y
# and will compute
# ϵ = ||Ax-b||/(1+||b||) + ||project_nonnegative(A'y-c)||/(1+||c||) + ||c'x - b'y||/(1+|c'x|+|b'y|)
# In the function called tolerance_LP, change the type of A to SparseArrays.SparseMatrixCSC{T,Int} 
function tolerance_LP(A::SparseArrays.SparseMatrixCSC{T,Int}, b::AbstractVector{T}, c::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where T<:Real
    ϵ = LinearAlgebra.norm(A*x-b,2)/(1+LinearAlgebra.norm(b,2)) + LinearAlgebra.norm(project_nonnegative(-A'*y-c), 2)/(1+LinearAlgebra.norm(c,2)) + LinearAlgebra.norm(c'*x + b'*y, 2)/(1+abs(c'*x)+abs(b'y))
    tolerance_pc = LinearAlgebra.norm(A*x - b, 2)
    tolerance_x = LinearAlgebra.norm(project_nonnegative(-x), 2)
    tolerance_z = LinearAlgebra.norm(project_nonnegative(-z), 2)
    return ϵ, tolerance_pc, tolerance_x, tolerance_z
end

# Write one iteration of PDHG
function PDHG_iteration!(problem::LP_Data, state::PDHG_state)

    # unpack the current state information
    x_k = state.x
    y_k = state.y
    z_k = state.z
    k = state.k

    # compute the next iterate

    # compute x_k_plus_1 = x_k - η*problem.A'*y_k- η*c
    x_k_plus_1 = x_k - state.η*z_k

    # project on to the positive orthant
    project_nonnegative!(x_k_plus_1)

    # compute y_k_plus_1 = y + τ*A*(2*x_k_plus_1 - x_k) - τ*b
    y_k_plus_1 = y_k + state.τ*problem.A*(2*x_k_plus_1 - x_k) - state.τ*problem.b
    z_k_plus_1 = problem.c + problem.A'y_k_plus_1

    # load the computed values in the PDHG_state
    state.x = x_k_plus_1
    state.y = y_k_plus_1
    state.z = z_k_plus_1
    state.k = k + 1

    # return the updated state
    return state

end

function PDHG_solver(problem::LP_Data, setting::PDHG_settings)
    if setting.verbose == true
        @info "*******************************************************"
        @info "SimplePDHG https://github.com/Shuvomoy/SimplePDHG.jl"
        @info "*******************************************************"
    end
    # this is the function that the end user will use to solve a particular problem, internally it is using the previously defined types and functions to run 
    # PDHG algorithm
    # create the intial state
    start_time = time()
    state = PDHG_state(problem)

    tc, tpc, tx, tz =  tolerance_LP(problem.A, problem.b, problem.c, state.x, state.y, state.z)
    # NOTE: only tc is used for termination
    
    ## time to run the loop
    while  (state.k < setting.maxit) &&  tc > setting.tol
        # print information if verbose = true
        if setting.verbose == true
            if mod(state.k, setting.freq) == 0
                @info "$(state.k) | $(problem.c'*state.x) | opt $(tc) | tpc $(tpc) | tx $(tx) | tz $(tz)"
            end
        end
        # compute a new state
        state = PDHG_iteration!(problem, state)
        tc, tpc, tx, tz =  tolerance_LP(problem.A, problem.b, problem.c, state.x, state.y, state.z)
    end
    ts = time() - start_time

    if setting.verbose == true
        # print information regarding the final state
        @info "=================================================================="

        @info "[🌹 ]                  FINAL ITERATION INFORMATION"

        @info "=================================================================="
        
        @info "$(state.k) | $(problem.c'*state.x) | opt $(tc) | tpc $(tpc) | tx $(tx) | tz $(tz)"
    end

    if tc ≤ setting.tol
        state.st = MOI.OPTIMAL
    else
        state.st = MOI.ITERATION_LIMIT
    end

    if tpc ≤ setting.tol && tx ≤ setting.tol
        state.sp = MOI.FEASIBLE_POINT
    else
        state.sp = MOI.NEARLY_FEASIBLE_POINT
    end

    if tz ≤ setting.tol
        state.sd = MOI.FEASIBLE_POINT
    else
        state.sd = MOI.NEARLY_FEASIBLE_POINT
    end

    return state, tc, ts
    
end


function solve_pdhg(
    A::SparseArrays.SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
    c::Vector{Float64},
    settings::PDHG_settings,
)::Tuple{MOI.TerminationStatusCode,MOI.ResultStatusCode,MOI.ResultStatusCode,Vector{Float64},Vector{Float64},Vector{Float64}}

    # create the data object
    m, n = size(A)
    problem = LP_Data(c, A, b, m, n)
    # solve the problem
    return PDHG_solver(problem, settings)
end

include("MOI.jl")

# export all the objects (functions, struct and so on) defined in this module in comma seperated form 

export LP_Data, PDHG_settings, PDHG_state, PDHG_iteration!, project_nonnegative!, project_nonnegative, tolerance_LP, PDHG_solver, solve_pdhg


end # end module