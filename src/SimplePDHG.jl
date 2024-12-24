## Module SimplePDHG

module SimplePDHG

import MathOptInterface as MOI
import SparseArrays
import LinearAlgebra

# Given a vector x in R^n, write code to project on to the positive orthant {x ∣ x ≥ 0}. The projection should be done in place, i.e. the function should modify the input vector x. The function should return the number of negative entries in x before projection.

# project_nonnegative!(x)

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

# Write a function that will take a large matrix which is of type SparseArrays.SparseMatrixCSC{T,Int} and compute its maximum singular value using any Julia package that is suitable to solve this problem

function max_singular_value_PDHG(A::SparseArrays.SparseMatrixCSC{T,Int}) where T<:Real
    σmaxA = LinearAlgebra.norm(A,2)
    return σmaxA
end

# Write a julia struct that will take c which is a vector of length n, A which is a matrix of size m x n, and b which is a vector of length m

struct LP_Data{T<:AbstractFloat, I <: Integer} 
    c::Vector{T} # cost vector of length n
    A::SparseArrays.SparseMatrixCSC{T,Int} # date matrix of size m x n
    b::Vector{T} # resource vector of length m
    m::I # number of rows of A
    n::I # number of columns of A
end

# contains the problem setting, i.e., the parameter γ as set by the user

mutable struct PDHG_settings

    # user settings to solve the problem using PDHG 
    # =============================================
    maxit::Int64 # maximum number of iteration
    tol::Float64 # tolerance, i.e., if |||| ≤ tol, we take x to be an optimal solution
    verbose::Bool # whether to print information about the iterates
    freq::Int64 # how often print information about the iterates

    # constructor for the structure, so if user does not specify any particular values, 
    # then we create a setting object with default values
    function PDHG_settings(;maxit, tol, verbose, freq) 
        new(maxit, tol, verbose, freq)
    end
    
end

# test PDHG_settings

# default setting
setting = PDHG_settings(maxit=100000, tol=1e-4, verbose=false, freq=1000)


# construct PDHG state that is going to contain the necessary information to completely describe one iteration of PDHG algorithm

mutable struct PDHG_state{T <: AbstractVecOrMat{<: Real}, I <: Integer} # contains information regarding one iterattion sequence
    x::T # iterate x_n
    y::T # iterate y_n
    z::T # iterate z_n
    η::Float64 # step size
    τ::Float64 # step size
    k::I # iteration counter  
    st::MOI.TerminationStatusCode # termination status
    sp::MOI.ResultStatusCode # primal status
    sd::MOI.ResultStatusCode # dual status
end

# We need a method to construct the initial value of the type PDHG_state

function PDHG_state(problem::LP_Data)
    n = problem.n
    m = problem.m
    σmaxA = max_singular_value_PDHG(problem.A)
    η_preli = (1/(σmaxA)) - 1e-6
    τ_preli = (1/(σmaxA)) - 1e-6
    x_0 = zeros(n)
    y_0 = zeros(m)
    return PDHG_state(x_0, y_0, problem.c, η_preli, τ_preli, 1, MOI.OPTIMIZE_NOT_CALLED, MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS)
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
        state =  PDHG_iteration!(problem, state)
        tc, tpc, tx, tz =  tolerance_LP(problem.A, problem.b, problem.c, state.x, state.y, state.z)
    end

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

    return state, tc
    
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
    state, _ = PDHG_solver(problem, settings)

    return state.st, state.sp, state.sd, state.x, state.y, state.z
end



MOI.Utilities.@product_of_sets(RHS, MOI.Zeros)

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        Vector{Float64},
        RHS{Float64},
    },
}

function MOI.add_constrained_variables(
    model::OptimizerCache,
    set::MOI.Nonnegatives,
)
    x = MOI.add_variables(model, MOI.dimension(set))
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Nonnegatives}(x[1].value)
    return x, ci
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    x_primal::Dict{MOI.VariableIndex,Float64}
    y_dual::Dict{MOI.ConstraintIndex,Float64}
    z_dual::Dict{MOI.ConstraintIndex,Float64}
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    settings::PDHG_settings

    function Optimizer()
        return new(
            Dict{MOI.VariableIndex,Float64}(),
            Dict{MOI.ConstraintIndex,Float64}(),
            Dict{MOI.ConstraintIndex,Float64}(),
            MOI.OPTIMIZE_NOT_CALLED,
            MOI.UNKNOWN_RESULT_STATUS,
            MOI.UNKNOWN_RESULT_STATUS,
            deepcopy(SimplePDHG.setting)
        )
    end
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.x_primal) &&
        model.termination_status == MOI.OPTIMIZE_NOT_CALLED
end

function MOI.empty!(model::Optimizer)
    empty!(model.x_primal)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    return
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{MOI.Zeros},
)
    return true
end

MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false

function MOI.supports_add_constrained_variables(
    ::Optimizer,
    ::Type{MOI.Nonnegatives},
)
    return true
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    @assert all(iszero, cache.variables.lower)
    @assert all(==(Inf), cache.variables.upper)
    A = convert(
        SparseArrays.SparseMatrixCSC{Float64,Int},
        cache.constraints.coefficients,
    )
    b = cache.constraints.constants
    b = -b # because @odow set Ax+b ∈ {0}
    c = zeros(size(A, 2))
    offset = cache.objective.scalar_affine.constant
    for term in cache.objective.scalar_affine.terms
        c[term.variable.value] += term.coefficient
    end
    if cache.objective.sense == MOI.MAX_SENSE
        c *= -1
    end
    st, sp, sd, x, y, z = solve_pdhg(A, b, c, dest.settings)
    for i in MOI.get(src, MOI.ListOfVariableIndices())
        dest.x_primal[i] = x[index_map[i].value]
    end
    for i in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{Float64},MOI.Zeros}())
        dest.y_dual[i] = y[index_map[i].value]
    end
    for i in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.Nonnegatives}())
        dest.z_dual[i] = z[index_map[i].value]
    end
    dest.termination_status = st
    dest.primal_status = sp
    dest.dual_status = sd
    return index_map, false
end

MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute) = getfield(model.settings, Symbol(attr.name))
MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value) = setfield!(model.settings, Symbol(attr.name), value)
MOI.get(model::Optimizer, ::MOI.Silent) = !model.settings.verbose

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.settings.verbose = !value
end


MOI.get(model::Optimizer, ::MOI.VariablePrimal, x::MOI.VariableIndex) = model.x_primal[x]
MOI.get(model::Optimizer, ::MOI.ConstraintDual, x::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.Zeros}) = model.y_dual[x]
MOI.get(model::Optimizer, ::MOI.ConstraintDual, x::MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Nonnegatives}) = model.z_dual[x]

#

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.termination_status == MOI.OPTIMAL ? 1 : 0
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return "$(model.termination_status)"
end

#

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status
MOI.get(model::Optimizer, ::MOI.PrimalStatus) = model.primal_status
MOI.get(model::Optimizer, ::MOI.DualStatus) = model.dual_status

MOI.get(::Optimizer, ::MOI.SolverName) = "PDHG"

# export all the objects (functions, struct and so on) defined in this module in comma seperated form 

export LP_Data, PDHG_settings, PDHG_state, PDHG_iteration!, project_nonnegative!, project_nonnegative, tolerance_LP, PDHG_solver, solve_pdhg


end # end module