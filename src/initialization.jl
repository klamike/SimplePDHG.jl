
# Write a function that will take a large matrix which is of type SparseArrays.SparseMatrixCSC{T,Int} and compute its maximum singular value using any Julia package that is suitable to solve this problem

function max_singular_value_PDHG(A::SparseArrays.SparseMatrixCSC{T,Int}) where T<:Real
    σmaxA = LinearAlgebra.norm(A,2)
    return σmaxA
end

# Write a julia struct that will take c which is a vector of length n, A which is a matrix of size m x n, and b which is a vector of length m

struct LP_Data{T<:Real, I <: Integer} 
    c::AbstractVector{T} # cost vector of length n
    A::SparseArrays.SparseMatrixCSC{T,Int} # date matrix of size m x n
    b::AbstractVector{T} # resource vector of length m
    m::I # number of rows of A
    n::I # number of columns of A
end

# construct PDHG state that is going to contain the necessary information to completely describe one iteration of PDHG algorithm

mutable struct PDHG_state{T <: AbstractVecOrMat{<: Real}, I <: Integer} # contains information regarding one iterattion sequence
    x::T # iterate x_n
    y::T # iterate y_n
    z::T # iterate z_n
    η::Real # step size
    τ::Real # step size
    k::I # iteration counter  
    st::MOI.TerminationStatusCode # termination status
    sp::MOI.ResultStatusCode # primal status
    sd::MOI.ResultStatusCode # dual status
end

# We need a method to construct the initial value of the type PDHG_state

function PDHG_state(problem::LP_Data{T,I}) where {T<:Real, I<:Integer}
    n = problem.n
    m = problem.m
    σmaxA = max_singular_value_PDHG(problem.A)
    η_preli = (1/(σmaxA)) - 1e-6
    τ_preli = (1/(σmaxA)) - 1e-6
    @assert η_preli > 0 && τ_preli > 0 "Got negative initial step sizes"
    x_0 = zeros(T, n)
    y_0 = zeros(T, m)
    return PDHG_state(x_0, y_0, problem.c, η_preli, τ_preli, 1, MOI.OPTIMIZE_NOT_CALLED, MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS)
end
