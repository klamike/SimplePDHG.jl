## MOI interface for SimplePDHG


MOI.Utilities.@product_of_sets(RHS, MOI.Zeros)

const OptimizerCache{T <: Real} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    MOI.Utilities.MatrixOfConstraints{T,
        MOI.Utilities.MutableSparseMatrixCSC{T, Int, MOI.Utilities.OneBasedIndexing},
        Vector{T}, RHS{T},
    },
}

function MOI.add_constrained_variables(
    model::OptimizerCache{T},
    set::MOI.Nonnegatives,
) where {T}
    x = MOI.add_variables(model, MOI.dimension(set))
    MOI.add_constraint.(model, x, MOI.GreaterThan(zero(T)))
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Nonnegatives}(x[1].value)
    return x, ci
end

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    x_primal::Dict{MOI.VariableIndex,T}
    y_dual::Dict{MOI.ConstraintIndex,T}
    z_dual::Dict{MOI.ConstraintIndex,T}
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    settings::PDHG_settings
    solve_time::Float64

    function Optimizer{T}() where {T <: Real}
        return new(
            Dict{MOI.VariableIndex,T}(), Dict{MOI.ConstraintIndex,T}(), Dict{MOI.ConstraintIndex,T}(),
            MOI.OPTIMIZE_NOT_CALLED, MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS, deepcopy(SimplePDHG.setting), 0.0,
        )
    end
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.x_primal) && isempty(model.y_dual) && isempty(model.z_dual) && model.termination_status == MOI.OPTIMIZE_NOT_CALLED
end

function MOI.empty!(model::Optimizer)
    empty!(model.x_primal); empty!(model.y_dual); empty!(model.z_dual); model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    return
end

MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorAffineFunction{T}}, ::Type{MOI.Zeros}) where {T <: Real} = true
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Nonnegatives}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}) where {T <: Real} = true

function MOI.optimize!(dest::Optimizer{T}, src::MOI.ModelLike) where {T<:Real}
    cache = OptimizerCache{T}()
    index_map = MOI.copy_to(cache, src)
    @assert all(iszero, cache.variables.lower)
    @assert all(==(Inf), cache.variables.upper)
    A = convert(
        SparseArrays.SparseMatrixCSC{T,Int},
        cache.constraints.coefficients,
    )
    b = cache.constraints.constants
    b = -b # because @odow set Ax+b ∈ {0}
    c = zeros(T, size(A, 2))
    offset = cache.objective.scalar_affine.constant
    for term in cache.objective.scalar_affine.terms
        c[term.variable.value] += term.coefficient
    end
    if cache.objective.sense == MOI.MAX_SENSE
        c *= -1
    end
    state, _, solve_time = solve_pdhg(A, b, c, dest.settings)
    for i in MOI.get(src, MOI.ListOfVariableIndices())
        dest.x_primal[i] = state.x[index_map[i].value]
    end
    for i in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction,MOI.Zeros}())
        dest.y_dual[i] = state.y[index_map[i].value]
    end
    for i in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.Nonnegatives}())
        dest.z_dual[i] = state.z[index_map[i].value]
    end
    dest.termination_status = state.st
    dest.primal_status = state.sp
    dest.dual_status = state.sd
    dest.solve_time = solve_time
    return index_map, false
end

MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute) = getfield(model.settings, Symbol(attr.name))
MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value) = setfield!(model.settings, Symbol(attr.name), value)

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.settings.verbose = !value
end

MOI.get(::Optimizer{T}, ::MOI.SolverName) where {T} = "PDHG{$(T)}"
MOI.get(model::Optimizer, ::MOI.Silent) = !model.settings.verbose
MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time
MOI.get(model::Optimizer, ::MOI.VariablePrimal, x::MOI.VariableIndex) = model.x_primal[x]
MOI.get(model::Optimizer, ::MOI.ConstraintDual, x::MOI.ConstraintIndex{MOI.VectorAffineFunction,MOI.Zeros}) = model.y_dual[x]
MOI.get(model::Optimizer, ::MOI.ConstraintDual, x::MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Nonnegatives}) = model.z_dual[x]
MOI.get(model::Optimizer, ::MOI.ResultCount) = model.termination_status == MOI.OPTIMAL ? 1 : 0
MOI.get(model::Optimizer, ::MOI.RawStatusString) = "$(model.termination_status)"
MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status
MOI.get(model::Optimizer, ::MOI.PrimalStatus) = model.primal_status
MOI.get(model::Optimizer, ::MOI.DualStatus) = model.dual_status
