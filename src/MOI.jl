## MOI interface for SimplePDHG


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
    solve_time::Float64

    function Optimizer()
        return new(
            Dict{MOI.VariableIndex,Float64}(),
            Dict{MOI.ConstraintIndex,Float64}(),
            Dict{MOI.ConstraintIndex,Float64}(),
            MOI.OPTIMIZE_NOT_CALLED,
            MOI.UNKNOWN_RESULT_STATUS,
            MOI.UNKNOWN_RESULT_STATUS,
            deepcopy(SimplePDHG.setting),
            0.0,
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
    state, _, solve_time = solve_pdhg(A, b, c, dest.settings)
    for i in MOI.get(src, MOI.ListOfVariableIndices())
        dest.x_primal[i] = state.x[index_map[i].value]
    end
    for i in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{Float64},MOI.Zeros}())
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
MOI.get(model::Optimizer, ::MOI.Silent) = !model.settings.verbose
MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

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
