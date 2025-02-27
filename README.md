# SimplePDHG.jl

> [!NOTE]
> This is a fork of the original [Shuvomoy/SimplePDHG.jl](https://github.com/Shuvomoy/SimplePDHG.jl) with the following changes:
> - Support arbitrary float arithmetic
> - Support querying dual solution
> - Set solver settings via MOI attributes
> - Support SolveTimeSec, Silent
> - Better termination/result status
>
> The original README follows.

I wrote this simple educational Julia package (less than 350 lines of code) for the MIT Course 15.084/6.7220 Nonlinear Optimization. The purpose of this package is to demonstrate to the students how simple it is to implement an optimization algorithm in  [Julia](https://julialang.org/) and connect it to the optimization modeling language [`JuMP`](https://jump.dev/) so that anyone can use your algorithm.

Big thanks to [Oscar Dowson](https://odow.github.io/) for providing [`MathOptInterface.jl `](https://jump.dev/MathOptInterface.jl/stable/) code to connect this simple solver to `JuMP`! ([discourse link](https://discourse.julialang.org/t/connecting-a-simple-first-order-solver-to-solve-standard-form-linear-program-to-jump/95694))

## What does `SimplePDHG.jl` do?

This is an educational package used to demonstrate the ease of implementing an algorithm in `Julia` and incorporating it with one of Julia's main optimization modeling language `JuMP`. The package is  designed to solve linear programming problems of the form:

```julia
minimize    c'x
subject to  A x = b
            G x ≤ h
            x ∈ ℝ^n
```

where `x` is the decision variable. Under the hood the `SimplePDHG.jl` implements the vanilla PDHG algorithm (see Section 3.3 of [this book](https://large-scale-book.mathopt.com/LSCOMO.pdf)) to solve standard form linear optimization problem of the form `min{c'x ∣ Ax=b, x ≥ 0, x ∈ ℝ^n}`.

##  Installation 

Type the following in Julia REPL to the stable version:

```
] add SimplePDHG
```

To get the latest branch, type:

```julia
] add https://github.com/Shuvomoy/SimplePDHG.jl.git
```

## Usage through `JuMP`

```julia
using JuMP, SimplePDHG
model =  Model(SimplePDHG.Optimizer)
@variable(model, x >= 0)
@variable(model, 0 <= y <= 3)
@objective(model, Min, 12x + 20y)
@constraint(model, c1, 6x + 8y >= 100)
@constraint(model, c2, 7x + 12y >= 120)
optimize!(model)
println("Objective value: ", objective_value(model))
println("x = ", value(x))
println("y = ", value(y))
```

Output should be:

```julia
Objective value: 205.000090068938
x = 14.999887019427522
y = 1.2500722917903861
```

## Vector syntax in JuMP

Thanks to `JuMP` and `MathOptInterface.jl `, we can use vectorized syntax to solve our optimization problem as well. 

```julia
# data 
A = [1 1 9 5; 3 5 0 8; 2 0 6 13]
b = [7, 3, 5]
c = [1, 3, 5, 2]
m, n = size(A)
G = [0.5012005468024234 -1.5806753104910911 1.1908183108070869 1.6527613262371468; -1.7596263752677483 -0.5235246034519885 0.4618550523688477 0.4871842582808355; -0.6305269735894394 0.023788955821653315 -0.5208935392017503 -1.667410808905106; 1.02249016425841 0.6890017766482583 1.2904648745012357 1.398062622113161; -0.9763001854265912 0.866180139889124 -0.18426778358700338 1.1436405988912726; 0.4004591856282607 -0.6315453522080423 -0.32707956849441 -1.192277331736516];
h = 2*ones(2*m)

# JuMP code
using JuMP, SimplePDHG
model =  Model(SimplePDHG.Optimizer)
@variable(model, x[1:n] >= 0)
@objective(model, Min, c'*x)
@constraint(model, A*x .== b)
@constraint(model, G*x .<= h)
optimize!(model)
println("Objective value: ", objective_value(model))
println("x = ", value.(x))
x_star = value.(x)
```

The output should be:

```julia
Objective value: 4.922528390226832

x = [0.42344643304517904, 0.34592985413549193, 0.6922584789550353, 0.0]
```



