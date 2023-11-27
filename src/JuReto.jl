module JuReto

# Write your package code here.
export Variable, backward!, init_grad!
export backward, Square, square, Exp, exp, Add, numerical_diff, +, *
export no_grad

include("step22.jl")
using .Step22

include("step04.jl")
using .Step04
end
