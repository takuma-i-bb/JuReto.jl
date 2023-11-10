module JuReto

# Write your package code here.
export Variable, backward, Square, square, Exp, exp, Add, numerical_diff, +
include("step13.jl")
using .Step13

include("step04.jl")
using .Step04
end
