module JuReto

# Write your package code here.
export Variable, backward, Square, square, Exp, exp, Add, numerical_diff
include("step11.jl")
using .Step11

include("step04.jl")
using .Step04
end
