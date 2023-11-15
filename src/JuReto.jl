module JuReto

# Write your package code here.
export Variable, backward!, init_grad!
export backward, Square, square, Exp, exp, Add, numerical_diff, +
include("step16.jl")
using .Step16

include("step04.jl")
using .Step04
end
