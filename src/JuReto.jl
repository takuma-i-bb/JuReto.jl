module JuReto

# Write your package code here.
export Variable, Square, Exp, numerical_diff
include("step01.jl")
using .Step01

include("step03.jl")
using .Step03

include("step04.jl")
using .Step04
end
