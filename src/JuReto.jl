module JuReto

# Write your package code here.
export Variable, backward!, init_grad!
export backward, exp, numerical_diff, get_dot_graph, plot_dot_graph
export no_grad

include("core_simple.jl")
using .CoreSimple

include("functions.jl")
using .Functions

include("utils.jl")
using .Utils
end
