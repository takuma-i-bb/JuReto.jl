using JuReto
include("step24.jl")

x = Variable(1.0)
y = Variable(1.0)
z = goldstein(x, y)
backward!(z)

x.name = "x"
y.name = "y"
z.name = "z"
plot_dot_graph(z, false, "goldstein.png")