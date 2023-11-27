module Step06

export Variable

mutable struct Variable{T}
    data::T
    grad::Union{T, Nothing}
end

function Variable(data)
    Variable(data, nothing)
end

using JuReto
export Square, Exp, backward

abstract type MyFunction end

mutable struct Square <: MyFunction
    input::Union{Variable, Nothing}
end

function Square()
    Square(nothing)
end

function forward(f::Square, x)
    x^2
end

function backward(f::Square, dy)
    x = f.input.data
    2x*dy
end

mutable struct Exp <: MyFunction
    input::Union{Variable, Nothing}
end

function Exp()
    Exp(nothing)
end

function forward(f::Exp, x)
    exp(x)
end

function backward(f::Exp, dy)
    x = f.input.data
    exp(x)*dy
end

function (f::MyFunction)(input::Variable)
    x = input.data
    y = forward(f,x)
    output = Variable(y)
    f.input = input
    return output
end

end

