module Step08
abstract type MyFunction end
using JuReto

export Variable, Square, Exp, backward

mutable struct Variable{T}
    data::T
    grad::Union{T, Nothing}
    creator::Union{MyFunction, Nothing}
end

function Variable(data)
    Variable(data, nothing, nothing)
end

function set_creator(val::Variable, func::MyFunction)
    val.creator = func
end

function backward(val::Variable)
    funcs = MyFunction[val.creator]
    while !isempty(funcs)
        f = pop!(funcs)
        x, y = f.data.input, f.data.output
        x.grad = backward(f, y.grad)

        if x.creator !== nothing
            push!(funcs, x.creator)
        end
    end
end

mutable struct FunctionData
    input::Union{Variable, Nothing}
    output::Union{Variable, Nothing}
end

function FunctionData()
    FunctionData(nothing, nothing)
end

struct Square <: MyFunction
    data::FunctionData
end

function Square()
    Square(FunctionData())
end

function forward(f::Square, x)
    x^2
end

function backward(f::Square, dy)
    x = f.data.input.data
    2x*dy
end

struct Exp <: MyFunction
    data::FunctionData
end

function Exp()
    Exp(FunctionData())
end

function forward(f::Exp, x)
    exp(x)
end

function backward(f::Exp, dy)
    x = f.data.input.data
    exp(x)*dy
end

function (f::MyFunction)(input::Variable)
    x = input.data
    y = forward(f,x)
    output = Variable(y)
    set_creator(output, f)
    f.data.input = input
    f.data.output = output
    return output
end

end

