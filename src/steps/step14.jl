module Step14
abstract type MyFunction end
using JuReto

export Variable, backward!, init_grad!
export backward, Square, square, Exp, exp, Add, +

mutable struct Variable{T<:Union{Real, Array{ElT} where {ElT<:Real}}}
    data::T
    grad::Union{T, Nothing}
    creator::Union{MyFunction, Nothing}
end

function Variable(data)
    Variable(data, nothing, nothing)
end

function _set_creator!(val::Variable, func::MyFunction)
    val.creator = func
end

function init_grad!(val::Variable)
    val.grad = nothing
    val
end

function backward!(val::Variable)
    if val.grad === nothing
        initial_grad = collect(val.data) |> similar |> (x->fill!(x, one(eltype(x)))) # 同一形状・データ型の1埋め配列
        if ndims(initial_grad) == 0
            val.grad = initial_grad[1] # 0次元配列をスカラーに変更
        else
            val.grad = initial_grad
        end
    end

    funcs = MyFunction[val.creator]
    while !isempty(funcs)
        f = pop!(funcs)
        gys = [output.grad for output in f.data.outputs]
        gxs = backward(f, gys...)
        isa(gxs, Tuple) || (gxs = (gxs, ))

        for (x, gx) in zip(f.data.inputs, gxs)
            (x.grad === nothing) ? (x.grad = gx) : (x.grad += gx)
            
            if x.creator !== nothing
                push!(funcs, x.creator)
            end
        end
    end
end

mutable struct FunctionData
    inputs::Union{Vector{Variable}, Nothing}
    outputs::Union{Vector{Variable}, Nothing}
end

function FunctionData()
    FunctionData(nothing, nothing)
end

function (f::MyFunction)(inputs::Variable ...)
    xs = [input.data for input in inputs]
    ys = forward(f, xs...)
    outputs = Variable.(ys)
    
    for output in outputs
        _set_creator!(output, f)
    end

    f.data.inputs = [i for i in inputs]
    f.data.outputs = [i for i in outputs]
    return (length(outputs)==1) ? outputs[1] : outputs
end

struct Square <: MyFunction
    data::FunctionData
end

function Square()
    Square(FunctionData())
end

# 本当はJuliaの演算子に多重ディスパッチでメソッド追加したいが、2乗限定の関数はない
# pow()関数を実装時に行う
function square(x::Variable)
    Square()(x)
end

function forward(f::Square, x)
    (x^2,)
end

function backward(f::Square, dy)
    x = f.data.inputs[1].data
    2x*dy
end

struct Exp <: MyFunction
    data::FunctionData
end

function Exp()
    Exp(FunctionData())
end

import Base: exp
function exp(x::Variable)
    Exp()(x)
end

function forward(f::Exp, x)
    (exp(x),)
end

function backward(f::Exp, dy)
    x = f.data.inputs[1].data
    exp(x)*dy
end

struct Add <: MyFunction
    data::FunctionData
end

function Add()
    Add(FunctionData(nothing, nothing))
end

import Base:+
function +(x1::Variable, x2::Variable)
    Add()(x1, x2)
end

function forward(f::Add, x0, x1)
    y = x0 + x1
    return (y,)
end

function backward(f::Add, dy)
    return dy, dy
end

end