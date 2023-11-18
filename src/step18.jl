module Step18
abstract type MyFunction end
using JuReto
using DataStructures

export Variable, backward!, init_grad!
export MyFunction, backward, Square, square, Exp, exp, Add, +
export no_grad

mutable struct Config 
    enable_backprop::Bool
end
config::Config = Config(true)

function using_config(f, fieldname::Symbol, value)
    old_value = getfield(config, fieldname)
    setfield!(config, fieldname, value)
    try
        f()
    finally
        setfield!(config, fieldname, old_value)
    end
end

function no_grad(f)
    using_config(f, :enable_backprop, false)
end

mutable struct Variable{T<:Union{Real, Array{ElT} where {ElT<:Real}}}
    data::T
    grad::Union{T, Nothing}
    creator::Union{MyFunction, Nothing}
    generation::Int64
end

function Variable(data)
    Variable(data, nothing, nothing, 1)
end

function _set_creator!(val::Variable, func::MyFunction)
    val.creator = func
    val.generation = func.data.generation+1
    return val
end

function init_grad!(val::Variable)
    val.grad = nothing
    return val
end

function backward!(val::Variable; retain_grad=false)
    if val.grad === nothing
        initial_grad = collect(val.data) |> similar |> (x->fill!(x, one(eltype(x)))) # 同一形状・データ型の1埋め配列
        if ndims(initial_grad) == 0
            val.grad = initial_grad[1] # 0次元配列をスカラーに変更
        else
            val.grad = initial_grad
        end
    end

    funcs = PriorityQueue{MyFunction, Int64}()
    seen_set = Set{MyFunction}()

    function push_func(f)
        if !(f in seen_set)
            enqueue!(funcs, f, -f.data.generation)
            push!(seen_set, f)
        end
    end

    push_func(val.creator)
    while !isempty(funcs)
        f = dequeue!(funcs)
        gys = [output.grad for output in f.data.outputs]
        gxs = backward(f, gys...)
        isa(gxs, Tuple) || (gxs = (gxs, ))

        for (x, gx) in zip(f.data.inputs, gxs)
            (x.grad === nothing) ? (x.grad = gx) : (x.grad += gx)
            
            if x.creator !== nothing
                push_func(x.creator)
            end
        end

        if !retain_grad
            for y in f.data.outputs
                y.grad = nothing
            end
        end
    end
end

mutable struct FunctionData
    inputs::Union{Vector{Variable}, Nothing}
    outputs::Union{Vector{Variable}, Nothing}
    generation::Int64
end

function FunctionData()
    FunctionData(nothing, nothing, 1)
end

function (f::MyFunction)(inputs::Variable ...)
    xs = [input.data for input in inputs]
    ys = forward(f, xs...)
    outputs = Variable.(ys)

    if config.enable_backprop
        f.data.generation = maximum([i.generation for i in inputs])
        for output in outputs
            _set_creator!(output, f)
        end
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
    Add(FunctionData())
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