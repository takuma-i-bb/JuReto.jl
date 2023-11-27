module Step22
abstract type MyFunction end
using JuReto
using DataStructures

export Variable, backward!, init_grad!
export MyFunction, backward, -, Square, square, Exp, exp, Add, +
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
    
    name::Union{String, Nothing}
end

function Variable(data; name = nothing)
    Variable(data, nothing, nothing, 1, name)
end
import Base:convert, promote_rule
convert(::Type{Variable}, x) = Variable(x)
convert(::Type{Variable}, x::Variable) = x
convert(::Type{Variable{T}}, x::Variable{T}) where T<:Real = x
convert(::Type{Variable{T}}, x) where {T<:Real} = Variable(x)
function promote_rule(::Type{Variable{T}}, ::Type{S}) where {T<:Real, S<:Real}
    @show T, S
    Variable{promote_type(T, S)}
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

# function as_variable(data)
#     if isa(data, Variable)
#         return data
#     else
#         return Variable(data)
#     end
# end

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

import Base:show, size, ndims, eltype
size(val::Variable, d::Integer) = size(val.data, d)
size(val::Variable) = size(val.data)
ndims(val::Variable) = ndims(val.data)
eltype(val::Variable) = eltype(val.data)
show(io::IO, val::Variable) = print(io, "Variable{$(eltype(val))}($(val.data))")


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
# function (f::MyFunction)(inputs...)
#     f(as_variable.(inputs))
# end

struct Neg <: MyFunction
    data::FunctionData
end

function Neg()
    Neg(FunctionData())
end

import Base:-
function -(x::Variable)
    Neg()(x)
end

function forward(f::Neg, x)
    (-x,)
end

function backward(f::Neg, dy)
    -dy
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
+(x1::Union{Number, Variable}, x2::Union{Number, Variable}) = +(promote(x1, x2)...)

function forward(f::Add, x0, x1)
    y = x0 + x1
    return (y,)
end

function backward(f::Add, dy)
    return dy, dy
end

struct Sub <: MyFunction
    data::FunctionData
end

function Sub()
    Sub(FunctionData())
end

import Base:-
function -(x1::Variable, x2::Variable)
    Sub()(x1, x2)
end
-(x1::Union{Number, Variable}, x2::Union{Number, Variable}) = -(promote(x1, x2)...)

function forward(f::Sub, x1, x2)
    y = x1 - x2
    return (y,)
end

function backward(f::Sub, dy)
    return dy, -dy
end

struct Mul <: MyFunction
    data::FunctionData
end

function Mul()
    Mul(FunctionData())
end

import Base:*
function *(x1::Variable, x2::Variable)
    Mul()(x1, x2)
end
*(x1::Union{Number, Variable}, x2::Union{Number, Variable}) = *(promote(x1, x2)...)

function forward(f::Mul, x0, x1)
    y = x0 * x1
    return (y,)
end

function backward(f::Mul, dy)
    x0, x1 = f.data.inputs[1].data, f.data.inputs[2].data
    return dy*x1, dy*x0
end

struct Div <: MyFunction
    data::FunctionData
end

function Div()
    Div(FunctionData())
end

import Base:/
function /(x1::Variable, x2::Variable)
    Div()(x1, x2)
end
/(x1::Union{Number, Variable}, x2::Union{Number, Variable}) = /(promote(x1, x2)...)

function forward(f::Div, x1, x2)
    y = x1 / x2
    return (y,)
end

function backward(f::Div, dy)
    x1, x2 = f.data.inputs[1].data, f.data.inputs[2].data
    return dy / x2, -dy * x1 / x2^2
end

struct Pow <: MyFunction
    data::FunctionData
end

function Pow()
    Pow(FunctionData())
end

import Base:^
function ^(x1::Variable, x2::Variable)
    Pow()(x1, x2)
end
^(x1::Union{Number, Variable}, x2::Union{Number, Variable}) = ^(promote(x1, x2)...)

function forward(f::Pow, x1, x2)
    y = x1^x2
    return (y,)
end

function backward(f::Pow, dy)
    x1, x2 = f.data.inputs[1].data, f.data.inputs[2].data
    return x2*x1^(x2-1)*dy, log(x1)*x1^x2*dy
end

end