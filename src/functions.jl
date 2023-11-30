module Functions
using JuReto
import JuReto.CoreSimple:MyFunction,FunctionData,forward, backward
export forward, backward

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

struct Pow{T<:Real} <: MyFunction
    data::FunctionData
    c::T
end

function Pow(c::Real)
    Pow(FunctionData(), c)
end

struct Pow2 <: MyFunction
    data::FunctionData
end

function Pow2()
    Pow2(FunctionData())
end

import Base:^
^(x1::Union{Number, Variable}, x2::Real) = Pow(x2)(x1)
function ^(x1::Variable, x2::Variable)
    Pow2()(x1, x2)
end
^(x1::Union{Number, Variable}, x2::Union{Variable}) = ^(promote(x1, x2)...)

function forward(f::Pow, x1)
    y = x1^f.c
    return (y,)
end

function backward(f::Pow, dy)
    x1 = f.data.inputs[1].data
    return f.c*x1^(f.c-1)*dy
end

function forward(f::Pow2, x1, x2)
    y = x1^x2
    return (y,)
end

function backward(f::Pow2, dy)
    x1, x2 = f.data.inputs[1].data, f.data.inputs[2].data
    return x2*x1^(x2-1)*dy, log(x1)*x1^x2*dy
end
end