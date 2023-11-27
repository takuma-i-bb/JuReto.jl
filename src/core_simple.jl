module CoreSimple
abstract type MyFunction end
forward(::MyFunction, ::Any ...) = nothing
backward(::MyFunction, ::Any ...) = nothing

using JuReto
using DataStructures

export Variable, backward!, init_grad!
export MyFunction, FunctionData, backward, -, Square, square, Exp, exp, Add, +
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

function backward!(val::Variable; retain_grad=false)
    if val.grad === nothing
        initial_grad = collect(val.data) |> similar |> (x->fill!(x, one(eltype(x)))) # 同一形状・Float32の1埋め配列
        # initial_grad = Array{Float32}(undef, size(val.data)) # 勾配はFloat32に
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

end