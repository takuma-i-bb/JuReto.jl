module Utils
using JuReto
export numerical_diff

function modify_argument(func, arg_index, args)
    return (new_arg) -> begin
        modified_args = collect(args)
        modified_args[arg_index] = new_arg
        return func(modified_args...)
    end
end

function numerical_diff(f, x::Variable, eps=1e-8)
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    (y1.data - y0.data) / (2eps)
end

function numerical_diff(f, xs::Vector{T}; eps=1e-4) where {T<:Variable}
    dxs = Vector{promote_type([typeof(x.data) for x in xs]...)}(undef, length(xs))
    for i in 1:length(xs)
        dxs[i] = numerical_diff(modify_argument(f, i, xs), xs[i])
    end
    return Tuple(dxs)
end
end