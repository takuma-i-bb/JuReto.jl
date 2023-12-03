module Utils
using JuReto
import JuReto.CoreSimple:MyFunction
export numerical_diff, get_dot_graph, plot_dot_graph

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

function _dot_var(v::Variable, verbose=false)
    name = v.name === nothing ? "" : v.name

    if verbose && v.data !== nothing
        if v.name !== nothing
            name *= ": "
        end
        name *= string(size(v)) * " " * string(eltype(v))
    end
    dot_var = "$(Int128(objectid(v))) [label=\"$(name)\", color=orange, style=filled]\n"
end

function _dot_func(f::MyFunction)
    f_name = split(string(typeof(f)), ".")[end]
    txt = "$(Int128(objectid(f))) [label=\"$(f_name)\", color=lightblue, style=filled, shape=box]\n"

    for x in f.data.inputs
        txt *= "$(objectid(x)) -> $(objectid(f))\n"
    end
    for y in f.data.outputs
        txt *= "$(objectid(f)) -> $(objectid(y))\n"
    end
    return txt
end

function get_dot_graph(output, verbose=true)
    txt = ""
    funcs = Vector{MyFunction}()
    seen_set = Set{MyFunction}()

    function push_func(f)
        if !(f in seen_set)
            push!(funcs, f)
            push!(seen_set, f)
        end
    end

    push_func(output.creator)
    txt *= _dot_var(output, verbose)

    while !isempty(funcs)
        func = pop!(funcs)
        txt *= _dot_func(func)
        for x in func.data.inputs
            txt *= _dot_var(x, verbose)

            if x.creator !== nothing
                push_func(x.creator)
            end
        end
    end

    return "digraph g{\n" * txt * "}"
end

function plot_dot_graph(output, verbose=true, to_file="graph.png")
    dot_graph = get_dot_graph(output, verbose)

    # dotデータをファイルに保存
    tmp_dir = pwd()*"/.JuReto"
    ".JuReto" in readdir(pwd()) || mkdir(tmp_dir)
    graph_path = tmp_dir * "/tmp_graph.dot"

    open(graph_path, "w") do io
        write(io, dot_graph)
    end

    extension = splitext(to_file)[end][2:end]
    run(`dot $(graph_path) -T $(extension) -o $(to_file)`)
end
end