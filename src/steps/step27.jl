using JuReto
import JuReto.CoreSimple:MyFunction,FunctionData,forward, backward

struct Sin <:MyFunction
    data::FunctionData
end

function Sin()
    Sin(FunctionData())
end

import Base:sin
function sin(x::Variable)
    Sin()(x)
end

function forward(f::Sin, x)
    (sin(x),)
end

function backward(f::Sin, dy)
    x = f.data.inputs[1].data
    dy * cos(x)
end

function my_sin(x, threshold=0.0001)
    y = 0
    for i in 0:1e6
        c = (-1.0) ^ i / factorial(2Int(i)+1)
        t = c*x^(2.0i+1.0)
        y += t
        if abs(t.data) < threshold
            break
        end
    end
    y
end