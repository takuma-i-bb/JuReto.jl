module Step04
using JuReto
export numerical_diff

function numerical_diff(f, x::Variable, eps=1e-4)
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    (y1.data - y0.data) / (2eps)
end
end