module Step03
using JuReto
export Square, Exp

abstract type MyFunction end

struct Square <: MyFunction
    forward::Function
    function Square()
        forward = x->x^2
        new(forward)
    end
end

struct Exp <: MyFunction
    forward::Function
    function Exp()
        forward = x->exp(x)
        new(forward)
    end
end

function (f::MyFunction)(input::Variable)
    x = input.data
    y = f.forward(x)
    output = Variable(y)
end

end