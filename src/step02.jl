module Step02
using JuReto
export Square

abstract type MyFunction end

struct Square <: MyFunction
    forward::Function
    function Square()
        forward = x->x^2
        new(forward)
    end
end

function (f::MyFunction)(input::Variable)
    x = input.data
    y = f.forward(x)
    output = Variable(y)
end

end