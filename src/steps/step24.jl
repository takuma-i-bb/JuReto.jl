using JuReto

function sphere(x, y)
    z = x^2.0 + y^2.0
end

x = Variable(1.0)
y = Variable(1.0)
z = sphere(x, y)
backward!(z)
@show x.grad, y.grad

function matyas(x, y)
    z = 0.26 * sphere(x, y) - 0.48 * x * y
end
x = Variable(1.0)
y = Variable(1.0)
z = matyas(x, y)
backward!(z)
@show x.grad, y.grad

function goldstein(x, y)
    z = (1.0 + (x + y + 1.0)^2.0 * (19.0 - 14.0x + 3.0x^2.0 - 14.0y + 6.0x*y + 3.0y^2.0)) * (30.0 + (2.0x - 3.0y)^2.0 * (18.0 - 32.0*x + 12.0x^2.0 + 48.0y - 36.0x*y + 27.0y^2.0))
end

x = Variable(1.0)
y = Variable(1.0)
z = goldstein(x, y)
backward!(z)
@show x.grad, y.grad
