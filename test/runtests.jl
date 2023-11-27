using JuReto
using Test

@testset "Config" begin
    x = Variable(1.0)
    y = no_grad() do 
        y = x|>square|>square
    end
    @test y.data == 1.0
    @test y.creator === nothing
    y = x|>square|>square
    @test isa(y.creator, JuReto.MyFunction)
end

@testset "Variable" begin
    x = Variable(0.5)
    y = square(x)
    backward!(y)
    @test x.data == 0.5
    @test x.grad !== nothing
    @test y.grad === nothing
    init_grad!(x)
    @test x.grad === nothing
    backward!(y, retain_grad=true)
    @test x.grad !== nothing
    @test y.grad == 1.0
end

@testset "square" begin
    # Write your tests here.
    x = Variable(2.0)
    @test square(x).data == 4.0
    
    x = Variable(3.0)
    backward!(square(x))
    @test x.grad == 6.0
    @test isapprox(x.grad, numerical_diff(Square(), x); atol=1e-3) 
    
    x = Variable(rand(Float32, 1)[1])
    backward!(square(x))
    @test isapprox(x.grad, numerical_diff(Square(), x); atol=1e-3)
end

@testset "exp" begin
    x = Variable(2.0)
    y = exp(x)
    @test y.data == exp(2.0)
    backward!(y)
    @test isapprox(x.grad, numerical_diff(Exp(), x); atol=1e-3) 
end

@testset "-" begin
    x = Variable(2.0)
    y = -x
    @test y.data == -x.data
    backward!(y)
    @test isapprox(x.grad, numerical_diff(-, x); atol=1e-3) 
end

@testset "+" begin
    x1, x2 = Variable.(rand(2))
    y = x1 + x2
    @test y.data == x1.data + x2.data
    backward!(y)
    numerical_grad = numerical_diff(+, [x1, x2])
    @test isapprox.(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox.(x2.grad, numerical_grad[2]; atol=1e-3)
    init_grad!.((x1, x2))
    y = x1 + x2 + x1
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->2x1+x2, [x1,x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
    
    # 定数（not Variable）との和
    x1, x2 = Variable.(rand(2))
    x3 = rand()
    y = x1 + x2 + x3
    @test y.data == x1.data + x2.data + x3
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->x1+x2+x3, [x1, x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3) 
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3) 
end

@testset "-" begin
    x1, x2 = Variable.(rand(2))
    y = x1 - x2
    @test y.data == x1.data - x2.data
    backward!(y)
    numerical_grad = numerical_diff(-, [x1, x2])
    @test isapprox.(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox.(x2.grad, numerical_grad[2]; atol=1e-3)
    init_grad!.((x1, x2))
    y = -x1 - x2 - x1
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->-2x1-x2, [x1,x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
    
    # 定数（not Variable）との和
    x1, x2 = Variable.(rand(2))
    x3 = rand()
    y = x1 - x2 - x3
    @test y.data == x1.data - x2.data - x3
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->x1-x2-x3, [x1, x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3) 
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
end

@testset "*" begin
    x1, x2 = Variable.(rand(2))
    y = x1 * x2
    @test y.data == x1.data * x2.data
    backward!(y)
    numerical_grad = numerical_diff(*, [x1, x2])
    @test isapprox.(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox.(x2.grad, numerical_grad[2]; atol=1e-3)
    init_grad!.((x1, x2))
    y = x1 * x2 * x1
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->x1^2*x2, [x1,x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
    
    # 定数（not Variable）との積
    x1, x2 = Variable.(rand(2))
    x3 = rand()
    y = x1 * x2 * x3
    @test y.data == x1.data * x2.data * x3
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->x1*x2*x3, [x1, x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
end

@testset "/" begin
    x1, x2 = Variable.(rand(2))
    y = x1 / x2
    @test y.data == x1.data / x2.data
    backward!(y)
    numerical_grad = numerical_diff(/, [x1, x2])
    @test isapprox.(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox.(x2.grad, numerical_grad[2]; atol=1e-3)
    init_grad!.((x1, x2))
    y = 2.0 / x2 / x1
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->2.0/x2/x1, [x1,x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
    
    # 定数（not Variable）との積
    x1, x2 = Variable.(rand(2))
    x3 = rand()
    y = x1 / x2 / x3
    @test y.data == x1.data / x2.data / x3
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->x1/x2/x3, [x1, x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3) 
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
end

@testset "^" begin
    x1, x2 = Variable.(rand(2))
    y = x1 ^ x2
    @test y.data == x1.data ^ x2.data
    backward!(y)
    numerical_grad = numerical_diff(^, [x1, x2])
    @test isapprox.(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox.(x2.grad, numerical_grad[2]; atol=1e-3)
    init_grad!.((x1, x2))
    y = x1 ^ x2 ^ x1
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->x1^x2^x1, [x1,x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
    
    # 定数（not Variable）との積
    x1, x2 = Variable.(rand(2))
    x3 = rand()
    y = x1 ^ x2 ^ x3
    @test y.data == x1.data ^ x2.data ^ x3
    backward!(y)
    numerical_grad = numerical_diff((x1, x2)->x1^x2^x3, [x1, x2])
    @test isapprox(x1.grad, numerical_grad[1]; atol=1e-3)
    @test isapprox(x2.grad, numerical_grad[2]; atol=1e-3)
end

