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
    @test isapprox(x.grad, numerical_diff(Square(), x)) 
    
    x = Variable(rand(Float32, 1)[1])
    backward!(square(x))
    @test isapprox(x.grad, numerical_diff(Square(), x))
end

@testset "exp" begin
    x = Variable(2.0)
    y = exp(x)
    @test y.data == exp(2.0)
    backward!(y)
    @test isapprox(x.grad, numerical_diff(Exp(), x)) 
end

@testset "+" begin
    x1, x2 = Variable.(rand(2))
    y = x1 + x2
    @test y.data == x1.data + x2.data
    backward!(y)
    @test (x1.grad, x2.grad) == (1.0, 1.0)
    init_grad!.((x1, x2))
    y = x1 + x2 + x1
    backward!(y)
    @test (x1.grad, x2.grad) == (2.0, 1.0)
    
    # 定数（not Variable）との和
    x1, x2 = Variable.(rand(2))
    x3 = rand()
    y = x1 + x2 + x3
    @test y.data == x1.data + x2.data + x3
    backward!(y)
    @test (x1.grad, x2.grad) == (1.0, 1.0) 
end

@testset "*" begin
    x1, x2 = Variable.(rand(2))
    y = x1 * x2
    @test y.data == x1.data * x2.data
    backward!(y)
    @test (x1.grad, x2.grad) == (x2.data, x1.data)
    init_grad!.((x1, x2))
    y = x1 * x2 * x1
    backward!(y)
    @test (x1.grad, x2.grad) == (2x1.data*x2.data, x1.data^2)
    
    # 定数（not Variable）との積
    x1, x2 = Variable.(rand(2))
    x3 = rand()
    y = x1 * x2 * x3
    @test y.data == x1.data * x2.data * x3
    backward!(y)
    @test (x1.grad, x2.grad) == (x2.data*x3, x1.data*x3) 
end

