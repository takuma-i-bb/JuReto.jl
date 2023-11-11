using JuReto
using Test

@testset "Variable" begin
    x = Variable(0.5)
    y = square(x)
    backward!(y)
    @test x.grad !== nothing
    init_grad!(x)
    @test x.grad === nothing
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
end

