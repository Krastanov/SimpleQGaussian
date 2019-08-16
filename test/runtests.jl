using SimpleQGaussian, Test, Random, Documenter, LinearAlgebra

function F_weedbrook(l,r) # using arXiv:1110.3234
    d = quadrature(l)-quadrature(r)
    V₁ = covariance(l)
    V₂ = covariance(r)
    δ = 0 # for one mode (det(V₁)-1) * (det(V₂)-1) = 0
    V = V₁+V₂
    Δ = det(V)
    2/(√(Δ+δ)-√δ) * exp(-(d'*inv(V)*d)/2)
end
function F_nha(l,r) # using arXiv:quant-ph/0410111
    d = quadrature(l)-quadrature(r)
    d /= 2
    V₁ = covariance(l)
    V₂ = covariance(r)
    δ = 0 # for one mode (det(V₁)-1) * (det(V₂)-1) = 0
    V = V₁+V₂
    Δ = det(V)
    √(2/(√(Δ+δ)-√δ)) * exp(-(d'*inv(V)*d))
end
function F_wigner_byhand(l,r) # by using ⟨α|β⟩ = 2πħ∫WₐWᵦ
    V₁ = covariance(l)
    V₂ = covariance(r)
    Λ₁ = inv(V₁)
    Λ₂ = inv(V₂)
    q₁ = quadrature(l)
    q₂ = quadrature(r)
    η₁ = Λ₁*q₁
    η₂ = Λ₂*q₂
    ζ₁ = - (2*log(2pi) - log(det(Λ₁)) + η₁'*V₁*η₁) / 2
    ζ₂ = - (2*log(2pi) - log(det(Λ₂)) + η₂'*V₂*η₂) / 2
    Λs = Λ₁+Λ₂
    ηs = η₁+η₂
    ζs = - (2*log(2pi) - log(det(Λs)) + ηs'*inv(Λs)*ηs) / 2
    (exp(ζ₁+ζ₂-ζs)*4pi)
end
function F_wigner_byhand2(l,r) # by using ⟨α|β⟩ = 2πħ∫WₐWᵦ
    V₁ = covariance(l)
    V₂ = covariance(r)
    Λ₁ = inv(V₁)
    Λ₂ = inv(V₂)
    q₁ = quadrature(l)
    q₂ = quadrature(r)
    η = (q₁'*Λ₁+q₂'*Λ₂)'
    2 * exp(-(q₁'*Λ₁*q₁+q₂'*Λ₂*q₂)/2) / √det(Λ₁+Λ₂) * exp(η'*inv(Λ₁+Λ₂)*η/2)
end


function tests()

Random.seed!(42)

@testset "Doctests" begin
    #DocMeta.setdocmeta!(SimpleQGaussian, :DocTestSetup, :(using SimpleQGaussian); recursive=true)
    #doctest(SimpleQGaussian)
end

@testset "Test inner product equivalencies" begin
    @testset "Coherent states" begin
        c1 = coh(0)
        c2 = coh(3,4)
        c3 = coh(-1.23,4.56)
        @test inner(c1,c1) ≈ 1
        @test inner(c2,c2) ≈ 1
        @test inner(dense(c1),c1) ≈ 1
        @test inner(dense(c1),c2) ≈ inner(c1,c2)
        @test inner(dense(c3),c2) ≈ inner(c3,c2)
        @test absinner(c1,c2) ≈ abs(inner(c1,c2))
        @test absinner(c3,c2) ≈ abs(inner(c3,c2))
    end
    @testset "Gauss states" begin
        c1 = coh(0)
        c2 = coh(.3,.4)
        c3 = coh(-.123,.456)
        g2 = gauss(3,4,0.3π,0.5)
        g3 = gauss(-2,-1,0.2π,0.345)
        @test inner(g2,g2) ≈ 1
        @test inner(dense(g2),g2) ≈ 1
        @test absinner(dense(c1),g2) ≈ absinner(c1,g2)

        @test absinner(dense(c2),c3)^2 ≈ abs2(inner(dense(c2),c3)) ≈ F_weedbrook(c2,c3) ≈ F_wigner_byhand(c2,c3) ≈ F_wigner_byhand2(c2,c3) ≈ F_nha(c2,c3)^2

        @test absinner(dense(c2),g2)^2 ≈ abs2(inner(dense(c2),g2)) ≈ F_weedbrook(c2,g2) ≈ F_wigner_byhand(c2,g2) ≈ F_wigner_byhand2(c2,g2) ≈ F_nha(c2,g2)^2

        @test absinner(dense(g2),g3)^2 ≈ abs2(inner(dense(g2),g3)) ≈ F_weedbrook(g2,g3) ≈ F_wigner_byhand(g2,g3) ≈ F_wigner_byhand2(g2,g3) ≈ F_nha(g2,g3)^2
    end
    @testset "Inner product of a coherent state and vacuum squeezed state" begin
        a = 0.1+0.2im
        r = 0.1234
        t = 0.4321

        @test displace_squeeze_overlap(a,r*exp(-2im*t)) ≈ displace_squeeze_overlap(a,t,r) ≈ inner(dense(coh(a)),gauss(0,t,r))
        @test displace_squeeze_overlap(0,r*exp(-2im*t)) ≈ displace_squeeze_overlap(0,t,r) ≈ inner(dense(coh(0)),gauss(0,t,r)) ≈ fockcoeff_squeeze(r*exp(-2im*t),0) ≈ dense(gauss(0,t,r)).vec[1] ≈ absinner(coh(0),gauss(0,t,r))
    end
    @testset "Inner product of two gaussian states" begin
        a1 = 0.1+0.2im
        r1 = 0.1234
        t1 = 0.4321
        z1 = r1*exp(-2im*t1)
        a2 = 0.3-0.1im
        r2 = 0.0543
        t2 = -0.0234
        z2 = r2*exp(-2im*t2)

        @test abs(inner(dense(gauss(a2,t2,r2)),gauss(a1,t1,r1))) ≈ absinner(gauss(a2,t2,r2),gauss(a1,t1,r1))
        @test displace_squeeze_overlap(a2,0,0,z1) ≈ displace_squeeze_overlap(a2,z1) ≈ displace_squeeze_overlap(a2,t1,r1)
        @test displace_squeeze_overlap(0,0,a1,z1) ≈ displace_squeeze_overlap(-a1,z1) ≈ inner(dense(coh(0)),gauss(a1,t1,r1)) ≈ inner(dense(coh(-a1)),gauss(0,t1,r1))
        @test absinner(coh(0),gauss(a1,t1,r1)) ≈ absinner(coh(-a1),gauss(0,t1,r1)) ≈ absinner(gauss(-a1,0,0),gauss(0,t1,r1))
        @test inner(dense(gauss(a2,0,0)),gauss(a1,t1,r1)) ≈ displace_squeeze_overlap(a2,0,a1,z1)
        @test inner(dense(gauss(0,t2,r2)),gauss(a1,t1,r1)) ≈ displace_squeeze_overlap(0,z2,a1,z1)
        @test inner(dense(gauss(a2,t2,r2)),gauss(a1,t1,r1)) ≈ displace_squeeze_overlap(a2,z2,a1,z1) ≈ displace_squeeze_overlap(a2,t2,r2,a1,t1,r1)
        @test inner(dense(gauss(a2,t2,r2)),gauss(a1,t1,r1)) ≈ inner(gauss(a2,t2,r2),gauss(a1,t1,r1))
    end
end

@testset "Closed form Fock coefficients" begin
    α = 0.123 + 0.321im
    @test [fockcoeff_displace(α,i) for i in 0:5] ≈ dense(coh(α)).vec[1:6]
    @test [fockcoeff_squeeze(1+0im,i) for i in 0:5] ≈ dense(gauss(0,0,1)).vec[1:6]
end

@testset "Commutation identities" begin
    θ = 0.4321
    r = 0.1234
    n = 40
    nsub = n÷2
    @test rotate_f(-θ,n)*squeeze_f(r,n)*rotate_f(θ,n) ≈ squeeze_f(r*exp(im*θ*2),n)
    @test squeeze_f(r,n)*squeeze_f(-r,n) ≈ LinearAlgebra.I
    @test squeeze_f(r*exp(im*θ*2),n)*squeeze_f(-r*exp(im*θ*2),n) ≈ LinearAlgebra.I
    @test rotate_f(θ,n)*squeeze_f(r,n)[:,1] ≈ squeeze_f(r*exp(-im*θ*2),n)[:,1]

    α = 0.123 + 0.321im
    β = 0.456 + 0.654im
    @test (displace_f(α,n)*displace_f(β,n))[1:nsub,1:nsub] ≈ (displace_f(β+α,n)*SimpleQGaussian.displace_prodphase(α,β))[1:nsub,1:nsub]

    z = r*exp(im*θ)
    @test (squeeze_f(z,n)*displace_f(α,n))[1:nsub,1:nsub] ≈ (displace_f(SimpleQGaussian.squeeze_displace_commparam(α,z),n)*squeeze_f(z,n))[1:nsub,1:nsub]

    z1 = 0.0123 + 0.0321im
    z2 = 0.0456 + 0.0654im
    z3,phase = SimpleQGaussian.squeeze_prod(z1,z2)
    @test (squeeze_f(z1,n)*squeeze_f(z2,n))[1:nsub,1] ≈ phase*squeeze_f(z3,n)[1:nsub,1]
end

end

tests()
