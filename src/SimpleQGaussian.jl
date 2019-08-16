module SimpleQGaussian

using LinearAlgebra

export Coherent, coh, Gauss, gauss, PureSuperposition,
    DensePureState, dense,
    fockcoeff_squeeze, fockcoeff_displace,
    create_f, destroy_f, numop_f, displace_f, rotate_f, squeeze_f,
    quadrature, camplitude, covariance, squeezeparam, rotparam,
    inner, absinner,
    displace_squeeze_overlap

# TODO document that ħ=2 and κ=0.5 in this code
# TODO use StaticArray for small matrices
# TODO optimize d'*V*d type contractions
# TODO optimize exp(im*angle(...))

const ħ=2
const κ=0.5 # a=κ(q+ip), [q,p]=i/2κ²

rmat(θ) = [cos(θ) sin(θ);
          -sin(θ) cos(θ)]
smat(r) = [exp(-r) 0;
           0       exp(r)]

fockcoeff_displace(α::Complex,n::Integer) = exp(-abs2(α)/2) * α^n / √factorial(n)
fockcoeff_squeeze(r::Complex,n::Integer) = n%2==0 ? exp(im*angle(r)*n/2)*√(factorial(n)/cosh(abs(r))) * (-tanh(abs(r)))^(n÷2) / 2^(n÷2) / factorial(n÷2) : 0

destroy_f(n) = LinearAlgebra.Bidiagonal(zeros(ComplexF64,n),ComplexF64.(sqrt.(1:n-1)), :U) # TODO write this in a prettier fashion
create_f(n) = LinearAlgebra.Bidiagonal(zeros(ComplexF64,n),ComplexF64.(sqrt.(1:n-1)), :L) # TODO write this in a prettier fashion
numop_f(n) = LinearAlgebra.Diagonal(ComplexF64.(0:n-1)) # TODO write this in a prettier fashion
displace_f(α,n) = exp(collect(α*create_f(n)-conj(α)*destroy_f(n))) # TODO collect should not be necessary
squeeze_f(z,n) = exp(collect((conj(z)*destroy_f(n)*destroy_f(n)-z*create_f(n)*create_f(n))/2)) # TODO collect should not be necessary
rotate_f(θ,n) = exp(collect(-im*θ*numop_f(n))) # TODO collect should not be necessary

abstract type AbstractPureState end
abstract type AbstractPureGauss <: AbstractPureState end

struct DensePureState{T<:Complex} <: AbstractPureState # TODO DensePureState is a stupid name
    vec::Vector{T} # TODO why not AbstractVector
end

function DensePureState(s::AbstractPureGauss)
    n = 200
    vec = zeros(ComplexF64, n)
    vec[1] = 1
    di = displace_f(camplitude(s),n)
    ro = rotate_f(rotparam(s),n)
    sq = squeeze_f(squeezeparam(s),n)
    DensePureState(di*ro*sq*vec)
end
Base.convert(::Type{DensePureState{T}}, x::AbstractPureGauss) where T<:Complex = DensePureState(x)
dense = DensePureState

struct Coherent{T<:Complex} <: AbstractPureGauss
    alpha::T
    Coherent(a) = new{typeof(Complex(a))}(Complex(a)) # TODO there should be a pretty way to write this
end
Coherent(q,p) = Coherent(κ*(q+1im*p))
coh = Coherent

struct Gauss{T<:Real} <: AbstractPureGauss
    # TODO why not AbstractVector and AbstractMatrix
    loc::Vector{T} # TODO maybe the vector and matrix here should be static
    cov::Matrix{T} # TODO or maybe they should be Abstract, but static by default
end

function Gauss(x,p,θ,r)
    x,p,θ,r = [convert(AbstractFloat, a) for a in (x,p,θ,r)]
    Gauss([x,p],rmat(θ)*smat(2r)*rmat(θ)')
end
Gauss(α,θ,r) = Gauss(real(α)/κ,imag(α)/κ,θ,r)
Gauss(c::Coherent) = Gauss(quadrature(c),collect(covariance((c))*one(eltype(quadrature(c))))) # TODO there should be a pretty way to write this
Base.convert(::Type{Gauss}, x) = Gauss(x)
gauss = Gauss

quadrature(s::Coherent) = [real(s.alpha), imag(s.alpha)]/κ
quadrature(s::Gauss) = s.loc

camplitude(s::Coherent) = s.alpha
camplitude(s::Gauss) = κ*(s.loc[1]+1im*s.loc[2])

covariance(s::Coherent) = LinearAlgebra.I(2)
covariance(s::Gauss) = s.cov

squeezeparam(s::Coherent) = 0
squeezeparam(s::Gauss) = acosh((s.cov[1,1]+s.cov[2,2])/2)/2

rotparam(s::Coherent) = 0
rotparam_(s::Gauss) = atan(-2*s.cov[1,2]/(s.cov[1,1]-s.cov[2,2]))/2
rotparam(s::Gauss) = s.cov[1,1]==s.cov[2,2] ? zero(eltype(s.cov)) : s.cov[1,1]<s.cov[2,2] ? rotparam_(s) : pi/2+rotparam_(s)

struct PureSuperposition{T<:Complex} <: AbstractPureState
    weights::Vector{T}
    states::Vector{<:AbstractPureState}
end

PureSuperposition(s::AbstractPureState) = PureSuperposition([Complex(1)],[s])
PureSuperposition(s::PureSuperposition) = s

Base.convert(::Type{<:PureSuperposition}, x::AbstractPureState) = PureSuperposition(x)

struct PureBra{T<:AbstractPureState}
    state::T
end

function Base.show(io::IO, c::Coherent; bra=false)
    if bra
        print(io, "ᶜ⟨$(c.alpha)|")
    else
        print(io, "|$(c.alpha)⟩ᶜ")
    end
end

function Base.show(io::IO, g::Gauss; bra=false)
    if bra
        print(io, "ᵍ⟨$(camplitude(g)), $(rotparam(g)), $(squeezeparam(g))|")
    else
        print(io, "|$(camplitude(g)), $(rotparam(g)), $(squeezeparam(g))⟩ᵍ")
    end
end

function Base.show(io::IO, s::PureSuperposition; bra=false)
    for i in eachindex(s.weights)
        w = bra ? conj(s.weights[i]) : s.weights[i]
        if w==1
            print(io, " + ")
        elseif w==-1
            print(io, " - ")
        else
            if real(w)!=0 && imag(w)!=0 # is complex
                print(io," + ($(w))")
            elseif real(w)==0 # is purely imaginary
                if imag(w)>=0
                    print(io," + $(imag(w))im")
                else
                    print(io," $(imag(w))im")
                end
            else # is real
                if real(w)>=0
                    print(io," + $(real(w))")
                else
                    print(io," $(real(w))")
                end
            end
        end
        show(io,s.states[i],bra=bra)
    end
end

function Base.show(io::IO, s::PureBra)
    show(io,s.state,bra=true)
end

Base.promote_rule(::Type{PureSuperposition{T}}, ::Type{<:AbstractPureState}) where T<:Complex = PureSuperposition{T}
Base.promote_rule(::Type{DensePureState{T}}, ::Type{<:AbstractPureGauss}) where T<:Complex = DensePureState{T}

Base.:(*)(l::Number, r::AbstractPureState) = PureSuperposition([Complex(l)],[r])
Base.:(*)(l::Number, r::PureSuperposition) = PureSuperposition(l .* r.weights, r.states)
Base.:(*)(l::AbstractPureState, r::Number) = r*l
Base.:(/)(l::AbstractPureState, r::Number) = (1/r)*l
Base.:(+)(s::AbstractPureState) = s
Base.:(-)(s::AbstractPureState) = PureSuperposition([Complex(-1)],[s])
Base.:(-)(s::PureSuperposition) = PureSuperposition(-s.weights,s.states)
Base.:(+)(l::AbstractPureState, r::AbstractPureState) = +(promote(l,r)...)
Base.:(-)(l::AbstractPureState, r::AbstractPureState) = l + (-r)
Base.:(+)(l::PureSuperposition, r::PureSuperposition) = PureSuperposition(vcat(l.weights,r.weights),vcat(l.states,r.states))

Base.adjoint(s::AbstractPureState) = PureBra(s)
Base.:(*)(l::Number, r::PureBra) = PureBra(conj(l)*r.state)
Base.:(*)(l::PureBra, r::Number) = PureBra(conj(r)*l.state)
Base.:(/)(l::PureBra, r::Number) = (1/r)*l
Base.:(+)(s::PureBra) = s
Base.:(-)(s::PureBra) = PureBra(-s.state)
Base.:(+)(l::PureBra, r::PureBra) = PureBra(l.state+r.state)
Base.:(-)(s::PureBra, r::PureBra) = PureBra(l.state-r.state)

inner(l::AbstractPureState, r::AbstractPureState) = inner(promote(l,r)...)

function inner(l::DensePureState, r::DensePureState)
    l.vec' * r.vec # TODO use faster inplace operations
end

function inner(l::PureSuperposition, r::PureSuperposition)
    c = zero(promote_type(eltype(l.weights),eltype(r.weights)))
    for li in eachindex(l.weights)
        for ri in eachindex(r.weights)
            c += conj(l.weights[li])*r.weights[ri]*inner(l.states[li],r.states[ri])
        end
    end
    c
end

function inner(l::Coherent, r::Coherent)
    qₗ = camplitude(l)
    qᵣ = camplitude(r)
    exp(-(abs2(qₗ)+abs2(qᵣ))/2 + conj(qₗ)*qᵣ)
end

"""
`displace_squeeze_overlap(α,z)` is
ᶜ⟨α|0,0,z⟩ᵍ, i.e. ᶜ⟨α|0,-angle(z)/2,abs(z)⟩ᵍ
"""
function displace_squeeze_overlap(α,z)
    sqrt(exp(-abs2(α)-exp(im*angle(z))*tanh(abs(z))*conj(α)^2)/cosh(abs(z)))
end

"""
`displace_squeeze_overlap(α,θ,r)` is
ᶜ⟨α|0,θ,r⟩ᵍ, i.e. ᶜ⟨α|0,0,r⋅exp(-i2θ)⟩ᵍ
"""
function displace_squeeze_overlap(α,θ,r)
    sqrt(exp(-abs2(α)-exp(-2im*θ)*tanh(r)*conj(α)^2)/cosh(r))
end

"""
`displace_squeeze_overlap(β,q,α,p)` is
ᵍ⟨β,0,q|α,0,p⟩ᵍ, i.e. ᵍ⟨β,-angle(q)/2,abs(q)|α,-angle(p)/2,abs(p)⟩ᵍ
"""
function displace_squeeze_overlap(β,q,α,p)
    phaseDD = displace_prodphase(-β,α)
    γ = squeeze_displace_commparam(α-β,-q)
    z, phaseSS = squeeze_prod(-q,p)
    displace_squeeze_overlap(-γ,z)*phaseDD*phaseSS
end

"""
`displace_squeeze_overlap(β,φ,q,α,θ,p)` is
ᵍ⟨β,φ,q|α,θ,p⟩ᵍ, i.e. ᵍ⟨β,0,q⋅exp(-i2φ)|α,0,p⋅exp(-i2θ)⟩ᵍ
"""
function displace_squeeze_overlap(β,φ,q,α,θ,p)
    phaseDD = displace_prodphase(-β,α)
    γ = squeeze_displace_commparam(α-β,φ,-q)
    z, phaseSS = squeeze_prod(-q,-2φ,p,-2θ)
    displace_squeeze_overlap(-γ,z)*phaseDD*phaseSS
end

"""
`displace_f(α)*displace_f(β) = displace_f(α+β)*displace_sumphase(α,β)`
"""
displace_prodphase(α,β) = exp((α*conj(β)-conj(α)*β)/2)

"""
`squeeze_f(z)*displace_f(α)=displace_f(squeeze_displace_commparam(α,z))*squeeze_f(z)`
"""
squeeze_displace_commparam(α,z) = α*cosh(abs(z))-conj(α)*exp(im*angle(z))*sinh(abs(z))
squeeze_displace_commparam(α,θ,z) = α*cosh(z)-conj(α)*exp(-2im*θ)*sinh(z)

squeeze_z_to_t(z) = exp(im*angle(z)) * tanh(abs(z)) # https://physics.stackexchange.com/questions/67371
squeeze_t_to_z(t) = exp(im*angle(t)) * atanh(abs(t)) # https://physics.stackexchange.com/questions/67371
"""
```
z3,phase = SimpleQGaussian.squeeze_prod(z1,z2)
squeeze_f(z1,n)*squeeze_f(z2,n))[:,1] = phase*squeeze_f(z3,n)[:,1]
```
"""
function squeeze_prod(z1,z2) # https://physics.stackexchange.com/questions/67371
    t1 = squeeze_z_to_t(z1)
    t2 = squeeze_z_to_t(z2)
    t3 = (t1+t2)/(1+conj(t1)*t2)
    z3 = squeeze_t_to_z(t3)
    p = log((1+t1*conj(t2))/(1+conj(t1)*t2)) # This is always imaginary it seems...
    phase = exp(p/4)
    z3, phase
end
function squeeze_prod(r1,angle1,r2,angle2) # https://physics.stackexchange.com/questions/67371
    t1 = tanh(r1)
    t2 = tanh(r2)
    ct1t2 = t1*t2*exp(im*(angle2-angle1))
    t3 = (t1*exp(im*angle1)+t2*exp(im*angle2))/(1+ct1t2)
    z3 = squeeze_t_to_z(t3)
    p = log((1+conj(ct1t2))/(1+ct1t2)) # This is always imaginary it seems...
    phase = exp(p/4)
    z3, phase
end

function inner(l::AbstractPureGauss,r::AbstractPureGauss)
    β = camplitude(l)
    φ = rotparam(l)
    q = squeezeparam(l)
    α = camplitude(r)
    θ = rotparam(r)
    p = squeezeparam(r)
    displace_squeeze_overlap(β,φ,q,α,θ,p)
end

absinner(l::AbstractPureState,r::AbstractPureState) = abs(inner(l,r))
function absinner(l::AbstractPureGauss,r::AbstractPureGauss) # using arXiv:quant-ph/0410111
    d = quadrature(l)-quadrature(r)
    V₁ = covariance(l)
    V₂ = covariance(r)
    V = V₁+V₂
    Δ = det(V)
    √(2/√Δ) * exp(-(d'*inv(V)*d)/4)
end

end #module
