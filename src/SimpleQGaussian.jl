module SimpleQGaussian

using LinearAlgebra

export Coherent, coh, Gauss, gauss, PureSuperposition,
    quadrature, cquadrature, covariance, squeezeparam, rotparam,
    inner

rmat(θ) = [cos(θ) sin(θ);
          -sin(θ) cos(θ)]

smat(r) = [exp(-r) 0;
           0       exp(r)]

abstract type AbstractPureState end
abstract type AbstractPureGauss <: AbstractPureState end

struct Coherent{T<:Complex} <: AbstractPureGauss
    alpha::T
    Coherent(a) = new{typeof(Complex(a))}(Complex(a)) # TODO there should be a pretty way to write this
end

Coherent(x,p) = Coherent(x+1im*p)

coh = Coherent

struct Gauss{T<:Real} <: AbstractPureGauss
    loc::Vector{T} # TODO maybe the vector and matrix here should be static
    cov::Matrix{T} # TODO or maybe they should be Abstract, but static by default
end

function Gauss(x,p,θ,r)
    x,p,θ,r = [convert(AbstractFloat, a) for a in (x,p,θ,r)]
    Gauss([x,p],rmat(θ)*smat(2r)*rmat(θ)')
end
Gauss(q,θ,r) = Gauss(real(q),imag(q),θ,r)
Gauss(c::Coherent) = Gauss(quadrature(c),collect(covariance((c))*one(eltype(quadrature(c))))) # TODO there should be a pretty way to write this
Base.convert(::Type{Gauss}, x) = Gauss(x)

gauss = Gauss

quadrature(s::Coherent) = [real(s.alpha), imag(s.alpha)]
quadrature(s::Gauss) = s.loc

cquadrature(s::Coherent) = s.alpha
cquadrature(s::Gauss) = s.loc[1]+1im*s.loc[2]

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
        print(io, "ᵍ⟨$(cquadrature(g)), $(rotparam(g)), $(squeezeparam(g))|")
    else
        print(io, "|$(cquadrature(g)), $(rotparam(g)), $(squeezeparam(g))⟩ᵍ")
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

function inner(l::PureSuperposition, r::PureSuperposition)
    c = zero(promote_type(eltype(l.weights),eltype(r.weights)))
    for li in eachindex(l.weights)
        for ri in eachindex(r.weights)
            c += conj(l.weights[li])*r.weights[ri]*inner(l.states[li],r.states[ri])
        end
    end
    c
end

inner(l::Coherent, r::Coherent) =exp(-abs2(cquadrature(l)-cquadrature(r))/2)

function inner(l::AbstractPureGauss,r::AbstractPureGauss)
    V1 = covariance(l)
    V2 = covariance(r)
    Λ1 = inv(V1)
    Λ2 = inv(V2)
    q1 = quadrature(l)
    q2 = quadrature(r)
    η1 = Λ1*q1
    η2 = Λ2*q2
    ζ1 = - (2*log(2pi) - log(det(Λ1)) + η1'*V1*η1) / 2
    ζ2 = - (2*log(2pi) - log(det(Λ2)) + η2'*V2*η2) / 2
    Λs = Λ1+Λ2
    ηs = η1+η2
    ζs = - (2*log(2pi) - log(det(Λs)) + ηs'*inv(Λs)*ηs) / 2
    (exp(ζ1+ζ2-ζs)*4pi)^2 # TODO WTF is this *4pi)^2
end

end #module
