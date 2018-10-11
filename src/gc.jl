function probabilist_hermite_recursive(x::T, n::Integer) where T <: Real
    S = Vector{T}(undef, n+1)
    he!(S,x)
    return S
end

function he!(S::AbstractVector{T}, x::T) where T <: Real
    n = length(S)
    S[1] = one(T)
    S[2] = x
    @inbounds for i in 3:n
        S[i] = x * S[i-1] - (i-2) * S[i-2]
    end
end

function complete_bell_polynomial(x::AbstractVector{T}) where T <: Real
    m = length(x)
    S = Vector{T}(undef, m+1)
    S[1] = one(T)
    @inbounds for n in 0:(m-1)
        for i in 0:n
            S[n+2] += binomial(n,i) * S[n-i+1] * x[i+1]
        end
    end
    return S
end

function poisbin_approx_gramcharlier_log(p::AbstractVector{T}, moment_order::Integer=4) where T <: Real
    n = length(p)
    function f(t::T) where T
      result = zero(T)
      @inbounds for p_i in p
        result += log(1.0 - p_i + p_i * exp(t))
      end
      return result
    end
    target_cumulants = Vector{T}(undef, moment_order)
    fT = f(Taylor1(moment_order+1))
    @inbounds for k in 1:moment_order
        target_cumulants[k] = getcoeff(fT, k) * factorial(k)
    end
    d = Normal(target_cumulants[1], sqrt(target_cumulants[2])) 
    S = Vector{T}(undef, n+1)
    mod_cumulants = [0.0; 0.0; target_cumulants[3:end]]
    B = complete_bell_polynomial(mod_cumulants)
    He = Vector{T}(undef, moment_order+1)
    C = factorial.(0:moment_order) .* (sqrt(target_cumulants[2]) .^ (0:moment_order))
    @inbounds for k in 0:n
        he!(He, (k - target_cumulants[1]) / sqrt(target_cumulants[2]) )
        correction = max(sum(B .* He ./ C), zero(T))
        S[k+1] = logpdf(d, k) + log(correction)
    end
    return S
end

function poisbin_approx_gramcharlier(p::AbstractVector{T}, moment_order::Integer=4) where T <: Real
    exp.(poisbin_approx_gramcharlier_log(p, moment_order))
end