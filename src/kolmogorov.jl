function seeds_and_moments!(seed_probabilities::AbstractVector{T}, 
                            seed_moments::AbstractVector{T}, d::Distribution,
                            g::Function, n::Integer) where T <: Real
    moment_order = length(seed_moments)
    seed_probabilities .= pdf.(d, 0:n)

    gT = g(Taylor1(moment_order+1))
    for k in 1:moment_order
        seed_moments[k] = getcoeff(gT, k) * factorial(k)
    end
end

function seeds_and_moments(p::AbstractVector{T}, moment_order::Integer, 
                           method::KolmogorovBinomialSeedApproximation) where T <: Real
    n = length(p)
    d = Binomial(n, sum(p)/n)
    g(t) = (1.0 - d.p + d.p * exp(t)) ^ d.n
    seed_probabilities = Vector{T}(undef, n+1)
    seed_moments = Vector{T}(undef, moment_order)
    seeds_and_moments!(seed_probabilities, seed_moments, d, g, n)
    return seed_probabilities, seed_moments
end
function seeds_and_moments(p::AbstractVector{T}, moment_order::Integer, 
                           method::KolmogorovNormalSeedApproximation) where T <: Real
    n = length(p)
    initial_mu = sum(p)
    initial_sigma2 = sum(p .* (1.0 .- p))

    d = Normal(initial_mu, sqrt(initial_sigma2))
    g(t) = exp(initial_mu * t + 0.5 * initial_sigma2 * t^2)
    seed_probabilities = Vector{T}(undef, n+1)
    seed_moments = Vector{T}(undef, moment_order)
    seeds_and_moments!(seed_probabilities, seed_moments, d, g, n)
    return seed_probabilities, seed_moments
end

function poisbin_approx_kolmogorov(p::AbstractVector{T}, moment_order::Integer=6, 
                                   method::KolmogorovApproximation=
                                     KolmogorovBinomialSeedApproximation()) where T <: Real
    n = length(p)
    target_probabilities = Vector{T}(undef, n+1)
    seed_probabilities, seed_moments = seeds_and_moments(p, moment_order, method)
    delta = Vector{T}(undef, n+1)
    delta[1] = seed_probabilities[1]
    delta[2:end] .= seed_probabilities[2:end] .- seed_probabilities[1:(end-1)]

    function f(t::T) where T
      result = one(T)
      for p_i in p
        result *= (1 - p_i + p_i * exp(t))
      end
      return result
    end
    target_moments = Vector{T}(undef, moment_order)
    fT = f(Taylor1(moment_order+1))
    for k in 1:moment_order
        target_moments[k] = getcoeff(fT, k) * factorial(k)
    end
    kolmogorov!(target_probabilities, delta, seed_probabilities, seed_moments, target_moments)
    return target_probabilities
end

function kolmogorov!(target_probabilities::AbstractVector{T}, 
                     delta::AbstractVector{T},
                     seed_probabilities::AbstractVector{T}, 
                     seed_moments::AbstractVector{T}, 
                     target_moments::AbstractVector{T}) where T <: Real
    moment_order = length(target_moments)

    #calculate first matching coefficient
    mu = Vector{T}(undef, moment_order)
    mu[1] = seed_moments[1]

    a = Vector{T}(undef, moment_order)
    a[1] = -(target_moments[1] - mu[1])
    target_probabilities .= seed_probabilities .+ a[1] .* delta
    for k in 2:moment_order
        #find the kth differences
        delta[2:end] .= delta[2:end] .- delta[1:(end-1)]

        #calculate mu
        mu[k] = seed_moments[k]
        for j in 1:k-1
            for i in j:k
                ckji = binomial(k,i) * (-1)^j * factorial(j) * stirlings2(i,j)
                mu[k] += k-i > 0 ? a[j] * ckji * seed_moments[k-i] : a[j] * ckji
            end
        end
        #calculate a
        a[k] = (-1)^k * (target_moments[k] - mu[k]) / factorial(k)

        #calculate p
        target_probabilities .= target_probabilities .+ a[k] .* delta
    end
end
