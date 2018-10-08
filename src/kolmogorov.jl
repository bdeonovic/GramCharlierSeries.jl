
function cumulants_to_moments(k)
    [k[1],k[2]+k[1]^2, k[3]+3k[2]*k[1]+k[1]^3, 
         k[4]+4*k[3]*k[1]+3k[2]^2+6k[2]*k[1]^2+k[1]^4,
         k[5]+5k[4]*k[1]+10k[3]*k[2]+10k[3]*k[1]^2+15k[2]^2*k[1]+10k[2]*k[1]^3+k[1]^5,
         k[6]+6k[5]*k[1]+15k[4]*k[2]+15k[4]*k[1]^2+10k[3]^2+60k[3]*k[2]*k[1]+20k[3]*k[1]^3+
           15k[2]^3+45k[2]^2*k[1]^2+15k[2]*k[1]^4+k[1]^6]
end

function fitkolmogorov_binomialseed(n::Vector{Int}, p::Vector{T}, moment_order::Integer=6) where T <: Real
    #calculate probabilities p0 of initial approximating distribution E0
    initial_n = sum(n)
    initial_p = sum(n.*p)/initial_n

    pA = zeros(T, initial_n+1,moment_order+1)
    pA[:,1] = pdf.(Binomial(initial_n,initial_p),0:initial_n)

    #find first differences
    delta = zeros(T, initial_n+1, moment_order+1)
    delta[:,1] = [pA[1,1]; pA[2:end,1] - pA[1:end-1,1]]

    #find lth moment of true distribution l=1,...,6
    function f(t::T) where T
      result = one(T)
      for pi in p
        result *= (1 - pi + pi * exp(t))
      end
      return result
    end
    v = [getcoeff(f(Taylor1(moment_order+1)), k)*factorial(k) for k in 1:moment_order]

    #calculate moments of initial approximation
    mu = zeros(moment_order,moment_order)
    g(t) = (1 - initial_p + initial_p * exp(t))^initial_n
    mu[:,1] = [getcoeff(g(Taylor1(moment_order+1)), k)*factorial(k) for k in 1:moment_order]

    # calculate first matching coefficient
    a = zeros(moment_order)
    a[1] = -(v[1] - mu[1,1])
    pA[:,2] = pA[:,1] + a[1]*delta[:,1]
    for l in 2:moment_order
        #find the kth differences
        delta[:,l] = [delta[1,l-1]; delta[2:end,l-1] - delta[1:end-1,l-1]]

        #calculate mu
        mu[l,l] = mu[l,1]
        for j in 1:l-1
            for i in j:l
                clji = binomial(l,i)*(-1)^j*factorial(j)*stirlings2(i,j)
                mu[l,l] += l-i > 0 ? a[j] * clji * mu[l-i,1] : a[j] * clji
            end
        end
        #calculate a
        a[l] = (-1)^l *(v[l]-mu[l,l])/factorial(l)

        #calculate p
        pA[:,l+1] = pA[:,l] + a[l]*delta[:,l]
        #pA[:,l+1] = pA[:,l+1] ./ sum(pA[:,l+1])
    end
    return pA[:,end]
end
function fitkolmogorov_normalseed(p::Vector{T}, moment_order::Integer=6) where T <: Real
    #calculate probabilities p0 of initial approximating distribution E0
    n = length(p)
    initial_mu = sum(p)
    initial_sigma2 = sum(p .* (1 .- p))

    pA = zeros(T,n+1,moment_order+1)
    pA[:,1] = pdf.(Normal(initial_mu, sqrt(initial_sigma2)), 0:n)

    #find first differences
    delta = zeros(T, n+1, moment_order)
    delta[:,1] = [pA[1,1]; pA[2:end,1] - pA[1:end-1,1]]

    #find lth moment of true distribution
    function f(t::T) where T
      result = one(T)
      for pi in p
        result *= (1 - pi + pi * exp(t))
      end
      return result
    end
    v = [getcoeff(f(Taylor1(moment_order+1)), k)*factorial(k) for k in 1:moment_order]

    #calculate moments of initial approximation
    mu = zeros(moment_order,moment_order)
    g(t) = exp(initial_mu*t + 0.5*initial_sigma2*t^2)
    mu[:,1] = [getcoeff(g(Taylor1(moment_order+1)), k)*factorial(k) for k in 1:moment_order]

    # calculate first matching coefficient
    a = zeros(moment_order)
    a[1] = -(v[1] - mu[1,1])
    pA[:,2] = pA[:,1] + a[1]*delta[:,1]
    for l in 2:moment_order
        #find the kth differences
        delta[:,l] = [delta[1,l-1]; delta[2:end,l-1] - delta[1:end-1,l-1]]

        #calculate mu
        mu[l,l] = mu[l,1]
        for j in 1:l-1
            for i in j:l
                clji = binomial(l,i)*(-1)^j*factorial(j)*stirlings2(i,j)
                mu[l,l] += l-i > 0 ? a[j] * clji * mu[l-i,1] : a[j] * clji
            end
        end
        #calculate a
        a[l] = (-1)^l *(v[l]-mu[l,l])/factorial(l)

        #calculate p
        pA[:,l+1] = pA[:,l] + a[l]*delta[:,l]
    end
    return pA[:,end]
end
