#PearsonIII moments
#mu0 is center of generalized beta
#sigma is sigma of generlaized beta
#a and b are a and b of generalized beta
EXk = cumprod([1;[(a + r) / (a + b + r) for r in 0:(moment_order-1)]])
[sum(binomial.(j, 0:j) .* ((sigma .^ (j:-1:0))) .* EXk[(j:-1:0) .+ 1] .* ( mu0 .^ (0:j))) for j in 1:moment_order]


function binomial_6_cumulants(n, p)
    [n*p, n*p*(1-p), n*p*(1-p)*(1-2p), n*p*(1-p)*(1-6p*(1-p)), 
         n*p*(1-p)*(1-2p)*(1-12p+12p^2), 
         n*p*(1-p)*(1-30p+150p^2-240p^3+120p^4)]
end

function normal_6_moments(mu, sigma2)
    [mu, 
     mu^2 + sigma2, 
     mu^3+3mu*sigma2, 
     mu^4+6mu^2*sigma2+3sigma2^2,
     mu^5+10mu^3*sigma2+15mu*sigma2^2,
     mu^6+15mu^4*sigma2+45mu^2*sigma2^2+15*sigma2^3]
end
