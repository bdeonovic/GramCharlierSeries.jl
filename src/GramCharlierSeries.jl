module GramCharlierSeries
using Distributions
using Combinatorics
using SpecialFunctions
using TaylorSeries

abstract type KolmogorovApproximation end
struct KolmogorovBinomialSeedApproximation <: KolmogorovApproximation end
struct KolmogorovNormalSeedApproximation <: KolmogorovApproximation end

include("kolmogorov.jl")
include("gc.jl")

export poisbin_approx_kolmogorov, poisbin_approx_gramcharlier, poisbin_approx_gramcharlier_log,
       KolmogorovApproximation, KolmogorovBinomialSeedApproximation,
       KolmogorovNormalSeedApproximation
end # module
