module BetheHessian

using NLopt

export 

#Matrix completion
demo_MC,
complete

#Community detection

include("demo_MC.jl")
include("complete.jl")
include("BH_subroutines.jl")
include("optimization_subroutines.jl")


end # module
