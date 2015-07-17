using BetheHessian
using Base.Test

# write your own tests here
println("Testing Matrix Completion")
MC_handler(r::Test.Success) = println("Matrix Completion working")
MC_handler(r::Test.Failure) = error("Matrix Completion not working")
MC_handler(r::Test.Error) = rethrow(r)

Test.with_handler(MC_handler) do
	@test demo_MC(force_rank = true,n=10,m=10,rank=1,epsilon=10,verbose=false)<1
end


