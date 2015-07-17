using BetheHessian
using Base.Test

# write your own tests here
# Testing matrix completion
@test demo_MC(n=10,m=10,rank=1,epsilon=10)

