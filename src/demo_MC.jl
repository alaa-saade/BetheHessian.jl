@doc doc"""
`MaCBetH : Matrix Completion with the Bethe Hessian`

demo code. run `demo()` to run an example with default parameters.
demo accepts keyword arguments to set a subset of the parameters.

List of keywords 

*`n::Int` : number of lines of the random matrix to be completed (default 1000)

*`m::Int` : number of columns of the random matrix to be completed (default 1000)

*`rank::Int` : rank of the matrix to be completed (default 10)

*`epsilon` : average number of revealed entries per row or column (see paper) (default 50)

*`Delta` : variance of gaussian additive noise (default 0)

*`stop_val::Float64` : stoping value of the NLopt solver. The optimization will stop if the RMSE on the observed entries is smaller than stop_val (default 1e-10)

*`maxiter::Int` : (approximate) maximum number of iterations of the NLopt solver (default 100)

*`tol_bet::Float64` : tolerance of the numerical solver for the parameter beta (default 1e-4)

*`force_rank::Bool` : set to true to force the algorithm to use the correct rank. 
By default, force_rank is set to false and Macbeth tries to infer the correct rank. 

*`max_rank::Int` : maximum possible rank when inferring the rank (default rank+1)

*`opt_algo::Symbol` : algorithm for the non-linear optimization. Choices include :LD_LBFGS, :LD_TNEWTON, :LD_VAR2, ... See [http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms](http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms) for a full list (default :LD_LBFGS)
	
*`verbose::Bool` : set to false to prevent the code from talking (default true)
""" ->
function demo_MC(;n::Int = 1000,m::Int = 1000,rank::Int = 10, epsilon = 50,Delta = 0,stop_val::Float64 = 1e-10, maxiter::Int = 100,tol_bet::Float64 = 1e-4,force_rank::Bool = false,verbose::Bool = true,max_rank::Int=0,opt_algo::Symbol = :LD_LBFGS)   	
	# srand(1)
	if max_rank == 0
		max_rank = rank+1
	end

	X = randn(n,rank)
	Y = randn(m,rank)

	noise = sprandn(n,m,epsilon/sqrt(n*m))
	obs = spones(noise)
	noise = sqrt(Delta)*noise
	true_A = X*Y'
	A_obs = true_A.*obs + noise 

	if verbose
		println("Observation matrix computed : rank ",rank,", ",n,"x",m," matrix," )
		println("with ", round(nnz(A_obs)/(n*m)*100,2), "% observed entries (",nnz(A_obs),")" );
	end

	if !force_rank
		# Unspecified rank
		X_inferred,Y_inferred,r = complete(A_obs,tol_bet = tol_bet,stop_val = stop_val,maxiter = maxiter,max_rank = max_rank,verbose = verbose,opt_algo = opt_algo)
	else 
		# Specified rank
		X_inferred,Y_inferred,r = complete(A_obs,tol_bet = tol_bet,stop_val = stop_val,maxiter = maxiter,force_rank = rank,verbose = verbose,opt_algo = opt_algo)
	end
	if r == 0
		RMSE = sqrt(mean((true_A - A_obs).^2))
	else
		RMSE = sqrt(mean((true_A - X_inferred*Y_inferred').^2))
	end
	if verbose
		println("Reconstruction error (RMSE) on full matrix with a rank ",r," approach : ",RMSE)
	end
	return RMSE
end
