@doc doc"""
`MaCBetH : Matrix Completion with the Bethe Hessian`

Main completion function. Usage : inferred_X,inferred_Y,inferred_r = `complete`(M)
where M is the matrix to be completed. Returns the inferred factors X and Y such that M ≈ XY', and
the inferred rank r. `complete` accepts keyword arguments to set a subset of the parameters.
See demo.jl for an example on a synthetic low-rank matrix.
Note that the input observation matrix should be `centered`.

List of keywords 

*`tol_bet::Float64` : tolerance of the numerical solver for the parameter beta (default 1e-4)

*`stop_val::Float64` : stoping value of the NLopt solver. The optimization will stop if the RMSE on the observed entries is smaller than stop_val (default 1e-10)

*`maxiter::Int` : (approximate) maximum number of iterations of the NLopt solver (default 100)

*`force_rank::Int` : set to nonzero value to force Macbeth to use the specified rank. 
Either force_rank or max_rank should be set to a nonzero value.

*`max_rank::Int` : Number of eigenvalues of the hessian to be computed. If all the eigenvalues 
computed are negative (i.e. if the inferred rank is larger than max_rank), Macbeth will 
give you a warning. Either force_rank or max_rank should be set to a nonzero value.

*`opt_algo::Symbol` : algorithm for the non-linear optimization. Choices include :LD_LBFGS, :LD_TNEWTON, :LD_VAR2, ... See [http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms](http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms) for a full list (default :LD_LBFGS)
	
*`verbose::Bool` : set to false to prevent the code from talking (default true)
""" ->
function complete(A;tol_bet::Float64 = 0.001,stop_val::Float64 = 1e-10,maxiter::Int = 100, force_rank::Int=0, max_rank::Int=0,verbose::Bool=true,opt_algo::Symbol = :LD_LBFGS)

	if max_rank==0 && force_rank==0
		error("Either max_rank or force_rank should be >0")
	end
	if max_rank!=0 && force_rank!=0
		error("Either max_rank or force_rank should be equal to 0")
	end
	if max_rank<0 || force_rank<0
		error("max_rank and force_rank should be non-negative integer")
	end

	if max_rank >0 && verbose
		println("Completion with unspecified rank (max_rank = ",max_rank,")")
	elseif force_rank>0 && verbose 
		println("Completion with specified rank (force_rank = ",force_rank,")")
	end

	n,m = size(A)
	A1 = spones(A);
	c_1 = mean(sum(A1,1))
	c_2 = mean(sum(A1,2))
	
	# bi-partite graph :
	i,j,s = findnz(A)

	J = [spzeros(n,n) sparse(i,j,s,n,m) ; sparse(j,i,s,m,n) spzeros(m,m);]

	BH = build_BH(J,tol_bet,c_1,c_2,verbose)
	if verbose
		println("Bethe Hessian built")
	end

	X0,Y0,r = infer_BH(BH,n,m,force_rank,max_rank,verbose = verbose)
	if r == 0
		inferred_X = 0
		inferred_Y = 0
	else
		if verbose
			println("Initial inference done, proceeding to local optimization")
		end
		starting_vec = vec([reshape(X0,n*r,1) ; reshape(Y0,m*r,1)])
		inferred_X,inferred_Y = local_optimization(starting_vec,r,i,j,s,n,m,stop_val,maxiter,verbose = verbose,opt_algo = opt_algo)
	end
	return inferred_X,inferred_Y,r
end