module BetheHessian

using NLopt
using Docile
using Lexicon

@docstrings


export 

#Matrix completion
demo_MC,
complete

#Community detection
#TBD

#source 
Docile.@doc """
`Matrix Completion with the Bethe Hessian (MaCBetH)`

demo code. run `demo_MC()` to run an example with default parameters.
demo_MC accepts keyword arguments to set a subset of the parameters.

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
function demo_MC(;n::Int = 1000,m::Int = 1000,rank::Int = 10, epsilon = 50,Delta = 0,stop_val::Float64 = 1e-10, maxiter::Int = 100,tol_bet::Float64 = 1e-4,force_rank::Bool = false,verbose::Bool = true,max_rank::Int=0,opt_algo::Symbol = :LD_LBFGS,regul::Float64 = 0.0)   	
	
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
		X_inferred,Y_inferred,r = complete(A_obs,tol_bet = tol_bet,stop_val = stop_val,maxiter = maxiter,max_rank = max_rank,verbose = verbose,opt_algo = opt_algo,regul = regul)
	else 
		# Specified rank
		X_inferred,Y_inferred,r = complete(A_obs,tol_bet = tol_bet,stop_val = stop_val,maxiter = maxiter,force_rank = rank,verbose = verbose,opt_algo = opt_algo,regul = regul)
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

Docile.@doc """
`Matrix Completion with the Bethe Hessian (MaCBetH)`

Main completion function. Usage : inferred_X,inferred_Y,inferred_r = `complete`(M)
where M is the matrix to be completed. Note that the input observation matrix should be **centered**.
Returns the inferred factors X and Y such that M ≈ XY', and
the inferred rank r. `complete` accepts keyword arguments to set a subset of the parameters.
See demo_MC.jl for an example on a synthetic low-rank matrix.

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
function complete(A;tol_bet::Float64 = 0.001,stop_val::Float64 = 1e-10,maxiter::Int = 100, force_rank::Int=0, max_rank::Int=0,verbose::Bool=false,opt_algo::Symbol = :LD_LBFGS,regul::Float64 = 0.0)

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
		if opt_algo == :ALS
			@time inferred_X,inferred_Y = ALS(X0,Y0,A,regul,stop_val,maxiter)
		else
			starting_vec = vec([reshape(X0,n*r,1) ; reshape(Y0,m*r,1)])
			inferred_X,inferred_Y = local_optimization(starting_vec,r,i,j,s,n,m,stop_val,maxiter,verbose = verbose,opt_algo = opt_algo)
		end
	end
	return inferred_X,inferred_Y,r
end

function build_BH(J,tol_bet,c_1,c_2,verbose)
	lin,col,val = findnz(J)
	N = length(J[1,:])
	bet = solve_beta(J,tol_bet,c_1,c_2)

	if maximum(abs(tanh(bet*val))) < 0.99
		if verbose
			println("Using canonical formulation of the Bethe Hessian")
		end
		num_obs = length(lin)
		linBH = [lin; collect(1:N);]
		colBH = [col; collect(1:N);]
		valBH = [ zeros(num_obs) ; ones(N) ;]

		for i=1:num_obs
			# X = tanh(bet*val[i])
			# Y = X/(1-X^2)
			valBH[i] = -0.5*sinh(2.0*bet*val[i])
			valBH[num_obs+lin[i]] += sinh(bet*val[i])^2
		end 
		BH = sparse(linBH,colBH,valBH,N,N)
	else 
		if verbose
			println("Using signed formulation of the Bethe Hessian for better numerical stability")
		end
		J1 = sparse(lin,col,sign(val),N,N)
		c = sqrt(c_1*c_2)
		D = vec(sum(abs(J1),1));
		BH = (c-1)*speye(N)-sqrt(c)*J1+spdiagm(D)
	end
	return BH
end

function infer_BH(BH,n,m,force_rank,max_rank;verbose::Bool=true)

	N = length(BH[1,:])
	Tr = trace(BH)
	BH2 = 1/N*(Tr*speye(N,N) - BH)

	if max_rank >0

		d,v,nconv = eigs( BH2 ; which = :LM , nev=max_rank);
		id = sortperm(d,rev=true)
		d = d[id]
		v = v[ :, id ]
		r = 0

		for i=1:max_rank
			if d[i]>1/N*Tr
				r += 1
			end
		end

		r = min(r,nconv)

		if r == 0
			warn("The Bethe Hessian found that the rank is 0 : failure. Will output 0.")
			X0 = 0
			Y0 = 0
			return X0,Y0,r
		elseif r == max_rank
			w=string("The rank is possibly larger than ",max_rank,". You might want to increase max_rank")
			warn(w)
		end
		if verbose
			println("Using an inferred rank equal to ", r ," for the factorization")
		end
		X0 = v[1:n,1:r]
		Y0 = v[n+1:m+n,1:r]
	else

		d,v = eigs( BH2 ; which = :LM , nev=force_rank)
		id = sortperm(d,rev=true)
		d = d[id]
		v = v[ :, id ]
		r = force_rank
		if verbose
			println("Using the specified rank ", r ," for the factorization")
		end
		X0 = v[1:n,1:r]
		Y0 = v[n+1:m+n,1:r]

	end

	return X0,Y0,r  
end

function solve_beta(J,tol,c_1,c_2)
	lin,col,val = findnz(J)
	# N = length(J[1,:])

	bet_min = 0.0
	bet_max = 1000.0
	current_value = F( 1/2 * (bet_max + bet_min),lin,col,val,c_1,c_2)
	
	while bet_max - bet_min > tol

		if current_value > 0
			bet_max = 1/2 * (bet_max + bet_min)
		else 
			bet_min = 1/2 * (bet_max + bet_min)
		end
		current_value = F( 1/2 * (bet_max + bet_min),lin,col,val,c_1,c_2)
		
	end

	bet = 1/2 * (bet_max + bet_min)

	return bet
end

function F(bet,lin,col,val,c_1,c_2)
	num_obs = length(lin)
	y = 0.0

	for i=1:num_obs
		y += tanh(bet*val[i])^2 
	end

	y = sqrt(c_1*c_2)*y/num_obs - 1
end

function ALS(X0,Y0,A,regul,stop_val,maxiter)
	n = size(X0,1)
	m = size(Y0,1)
	r = size(X0,2)
	At = A'
	X = X0'
	Y = Y0'
	iter = 0
	accuracy = stop_val + 0.1
	
	while iter < maxiter && accuracy > stop_val
		iter += 1
		X = ALS_update_X(Y,n,r,A,At,regul)
		Y = ALS_update_Y(X,m,r,A,At,regul)
	end

	return X',Y'
end

function ALS_update_X(Y,n,r,A,At,regul)

	X = zeros(r,n)	
	for i = 1:n
		
		neigh = find(At[:,i])
		num = zeros(r,1)
		denom = regul*eye(r,r)
		
		for k = 1:length(neigh)
			Base.LinAlg.BLAS.axpy!(r,A[i,neigh[k]],Y[:,neigh[k]],1,num,1)
			Base.LinAlg.BLAS.gemm!('N','T',1.0,Y[:,neigh[k]],Y[:,neigh[k]],1.0,denom)
		end
		
		X[:,i] = inv(denom)*num
	end
	return X
end

function ALS_update_Y(X,m,r,A,At,regul)
	Y = zeros(r,m)	
	for i = 1:m
		neigh = find(A[:,i])
		num = zeros(r,1)
		denom = regul*eye(r,r)
		for k = 1:length(neigh)
			Base.LinAlg.BLAS.axpy!(r,A[neigh[k],i],X[:,neigh[k]],1,num,1)
			Base.LinAlg.BLAS.gemm!('N','T',1.0,X[:,neigh[k]],X[:,neigh[k]],1.0,denom)
		end
		Y[:,i] = inv(denom)*num
	end
	return Y
end

function local_optimization(starting_vec,r,I,J,VAL,n,m,stop_val,maxiter;verbose::Bool=true,opt_algo::Symbol =:LD_LBFGS)
	global count
	count = 0
	size_prob = length(starting_vec)
	stop_val_cost_function = (length(VAL))*stop_val^2

	opt = Opt(opt_algo, size_prob)

	if verbose 
		println("Optimization method :",opt_algo)
	end

	min_objective!(opt, (x,g) -> myfunc(x,g,I,J,VAL,r,n,m,verbose,stop_val_cost_function,maxiter) )
	maxeval!(opt, maxiter)	
	stopval!(opt, stop_val_cost_function)

	(minf,minx,ret) = optimize(opt,starting_vec)

	inferred_X = reshape(minx[1:n*r],n,r)
	inferred_Y = reshape(minx[n*r+1:(n+m)*r],m,r)

	if verbose
		if ret == :STOPVAL_REACHED
			println("Optimization completed : stopping value reached (RMSE < $stop_val)")
		end
		if ret == :MAXEVAL_REACHED
			println("Optimization completed : maximum number of iterations reached ($maxiter)")
		end
	end

	return inferred_X,inferred_Y

end

function myfunc(x::Vector, grad::Vector,I,J,VAL,r,n,m,verbose::Bool,stop_val_cost_function,maxiter)

	rep=0.0
	if length(grad) > 0
		for a = 1:length(x)
			grad[a] = 0.0
		end
	end

	for k = 1:length(VAL)
		xyt = 0.0
		i = I[k]
		j = J[k]

		for mu = 1:r
			xyt += x[i+(mu-1)*n]*x[n*r+j+(mu-1)*m]
		end 

		if length(grad) > 0
			for mu=1:r
				grad[i+(mu-1)*n] += 2.0*(-VAL[k] + xyt)*x[n*r+j+(mu-1)*m]
				grad[n*r+j+(mu-1)*m] += 2.0*(-VAL[k] + xyt)*x[i+(mu-1)*n]
			end
		end

		rep += (VAL[k]-xyt)^2

	end
	global count
	count::Int +=1
	if verbose && (mod(count,20) == 1 || rep < stop_val_cost_function || count == maxiter)
		println("Iteration $count,\t RMSE on observed entries = ",sqrt( 1.0/(length(VAL))*rep ))
	end
	return rep
end

end # module
