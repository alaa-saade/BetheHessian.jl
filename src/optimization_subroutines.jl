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
