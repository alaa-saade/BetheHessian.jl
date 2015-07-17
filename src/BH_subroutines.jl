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

