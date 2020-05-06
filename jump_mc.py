# Control Variate MC for arithmetic Asian options;
# Geometric avg Asian is used as the control
import time, sys, math 
import numpy as np
import scipy.stats as scp
from jump_sample import jump_diffusion

def geo_asian(S0, K, r, sigma, T, m):
    sigsqT = sigma*sigma*T*(2*m+1)/(6*m+6)
    muT = 0.5*sigsqT + 0.5*(r - 0.5*sigma*sigma)*T
    d1 = (math.log(S0/K) + (muT + 0.5*sigsqT))/(math.sqrt(sigsqT))
    d2 = d1 - math.sqrt(sigsqT)
    geo_call = math.exp(-r*T)*( S0*math.exp(muT)*scp.norm.cdf(d1)-K*scp.norm.cdf(d2))    
    return geo_call

def jump_diff_mc(S0=[35], K=20, r=0.05, sigma=0.25, T=1, N=100, lamb=10, gamma=0, delta=0.2):
    start_time = time.time()
    dt = T/N
    d = S0.shape[0]

    # jump_diffusion(S=S0, X=K, T=1, mu=0.05, sigma=0.25, Lambda=lamb,
                       # a=(gamma - delta*delta/2), b=delta, Nsteps=N, Nsim=20000, alpha=0.05, seed=None)
       
    M_list = [50000]
    converge_rate = np.zeros(len(M_list))
    hat_C_M = np.zeros(len(M_list))
    hat_sigma_M = np.zeros(len(M_list))
    
    print("\n\n***Monte Carlo Estimation's Results:\n")
    
    for i in range(len(M_list)):
        M = M_list[i]
        Z = np.random.normal(0, 1, size=(N, M, d))

        dN = np.random.poisson(lamb*dt, [N, M, d])
        logj = np.random.normal(gamma - delta*delta/2, delta, size=(N, M, d))
        J = np.exp(logj)

        S = np.ones([M, 1])*S0
        S_avg = np.ones([M, 1])*S0/(T*(N+1))
        # S_avg_geo = math.pow(S0,1/(N+1))*np.ones(M)

        for n in range(N):
            S = S * ( 1 + r * dt + sigma * math.sqrt(dt) * np.tile(np.reshape(np.sum(Z[n], axis=1), [M , 1]), [1, d])) \
                                                                    + S*(J[n] - 1)*dN[n] #- S*lamb*(math.exp(gamma)-1)*dt
            S[np.where(S<0)] = 0
            # X = np.tile(np.reshape(np.sum(Z[n], axis=1), [M , 1]), [1, d]) 
            # print(X.shape)
            # S_avg = S_avg + S / (T*(N+1))

        payoff = (np.amax(S, axis=1) - K)*np.greater_equal(np.amax(S, axis=1) - K, 0)
        Y = math.exp(-r * T) * payoff

        hat_C_M[i] = np.mean(Y)
        hat_sigma_M[i] = np.std(Y)
            
        if i > 0: converge_rate[i] = abs((hat_C_M[i] - hat_C_M[i-1])/ hat_C_M[i-1])
                
        print("\tM = %d, \tS_mean = %1.2f, \that_C_M = %2.4f, \tconverge = %1.4f" % (M, np.mean(S[0]), hat_C_M[i], converge_rate[i]))

    end_time = time.time()

    print ("\n\tMC Elapsed time : %f" % (end_time - start_time) )
    return hat_C_M[-1]

