# Control Variate MC for arithmetic Asian options;
# Geometric avg Asian is used as the control
import time, sys, math 
import numpy as np
import scipy.stats as scp
# import random as rand

def geo_asian(S0, K, r, sigma, T, m):
    sigsqT = sigma*sigma*T*(2*m+1)/(6*m+6)
    muT = 0.5*sigsqT + 0.5*(r - 0.5*sigma*sigma)*T
    d1 = (math.log(S0/K) + (muT + 0.5*sigsqT))/(math.sqrt(sigsqT))
    d2 = d1 - math.sqrt(sigsqT)
    geo_call = math.exp(-r*T)*( S0*math.exp(muT)*scp.norm.cdf(d1)-K*scp.norm.cdf(d2))    
    return geo_call

S0 = 35
K = 20
r = 0.05
sigma = 0.25
T = 1
N = int(100)
dt = T/N

Smin = 0
Smax = S0*2
# S_list, dS = np.linspace(Smin, Smax, N+1, retstep=True)
M_list = np.arange(1,16)
converge_rate = np.zeros(len(M_list))
hat_C_M_cv = np.zeros(len(M_list))
hat_sigma_M_cv = np.zeros(len(M_list))

for i in range(len(M_list)):
    M = int(math.pow(2, M_list[i]))
    Z = np.random.normal(0, 1, size=(N,M))
    S = np.zeros(M)
    S_avg = np.zeros(M)
    S_avg_geo = np.zeros(M)

    start_time = time.time()

    S = np.array(S0*np.ones(M))
    S_avg = S0/(N+1)*np.ones(M)
    S_avg_geo = math.pow(S0,1/(N+1))*np.ones(M)

    for n in range(N):
        S = S * ( 1 + r * dt + sigma * math.sqrt(dt) * Z[n]);  
        S[np.where(S<0)] = 0
         
        S_avg = S_avg + S / (N+1)
        S_avg_geo = S_avg_geo * np.power(S, 1/(N+1)) # time-complex

    payoff = S - K
    payoff[np.where(payoff<0)] = 0
    Y = math.exp(-r * T) * payoff;
    payoff_star = S_avg_geo - K
    payoff_star[np.where(payoff_star<0)] = 0
    Y_star = math.exp(-r * T) * payoff_star;
    C_star = geo_asian(S0, K, r, sigma, T, N+1); #Geometric avg as control

    cov_mat = np.cov(Y, Y_star);
    beta = cov_mat[1][0]/(np.var(Y_star))
    Y_cv = Y + beta*(C_star - Y_star)

    hat_C_M_cv[i] = np.mean(Y_cv)
    hat_sigma_M_cv[i] = np.std(Y_cv)

    if i > 0: converge_rate[i] = abs(hat_C_M_cv[i] - hat_C_M_cv[i-1])/hat_C_M_cv[i-1]
    
    if (i % 1 == 0):
        print("M = %d, \t hat_C_M_cv = %2.4f, \t converge_rate = %1.4f" % (M, hat_C_M_cv[i], converge_rate[i]))

end_time = time.time()

print ("\n\tElapsed time : %f" % (end_time - start_time) )	


