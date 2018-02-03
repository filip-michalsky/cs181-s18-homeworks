#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:39:34 2018

@author: filipmichalsky
"""

# Problem 3 

import csv
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# 1) generate a poly

def gen_poly(X,K):
    
    if X.shape[0]!=1:
        X = X.reshape(1,-1)
    
    # generate arr K x shape 1 of x size
    Y = np.ones((K,X.shape[1]))*X

    # loop through rows and exponentiate them
    for i,j in enumerate(Y):
        Y[i]=np.power(Y[i],i)
        #print(i,Y[i])
        
    #create exponentiation coeffs in eye matrix form
    a_s = np.random.uniform(low=-1,high=1,size=K)
    #print(a_s)
    coeffs = np.eye(K)* a_s
    #coeffs.shape
    
    result = np.dot(coeffs,Y)
    
    # return the polynomial
    return np.sum(result,axis=0)
    
# 2) sample N points x_i

def sample_xi(N,low,high):
    if type(N)!=int:
        if N % np.floor(N)>0.5:
            N = np.ceil(N)
            N=np.int(N)
        else:
            N=np.floor(N)
            N=np.int(N)
    return np.random.uniform(low=low,high=high,size=N).reshape(1,-1)

#test
sample = sample_xi(3,-5,5)
print("Sample x_is: ",sample,"\n")
print("Sample polynomial: ",gen_poly(sample,K=10))

# 3) compute y_i = f(x_i) + err_i 

#returns both the true y_s as well as the variance of the true distribution
    
def get_ys_and_var(poly):

    f_min = min(poly)
    f_max = max(poly)

    err = np.random.normal(0,(f_max-f_min)/10,size=len(poly))

    y_is = poly + err
    
    var = (f_max-f_min)/10
    
    return y_is,var

N = 20
K=10

#get x_is
x_is = sample_xi(N,-5,5)
print("x_is: ",x_is,"\n\n")
#get poly with Ktrue =10
poly = gen_poly(x_is,10)
print("poly: ",poly,"\n\n")
y_s,var = get_ys_and_var(poly)

#print y_s and variance

print("Generated y_is :\n ",y_s)
print("\nConstant variance with K_true = 10 poly: ",var)

########################################################################################
########################################################################################

# Problem 3b

def minimize_chi2(K,y_s,x_is,var):
    
    # get coeffs from polyfit
    coeffs = np.polyfit(y_s.reshape(-1,),x_is.reshape(-1,),K-1)
    #print(coeffs)
    
    ##### create sum of a^j x^j
    if x_is.shape[0]!=1:
        x_is = x_is.reshape(1,-1)
    
    # generate arr K x shape 1 of x size
    Y = np.ones((K,x_is.shape[1]))*x_is

    # loop through rows and exponentiate them
    for i,j in enumerate(Y):
        Y[i]=np.power(Y[i],i)
    
    # get coeffs in eigenmatrix form
    coeffs2 = np.eye(K)*coeffs
    result = np.dot(coeffs2,Y)
    ##########################
    
    #sum for each x_i the polynomial terms to get 1 x len(x) sized array
    sum_over_K = np.sum(result,axis=0)

    chi2 = (1/var)*np.sum(np.power((y_s-sum_over_K),2))
    
    return chi2

print("Sample minimized Chi2: ",minimize_chi2(3,y_s,x_is,var))

# Check that chi^2 is a decreasing function with respect to K

plt.figure(1)
collect_Ks = [minimize_chi2(K,y_s,x_is,var) for K in range(1,6)]
plt.plot([i for i in range(1,6)],collect_Ks,color="red");
plt.xlabel("K")
plt.ylabel("$\chi^{2}_{{min}}$",{'fontsize' : 'large'});
plt.title("Chi^2_min as a function of K");

########################################################################################
########################################################################################

# 3c 
N = 20
K=10
trials = 500

p_xi = 1/10

#get x_is
x_is = sample_xi(N,-5,5)

BIC = lambda K: (N/2)*np.log(2*np.pi*var)-N*np.log(1/10)+0.5*minimize_chi2(K,y_s,x_is,var)

optimal_Ks = np.zeros(trials)

for idx,trial in enumerate(range(trials)):
    
    #generate poly
    poly = gen_poly(x_is,K=10)
    
    # get true y_s and variance
    y_s,var = get_ys_and_var(poly)
    
    #minimize BIC
    final_K = 1
    for i in range(2,10):
        if BIC(i)<BIC(final_K):
            final_K = i
    
    #save K correspodning to minimized BIC
    optimal_Ks[idx]=final_K
    
print("Mean of optimal K for {} trials is: {}".format(trials,np.mean(optimal_Ks)))
print("Variance of optimal K for {} trials is: {:2f}".format(trials,np.var(optimal_Ks)))


########################################################################################
########################################################################################

#Problem 3d

results_mean = {}
results_var = {}

for N in 3*np.logspace(0,4,40):
    K=10
    trials = 500

    p_xi = 1/10

    #get x_is
    x_is = sample_xi(N,-5,5)

    BIC = lambda K: (N/2)*np.log(2*np.pi*var)-N*np.log(1/10)+0.5*minimize_chi2(K,y_s,x_is,var)

    optimal_Ks = np.zeros(trials)

    for idx,trial in enumerate(range(trials)):

        poly = gen_poly(x_is,K=10)
        y_s,var = get_ys_and_var(poly)

        final_K = 1

        for i in range(2,10):
            if BIC(i)<BIC(final_K):
                final_K = i

        optimal_Ks[idx]=final_K

    print("Number of samples {} done.".format(N))
    results_mean[N] = np.mean(optimal_Ks)
    results_var[N] = np.var(optimal_Ks)
    
lists = sorted(results_mean.items()) # sorted by key, return a list of tuples
errors = sorted(results_var.items())

x, y = zip(*lists) # unpack a list of pairs into two tuples
x,y_errs = zip(*errors)

plt.figure(num=2,figsize=(10,10))
plt.semilogx(x, y)
plt.xlabel("Number of samples")
plt.ylabel("Optimal K")
plt.title("Without Variance")
plt.show()

lists = sorted(results_mean.items()) # sorted by key, return a list of tuples
errors = sorted(results_var.items(s))

x, y = zip(*lists) # unpack a list of pairs into two tuples
x,y_errs = zip(*errors)

plt.figure(num=3,figsize=(10,10))
plt.errorbar(x,y,yerr=y_errs)
plt.semilogx(x, y)
plt.xlabel("Number of samples")
plt.ylabel("Optimal K")
plt.title("With Variance")
plt.show()