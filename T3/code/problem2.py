# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        
        self.numsamples = numsamples
        self.X = None
        self.Y = None
        self.alphas = {}
        self.S = set()
        self.b = 0
        
    def __triv_kernel(self,X1,X2):
        return np.dot(X1.T,X2)
        
    def __rbf_kernel(self,X1,X2):
        pass # can implement rbf later

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        
        # loop through the number of times as there is samples
        #for i in tqdm(range(self.numsamples)):
        for i in range(self.numsamples): 
            # pick index of a random samples 
            t = int(np.random.choice(self.X.shape[0],1))
        
            (x_t,y_t) = self.X[t,:], self.Y[t]
            
            # reset kernel expansion sum over i in S
            # for alpha_i * K(x_t,x_i)
            
            # need to create a sparse matrix here!
            kernel_expansion = 0
            
            # loop through the set of support vector indices
            # calculate kernel expansion
            for j in self.S:
                kernel_expansion +=  self.alphas[j]*self.__triv_kernel(x_t,self.X[j,:])
            
            y_hat = kernel_expansion + self.b
            if y_t*y_hat <= 0:
                self.S =self.S.union([t]) # S <- S union t
                self.alphas[t] = y_t #alpha_i <- y_t
        
    def predict(self, X):
        
        def check_shape(X):
            try:
                shape = (X.shape[0], X.shape[1])
            except IndexError:
                shape = (1, X.shape[0])
            return shape
    
        if check_shape(X)[0]==1:
            y_pred = 0

            for key,val in self.alphas.items():
                y_pred += val*self.__triv_kernel(X,self.X[key,:])
                
            return np.sign(y_pred+self.b)
        
        else:
            y_hats = []

            for point in X:
                
                y_pred = 0
                for key,val in self.alphas.items():
                    y_pred += val*self.__triv_kernel(point,self.X[key,:])

                y_hats.append(np.sign(y_pred+self.b))

            return np.array(y_hats)
    
    def score(self,X_test,Y_test):
        
        test_length = X_test.shape[0]
        
        
        correct_cnt = 0
        # so, if we predict correctly, count it
        
        for idx,point in enumerate(X_test):
            if np.float(Y_test[idx]) ==np.float(self.predict(point)):
                correct_cnt+=1
            
        return correct_cnt/np.float(test_length)

# Implement this class
# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        
        self.numsamples = numsamples
        self.X = None
        self.Y = None
        self.alphas = {}
        self.S = set()
        self.b = 0
        self.beta = beta
        self.N = N
        
    def __triv_kernel(self,X1,X2):
        return np.dot(X1.T,X2)
        
    def __rbf_kernel(self,X1,X2):
        pass # can implement rbf later
    
    def __get_kernel_expansion(self,x_t):
        # calculate kernel expansion
        
        # reset kernel expansion sum over i in S
        # for alpha_i * K(x_t,x_i)

        # need to create a sparse matrix here!
        kernel_expansion = 0

        # loop through the set of support vector indices
        # calculate kernel expansion
        for j in self.S:
            kernel_expansion +=  self.alphas[j]*self.__triv_kernel(x_t,self.X[j,:])
        
        return kernel_expansion
                
                
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        
        # loop through the number of times as there is samples
        #for i in tqdm(range(self.numsamples)):
        
        for i in range(self.numsamples): 
            # pick index of a random samples 
            t = int(np.random.choice(self.X.shape[0],1))
        
            (x_t,y_t) = self.X[t,:], self.Y[t]
            
            y_hat = self.__get_kernel_expansion(x_t) + self.b
            
            if y_t*y_hat <= self.beta:
                self.S =self.S.union([t]) # S <- S union t
                self.alphas[t] = y_t #alpha_i <- y_t
                if len(self.S) > self.N:
                    max_val = -1000000000000000 # initialize -inf
                    max_idx = None
                    for k in self.S:
                        y_k_hat = self.__get_kernel_expansion(self.X[k,:]) # get y_hat for each support vector index
                        #now, loop and locate the support vector index with the max error 
                        curr_val = self.Y[k]*(y_k_hat-self.alphas[k]*self.__triv_kernel(self.X[k,:],self.X[k,:]))
                        if max_val < curr_val:
                            max_idx = k
                            max_val = curr_val
                        # remove the largest error from the support vector indices
                        self.S = self.S.difference(set([k]))
        
    def predict(self, X):
        
        def check_shape(X):
            try:
                shape = (X.shape[0], X.shape[1])
            except IndexError:
                shape = (1, X.shape[0])
            return shape
    
        if check_shape(X)[0]==1:
            y_pred = 0

            for key,val in self.alphas.items():
                y_pred += val*self.__triv_kernel(X,self.X[key,:])
                
            return np.sign(y_pred+self.b)
        
        else:
            y_hats = []

            for point in X:
                
                y_pred = 0
                for key,val in self.alphas.items():
                    y_pred += val*self.__triv_kernel(point,self.X[key,:])

                y_hats.append(np.sign(y_pred+self.b))

            return np.array(y_hats)
    
    def score(self,X_test,Y_test):
        # scores the functions predictive rate against a test set
        test_length = X_test.shape[0]
        
        
        correct_cnt = 0
        # so, if we predict correctly, count it
        
        for idx,point in enumerate(X_test):
            if np.float(Y_test[idx]) ==np.float(self.predict(point)):
                correct_cnt+=1
            
        return correct_cnt/np.float(test_length)



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# get validation set
validation = np.loadtxt("val.csv", delimiter=',')
X_test = validation[:, :2]
Y_test = validation[:, 2]

# test perceptron
test_perceptron = KernelPerceptron(400)
test_perceptron.fit(X,Y)

print("validation score: ",test_perceptron.score(X_test,Y_test))

budget_perceptron = BudgetKernelPerceptron(beta=5,N=300,numsamples=30000)
budget_perceptron.fit(X,Y)

print('budget perceptron score: ',budget_perceptron.score(X_test,Y_test))

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.

# NOTE - I play with these below, but the calculations take about 6 hours on a single CPU
beta = 0
N = 100
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=False, include_points=True)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=False, include_points=True)

#compare to naive SVM

from sklearn import svm

naive_svm = svm.LinearSVC()
naive_svm.fit(X,Y)

print('score with naive SVM from sklearn: ',naive_svm.score(X_test,Y_test))


# careful! the following code runs for about 7 hours on a single CPU 2400Mhz
# with complexity O(n^4)

speed_conver_dict_kernel = {}

speed_conver_dict_budget = {}

import time

#switch flag to true to get the play with params beta,N, and numsamples

run_playful_params = False
if run_playful_params:
	betas = [0,0.1,0.2,0.3,0.4,0.5,1,2,3,4,5]
	numsamples = [100,200,300,800,1000,5000,10000,150000,20000]
	N = [20,50,100,150,200,300]

	for num in tqdm(numsamples):
	    print("-"*50)
	    print('Random Samples: ',num)
	    # Get speed and accuracy of Kernel Perceptron for varios numsamples randomly drawn
	    t0 = time.time()
	    avg_time = []
	    avg_score = []
	    for i in range(6):
	        
	        test_perceptron = KernelPerceptron(num)
	        test_perceptron.fit(X,Y)
	        # Capture fitting time
	        t1 = time.time() - t0
	        avg_time.append(t1)
	        avg_score.append(test_perceptron.score(X_test,Y_test))
	    #Save speed and accuracy of the config (both mean and std)
	    speed_conver_dict_kernel[num] = (np.mean(np.array(avg_time)),np.std(np.array(avg_time)),np.std(np.array(avg_score)),np.mean(np.array(avg_score)))
	    
	    # now look at BudgetKernelPeceptron
	    for beta in betas:
	        
	        print('looping through beta =',beta,'...')
	        for number in N:
	            avg_time1 = []
	            avg_score1 = []
	            for i in range(6):
	                t0_prime = time.time()
	                budget_perceptron_test = BudgetKernelPerceptron(beta=beta,N=number,numsamples=num)
	                budget_perceptron_test.fit(X,Y)
	                t1_prime = time.time() - t0_prime
	                avg_time1.append(avg_score1)

	                # gather averages for the accuracy
	                avg_score1.append(budget_perceptron_test.score(X_test,Y_test))
	            speed_conver_dict_budget[num,beta,number] = (np.mean(np.array(avg_time1)),np.std(np.array(avg_time1)),np.std(np.array(avg_score1)),np.mean(np.array(avg_score1)))
	        

