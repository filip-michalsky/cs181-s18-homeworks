{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf200
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Implement this class\
class KernelPerceptron(Perceptron):\
    def __init__(self, numsamples):\
        \
        self.numsamples = numsamples\
        self.X = None\
        self.Y = None\
        self.alphas = \{\}\
        self.S = set()\
        self.b = 0\
        \
    def __triv_kernel(self,X1,X2):\
        return np.dot(X1.T,X2)\
        \
    def __rbf_kernel(self,X1,X2):\
        pass # can implement rbf later\
\
    def fit(self, X, Y):\
        self.X = X\
        self.Y = Y\
        \
        # loop through the number of times as there is samples\
        #for i in tqdm(range(self.numsamples)):\
        for i in range(self.numsamples): \
            # pick index of a random samples \
            t = int(np.random.choice(self.X.shape[0],1))\
        \
            (x_t,y_t) = self.X[t,:], self.Y[t]\
            \
            # reset kernel expansion sum over i in S\
            # for alpha_i * K(x_t,x_i)\
            \
            # need to create a sparse matrix here!\
            kernel_expansion = 0\
            \
            # loop through the set of support vector indices\
            # calculate kernel expansion\
            for j in self.S:\
                kernel_expansion +=  self.alphas[j]*self.__triv_kernel(x_t,self.X[j,:])\
            \
            y_hat = kernel_expansion + self.b\
            if y_t*y_hat <= 0:\
                self.S =self.S.union([t]) # S <- S union t\
                self.alphas[t] = y_t #alpha_i <- y_t\
        \
    def predict(self, X):\
        \
        y_pred = 0\
        \
        for key,val in self.alphas.items():\
            y_pred += val*self.__triv_kernel(X,self.X[key,:])\
            \
        return np.sign(y_pred+self.b)}