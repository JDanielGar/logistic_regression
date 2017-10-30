# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:22:47 2017

@author: Daniel
"""
#%%
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

#%%
df = pd.read_csv('dog.csv')
data = df.as_matrix()

#%% 
# Set X and Y

X = data[:, :-1].astype(float)
Y = data[:, -1:]

# Normalize the data
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

# Dimensions of Array

N, D = X.shape

X, Y = shuffle(X, Y)

# Initialize Weights and Bias Term.

W = np.random.rand(D, 1)
b = 0

#%%

def sigmoid(X):
    return 1 / (1 + np.exp(-X))
 
def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

#%%
    
Xtrain = X[:6, :]
Ytrain = Y[:6, :]

Xtest = X[-3:]
Ytest = Y[-3:]

#%%

def classification_rate(Y, P):
    return np.mean(Y == P)

#%%

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

#%%

train_costs = []
test_costs = []
learning_rate = .001

#%%


for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    # Gradient Descent
    
    W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate * (pYtrain - Ytrain).sum()
    
    if(1 % 1000 == 0):
        print(i, ctrain, ctest)
        
print("Final train Classification rate", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification rate", classification_rate(Ytest, np.round(pYtest)))

#%%

def predict(X, W, b):
    print(forward(X, W, b))
    
test = np.array([0, 0, 1, 0])

#%%

predict(test, W, b)

#