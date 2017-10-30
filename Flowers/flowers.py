#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#%%
df = pd.read_csv("data2.csv")
data = df.as_matrix()

#%%

D = 4;
N = 100;

#%%

# Data Preprocessing

# Normalize Data.

data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
data[:, 1] = (data[:, 1] - data[:, 1].mean()) / data[:, 1].std()
data[:, 2] = (data[:, 2] - data[:, 2].mean()) / data[:, 2].std()



data[data == 'Iris-setosa'] = 0;
data[data == 'Iris-versicolor'] = 1;

X = data[:, :-1].astype(float)
Y = data[:, -1:].astype(float)

X, Y = shuffle(X, Y)
W = np.random.randn(D, 1);
b = 0

#%%

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

#%%
    
def forward(X, W, b):
    z = sigmoid(X.dot(W) + b);
    print(z)
    return z;

def predict(X, W, b):
    forward(X, W, b)


#%%
    
Xtrain = X[:-70]
Ytrain = Y[:-70]

Xtest = X[-40:]
Ytest = Y[-40:]

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


for i in range(200):
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

prediction = X[0, :];