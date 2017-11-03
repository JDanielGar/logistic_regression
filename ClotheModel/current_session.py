# coding: utf-8

import matplotlib as plt
import numpy as np
import pandas as pd
    
data = pd.read_excel('data.xlsx')
data_likes = data.as_matrix()

# data[left:right, bottom:top] data[delete columns, delete rows]

# Samples and Dimension

N, D = data_likes.shape

# To Text

data_likes[:, 1][data_likes[:, 1] == 0] = 'Sin mangas'
data_likes[:, 1][data_likes[:, 1] == 1] = 'Corta'
data_likes[:, 1][data_likes[:, 1] == 2] = 'Larga'

# get_ipython().magic('save previous_session ~1/') # 1 = Ipython
# get_ipython().magic('save current_session ~0/') # 2 = Python

data_likes[:, 2][data_likes[:, 2] == 0] = 'Polo'
data_likes[:, 2][data_likes[:, 2] == 1] = 'Scoop'
data_likes[:, 2][data_likes[:, 2] == 2] = 'V-Neck'


# Here come the LIKES

# likes = np.zeros([D, 1])
likes = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0])

# likes = []

def getLikes():
    like = [];
    for samples in range(0, N):
        print("Nombre de prenda", data_likes[samples, 0], "\n Caracteristicas: \n  1)Manga:", data_likes[samples, 1])
        print("  2)Cuello:", data_likes[samples, 2], "\n  3)Bolsillos:", data_likes[samples, 3], "\n  4)Botones:", data_likes[samples, 4])
        print("  5)Imagen:", data_likes[samples, 5])
        likes[samples, :] = int(input("¿Te gusta?. Si(1) | No(2) -> "))

# Here Come the predictions

data_matrix = data.as_matrix()[:, 1:]

# 1) Normalize Columns

data_matrix[:, 0] = (data_matrix[:, 0] - data_matrix[:, 0].mean()) / data_matrix[:, 0].std()
data_matrix[:, 1] = (data_matrix[:, 1] - data_matrix[:, 1].mean()) / data_matrix[:, 1].std()

# 2) Stardard Varibles

X = data_matrix.astype(float)
Y = likes.astype(float)
W = np.random.rand(D-1, 1)
b = 0

# 3) Define Train and Test

Xtrain = X[:6, :]
Ytrain = Y[:6]

Xtest = X[-3:, :]
Ytest = Y[-3]

# 4) Define forward functions

def sigmoid(x): # Sigmoid function to give a importance to the data
    return 1/(1+np.exp(-x))
    
def forward(w, x, b): # Do the prediction
    return sigmoid(x.dot(w) + b)

# 5) Define predict function and costs

train_costs = []
test_costs = []

def predict_weight(w, x, b, learn_rate = 0.01, iterations=1000):
    for i in range(iterations):
        pYtrain = forward(w, Xtrain, b)
        pYtest = forward(w, Xtest, b)

        ctrain = cross_entropy(Ytrain, pYtrain)
        ctest = cross_entropy(Ytest, pYtest)

        train_costs.append(ctrain)
        test_costs.append(ctest)
        
        pYtrain = pYtrain.T
        print(pYtrain)
        print(Ytrain)
        print(pYtrain - Ytrain)
        
        gradientDescent(Xtrain, Ytrain, pYtrain, learn_rate);

        if(1 % 1000 == 0):
            print(i, ctrain, ctest)

    print("Final train Classification rate", classification_rate(Ytrain, np.round(pYtrain)))
    print("Final test classification rate", classification_rate(Ytest, np.round(pYtest)))


# 6) Classification for seeing the results

def classification_rate(Y, P):
    return np.mean(Y == P)


def gradientDescent(Xtrain, Ytrain, pYtrain, learn_rate):
    global W, b
    W -= learn_rate * Xtrain.T.dot((pYtrain - Ytrain).T)
    b -= learn_rate * (pYtrain - Ytrain).sum()
    

def cross_entropy(t, y): # Targets and predictions
    return t * np.log(y) + (1-t) * np.log(1-y)
