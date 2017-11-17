# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# An array in Machine Learning have the form [S, D] 
# An array in Algebra have the form [i, j]

# Data Preprocessing 

def GetData():
    X = []
    Y = []
    first = True
    for line in open('fer2013.csv'):
        if(first):
            first = False
        else:  
            inner_data = line.split(',')
            X.append([int(x) for x in inner_data[1].split()])
            Y.append(int(inner_data[0]))
    return np.array(X)/255, np.array(Y)



label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
def ShowImage(X, number):
    plt.imshow(X[number].reshape(48, 48), cmap='gray')
    plt.title(label_map[Y[number]])
    plt.show()

def FilterData(X, Y):
    x = []
    y = []
    for data in range(len(Y)):
        if(Y[data] == 0 or Y[data] ==1):
            x.append(X[data])
            y.append(Y[data])
    return np.array(x), np.array(y)

X, Y = FilterData(GetData()[0], GetData()[1])

# Data split into Training and Test sets.

# Train -> 4125.0 Samples

Xtrain = X[:4125, :]
Ytrain = Y[:4125]

# Test

Xtest = X[1375:, :]
Ytest = Y[1375:]

# Machine Learning

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(w, x, b):
    return sigmoid(W.dot(X) + b)


# model = LogisticRegression()
# model.fit(Xtrain, Ytrain)
