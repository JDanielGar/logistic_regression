# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LogisticRegression

# def getData(balance_ones=True):
#     # images are 48x48 = 2304 size vectors
#     Y = []
#     X = []
#     first = True
#     for line in open('fer2013.csv'):
#         if first:
#             first = False
#         else:
#             row = line.split(',')
#             Y.append(int(row[0]))
#             X.append([int(p) for p in row[1].split()])

#     X, Y = np.array(X) / 255.0, np.array(Y)

#     if balance_ones:
#         # balance the 1 class
#         X0, Y0 = X[Y!=1, :], Y[Y!=1]
#         X1 = X[Y==1, :]
#         X1 = np.repeat(X1, 9, axis=0)
#         X = np.vstack([X0, X1])
#         Y = np.concatenate((Y0, [1]*len(X1)))

#     return X, Y

# X, Y = getData(False)

data = pd.read_csv('fer2013.csv')
data = data.as_matrix()
data = [data[:, -1:] == 1 or data[:, -1:] == 0]
x = np.array([int(x) for x in data[0, :-1][1].split(' ')])/255.0
y = np.array(data[:, 0])

