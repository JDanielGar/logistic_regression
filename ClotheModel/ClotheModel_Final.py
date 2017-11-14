
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import linear_model

# Data

data = pd.read_excel('data.xlsx')
data_likes = data.as_matrix()
X = data_likes[:, 1:]
Y = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 0]]).T

# In[5]:

# Train and Test DataSets

Xtrain = X[:7, :]
Ytrain = Y[:7, :]

Xtest = X[-2:]
Ytest = Y[-2:]


model = linear_model.LogisticRegression()
model.fit(Xtrain, Ytrain)
model.get_params()
model.predict(X)

model.score(Xtrain, Ytrain)




