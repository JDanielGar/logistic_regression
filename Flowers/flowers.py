#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("data2.csv")
data = df.as_matrix()

#%%

D = 4;
N = 100;


data[data == 'Iris-setosa'] = 0;
data[data == 'Iris-versicolor'] = 1;

X = data[:, :-1].astype(float)

Y = data[:, -1:].astype(float)
W = np.random.randn(D, 1);

#%%

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

#%%
    
def forward(X, W):
    z = sigmoid(X.dot(W));
    print(z)
    return z;


#%%
    
# Probe ur self.
    
forward(X, W)
