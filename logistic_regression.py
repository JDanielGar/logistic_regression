import pandas as pd
import numpy as np

def get_data():
    df = pd.read_csv('ecommerce_data.csv') # Df = Data Frames    
    data = df.as_matrix()
    
    # Split The Data
    X = data[:, :-1]
    Y = data[:, -1]
    
    # Normalize numerical columns
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()
    
    # Categorical Data ( Time 1 | 2 | 3 | 4 )
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]
    
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1
    
    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


X, Y = get_binary_data()