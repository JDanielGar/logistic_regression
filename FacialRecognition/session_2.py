# coding: utf-8
get_ipython().magic('save facial_reg 0')
get_ipython().magic('save facial_reg 1')
get_ipython().magic('save facial_reg')
get_ipython().magic('edit')
X, Y = getData(balance_ones=False)
import numpy as no
import numpy as np
X, Y = getData(balance_ones=False)
X
Y
import pandas as pd
data = pd.read_csv('fer2013.csv')
data
data = data.as_matrix()
data
a = data[0, :]
a
a = data[0, :-1]
a
get_ipython().magic('save')
get_ipython().magic('save session_2')
get_ipython().magic('quickref')
get_ipython().magic('save session_2 ~0/')
