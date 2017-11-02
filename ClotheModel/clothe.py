import numpy as np
import pandas as pd

clothes = pd.read_excel('data.xlsx')
clothes_matrix = clothes.as_matrix()

# ML

data = clothes_matrix[:, 1:].astype(float)

# We dont have Y, what can we do?
# Time for likes and dislikes

