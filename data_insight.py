import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = 'Data/Csv/'
filename = 'sc_mnisthamn10000.csv'

df = pd.read_csv(data_path + filename)

descriptions = pd.DataFrame(data=None)

column_names = df.columns
for x in column_names:
    current_desc = df[x].describe()
    print(x, current_desc.shape)
