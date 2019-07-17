
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_palette('viridis')
sns.set_style('whitegrid')
np.random.seed(0)
train_df = pd.read_csv('C:/Users/vijaykumar.TECHUNISON/Downloads/INPUT/train.csv')
test_df = pd.read_csv('C:/Users/vijaykumar.TECHUNISON/Downloads/INPUT/test.csv')

df = pd.DataFrame(train_df)
dsf=df.aggregate(['sum', 'max']) 
print(dsf)

train_df.info()
train_df.head()

