
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_palette('viridis')
sns.set_style('whitegrid')
np.random.seed(0)

train_df = pd.read_csv('E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/train.csv')
test_df = pd.read_csv('E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/test.csv')

# df = pd.DataFrame(train_df)
# dsf=df.aggregate(['sum', 'max']) 
# print(dsf)

# train_df.info()
# train_df.head()

def df_CountPassengers(df, group):
    getCount_Passengers = df.groupby(group).agg({'PassengerId':'count', 'Survived': 'sum'}).rename(columns={'PassengerId': 'PassengersCount'})
    return getCount_Passengers

df = train_df.copy()
train_groupby_sex = df_CountPassengers(df, ['Sex'])
train_groupby_sex
f, (ax) = plt.subplots(1)
f.set_figwidth(16)
f.set_figheight(6)
sns.barplot(x=train_groupby_sex.index, y='Survived', data=train_groupby_sex, ax=ax)
ax.set_title('Passengers Survived Based On Sex')
ax.plot()
plt.show()
