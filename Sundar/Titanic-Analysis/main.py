import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("DataSample"))
sns.set_palette('viridis')
sns.set_style('whitegrid')
np.random.seed(42)
train_df = pd.read_csv('DataSample/train.csv')
test_df = pd.read_csv('DataSample/test.csv')
train_df.info()
train_df.head()
print(train_df.head())
def get_df_by_group(df, group):
    df_groupedby = df.groupby(group).agg({'PassengerId':'count', 'Survived': 'sum'}).rename(columns={'PassengerId': 'NumPassengers'})
    df_groupedby['Rate'] = df_groupedby['Survived'] / df_groupedby['NumPassengers'] 
    return df_groupedby

# Result Group By Sex

df = train_df.copy()
train_groupby_sex = get_df_by_group(df, ['Sex'])
# ******************** Group By Sex **********************
print("********************************************************")
print(train_groupby_sex)
print("********************************************************")
# ********************************************************

# Result sample By Sex and Embarked
# f, (ax1, ax2) = plt.subplots(1, 2)
# f.set_figwidth(16)
# f.set_figheight(6)
# sns.barplot(x=train_groupby_sex.index, y='Survived', data=train_groupby_sex, ax=ax1)
# sns.barplot(x=train_groupby_sex.index, y='NumPassengers', data=train_groupby_sex, ax=ax2)
# ax1.set_title('Passengers Survived Per Sex')
# ax2.set_title('Passengers Embarked Per Sex') 
# ax1.plot()
# ax2.plot()


# Result Group By Class

train_groupby_pclass = get_df_by_group(df, ['Pclass'])
# ******************** Group By Class ********************
print("********************************************************")
print(train_groupby_pclass)
print("********************************************************")
# ********************************************************
f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figwidth(16)
f.set_figheight(6)
sns.barplot(x=train_groupby_pclass.index, y='Survived', data=train_groupby_pclass, ax=ax1)
sns.barplot(x=train_groupby_pclass.index, y='NumPassengers', data=train_groupby_pclass, ax=ax2)
ax1.set_title('Passengers Survived Per Class')
ax2.set_title('Passengers Embarked Per Class') 
ax1.plot()
ax2.plot()
plt.show()