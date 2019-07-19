
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

# Uncomment below

# def df_CountPassengers(df, group):
#     getCount_Passengers = df.groupby(group).agg({'PassengerId':'count', 'Survived': 'sum'})#.rename(columns={'PassengerId': 'PassengersCount'})
#     return getCount_Passengers

# df = train_df.copy()
# train_groupby_sex = df_CountPassengers(df, ['Sex'])
# train_groupby_sex
# f, (ax) = plt.subplots(1)
# f.set_figwidth(10)
# f.set_figheight(6)
# sns.barplot(x=train_groupby_sex.index, y='Survived', data=train_groupby_sex, ax=ax)
# ax.set_title('Passengers Survived Based On Sex')
# ax.plot()
# plt.show()


#################################################################################################################################################

#total male and female with survives status


# def df_CountPassengers(df, group):
#     getCount_Passengers = df.groupby(group).agg({'PassengerId':'count', 'Survived': 'sum'})#.rename(columns={'PassengerId': 'PassengersCount'})
#     return getCount_Passengers

# df = train_df.copy()
# train_groupby_sex = df_CountPassengers(df, ['Sex'])
# print(train_groupby_sex)

################################################################################################################################################

# Percentage of Male and Female Survival status

# def df_CountPassengers(df,group):
#     df = train_df.copy()
#     percentage_survival=df.groupby(group).agg({'PassengerId':'count','Survived':'mean'})
#     return percentage_survival
    
# df = train_df.copy()
# answer=df_CountPassengers(df,['Sex'])
# print(answer)
 

################################################################################################################################################

# display sex based on Pclass

# df = pd.read_csv("E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/train.csv", usecols = ['Pclass','Sex'])
# print(df)

################################################################################################################################################

# display survival based on Pclass

# df = pd.read_csv("E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/train.csv", usecols = ['Pclass','Sex'])

# a=1
# b=2
# c=3

# # if a in list(range(1,10)):
# #     # print('Yes')
# #     a=22222222222
# #     print(a)
# # else :
# #     print('No')

# event_dictionary ={a : 'High', b : 'Medium', c : 'Low'} 


# df['Expectancy'] = df['Pclass'].map(event_dictionary) 
# print(df) 

################################################################################################################################################

# #Predictions based on SibSp (Siblings + Spouse)

# df = pd.read_csv("E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/train.csv", usecols = ['Pclass','Name','Sex','SibSp'])

# SibSp_df=pd.read_csv("E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/train.csv", usecols = ['SibSp'])

# x = list(range(0, 2, 1))
# print(x)

# # a=''
# # for entry in SibSp_df:#range(len(SibSp_df)):

    
   
# #     if entry in x:
# #         event_dictionary = {1 : 'Yes'}
# #     elif entry not in x:
# #         event_dictionary = {0 : 'No'}

# event_dictionary = {0 : 'No', 1 : 'Yes', 2 : 'Probable', 3 :'Not much', 4 : 'Improbable', 5 : '' , 6 : ''}
# df['SurvivalExpectancy']=df['SibSp'].map(event_dictionary)
# print(df)

###################################################################################################################################################






