import numpy as np
import pandas as pd
df_training = pd.read_csv(r"C:\Users\aswath.TECHUNISON\Documents\GitHub\data-science-trials\Aswath\DataSets\Titanic\train.csv")
df_training.shape
df_training.dtypes
training_passengerId = df_training.PassengerId

df_training.drop(columns=['PassengerId'],inplace=True)

#dropping Name and Ticket and fare as well out of the data
df_training.drop(columns=['Name','Ticket','Fare'],inplace=True)
df_training.head()
print('Survived value counts: ')
print(df_training.Survived.value_counts())

print('Count by class: ')
print(df_training.Pclass.value_counts())

print('count by sex: ')

print('Cabin or without cabin count')
print('Without cabin', df_training.Cabin.isnull().sum())
print('With cabin', df_training.shape[0] - df_training.Cabin.isnull().sum())

print('Count by Journey Embarking point:')
print(df_training.Embarked.value_counts())
#creating category types
df_training.Survived=df_training.Survived.astype('category')
df_training.Pclass=df_training.Pclass.astype('category')
df_training.Sex=df_training.Sex.astype('category')
df_training.Embarked = df_training.Embarked.astype('category')

# lets do feature engineering using cabin. if a passenger has cabin and if a passenger doesnot have a cabin.
df_training['cabinAllocated'] = df_training.Cabin.apply(lambda x: 0 if type(x)==float else 1)
df_training['cabinAllocated'] = df_training['cabinAllocated'].astype('category')
df_training.dtypes
df_training.drop(columns=['cabinAllocated'],inplace=True)
print("Min Age : {}, Max age : {}".format(df_training.Age.min(),df_training.Age.max()))

print(df_training)
df_training['family'] = df_training.Parch+df_training.SibSp+1
df_training.drop(columns=['SibSp','Parch'],inplace=True)
df_training.head()


#dropping Name and Ticket and fare as well out of the data
df_training.head()
print('Survived value counts: ')
print(df_training.Survived.value_counts())

print('Count by class: ')
print(df_training.Pclass.value_counts())

print('count by sex: ')
print(df_training.Sex.value_counts())

print('Cabin or without cabin count')
print('Without cabin', df_training.Cabin.isnull().sum())
print('With cabin', df_training.shape[0] - df_training.Cabin.isnull().sum())

print('Count by Journey Embarking point:')
print(df_training.Embarked.value_counts())
#creating category types
df_training.Survived=df_training.Survived.astype('category')
df_training.Pclass=df_training.Pclass.astype('category')
df_training.Sex=df_training.Sex.astype('category')
df_training.Embarked = df_training.Embarked.astype('category')

# lets do feature engineering using cabin. if a passenger has cabin and if a passenger doesnot have a cabin.
df_training['cabinAllocated'] = df_training.Cabin.apply(lambda x: 0 if type(x)==float else 1)
df_training['cabinAllocated'] = df_training['cabinAllocated'].astype('category')
df_training.drop(columns=['cabinAllocated'],inplace=True)
print("Min Age : {}, Max age : {}".format(df_training.Age.min(),df_training.Age.max()))
random_list = np.random.randint(df_training.Age.mean() - df_training.Age.std(), 
                                         df_training.Age.mean() + df_training.Age.std(), 
                                         size=df_training.Age.isnull().sum())
df_training['Age'][np.isnan(df_training['Age'])] = random_list
df_training['Age'] = df_training['Age'].astype(int)
df_training['AgeGroup'] = pd.cut(df_training.Age,5,labels=[1,2,3,4,5])
print(df_training)
df_training.drop(columns=['Age'],inplace=True)
df_training['family'] = df_training.Parch+df_training.SibSp+1
df_training.drop(columns=['SibSp','Parch'],inplace=True)
df_training.head()

