# Recursive Feature Elimination

from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


labelencoder_x = LabelEncoder()
train_df = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/train.csv',index_col=0)
test_df = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/test.csv')
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())
combine = [train_df, test_df]

train_df['Sex'] = labelencoder_x.fit_transform(train_df['Sex'])
test_df['Sex'] = labelencoder_x.fit_transform(test_df['Sex'])
train_df['Embarked'] = labelencoder_x.fit_transform(train_df['Embarked'].astype(str))
test_df['Embarked'] = labelencoder_x.fit_transform(test_df['Embarked'].astype(str))

X = train_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = train_df['Survived']

dataset = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/train.csv')
y_train = dataset.iloc[:, 0].values
X_train = dataset.iloc[:, [0, 2, 5]].values

imp_mean=Imputer(missing_values='NaN', strategy='mean',axis=1 ) #specify axis
imp_mean = imp_mean.fit(X_train[:, [ 0, 1, 2]])
dataset = imp_mean.transform(X_train[:, [ 0, 1, 2]])
# testdata=imp_mean.transform(X_train[:, [ 0, 1, 2]])

X = dataset
Y = y_train

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print('Num Features: ',fit.n_features_) 
print('Selected Features: ', fit.support_)
print('Feature Ranking: ', fit.ranking_)


#Feature Importance
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()



# #Co-relation matrix
corrmat = train_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(11,11))
g=sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()



#Univariate Selection

bestfeatures = SelectKBest(score_func=chi2, k=7)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 
print(featureScores.nlargest(7,'Score')) 




