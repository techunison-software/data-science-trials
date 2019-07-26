# Recursive Feature Elimination

from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import pandas as pd

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


