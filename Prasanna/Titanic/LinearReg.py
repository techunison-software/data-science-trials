import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#from sklearn.cross_validation import train_test_split
import numpy as np


train_df = pd.read_csv('train.csv',index_col=0)
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
#print(train_df.info())

#malesex_train_df = train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Sex', ascending=True)
#print(malesex_train_df)

#StatsModel

#lm1 = smf.ols(formula='Survived ~  Sex+ Name', data=train_df).fit()
lm1 = smf.ols(formula='Survived ~ Sex', data=train_df).fit()
#print(lm1.params)
#print(lm1.conf_int())
#print(lm1.pvalues)
#print(lm1.rsquared)
#print(lm1.summary())

# for dataset in combine:
#     dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

X_train = train_df['Sex']
Y_train = train_df["Survived"]
X_test  = test_df['Sex']

lm2 = LinearRegression()
lm2.fit(X_train, Y_train)
y_pred = lm2.predict(X_test)
final_pred = np.around(y_pred)
acc_log = round(lm2.score(X_train, Y_train) * 100, 2)
print(acc_log)

# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print(acc_log)


#Scikit-learn
# for dataset in combine:
#     dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# feature_cols = ['Sex']
# X = train_df[feature_cols]
# y = train_df.Survived
# lm2 = LinearRegression()
# lm2.fit(X, y)
#print(lm2.intercept_)
#print(lm2.coef_)


