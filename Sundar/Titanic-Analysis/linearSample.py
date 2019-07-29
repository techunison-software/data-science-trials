import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
dataset = pd.read_csv('DataSample/train.csv')
test_data = pd.read_csv('DataSample/test.csv')
y_train = dataset.iloc[:, 1].values
X_train = dataset.iloc[:, [2, 4, 5, 9]].values
X_test = test_data.iloc[:, [1, 3, 4, 8]].values

# X_train = dataset.iloc[:, [2, 4, 5, 6]].values
# X_test = test_data.iloc[:, [1, 3, 4, 5]].values
# X_train = dataset.iloc[:, [4, 5, 6]].values
# X_test = test_data.iloc[:, [3, 4, 5]].values
print(X_train)
print(X_test)
imp_mean = Imputer()
imp_mean = imp_mean.fit(X_train[:, 2:4])
imp_median = Imputer()
imp_median = imp_median.fit(X_train[:,2:4])
print(X_train[:, 2:4])
#X_train[:, 2:4] = imp_mean.transform(X_train[:, 2:4])
#imp_mean = imp_mean.fit(X_test[:, 2:4])
X_train[:, 2:4] = imp_median.transform(X_train[:, 2:4])
imp_median = imp_median.fit(X_test[:, 2:4])
X_test[:, 2:4] = imp_median.transform(X_test[:, 2:4])
# imp_mean = imp_mean.fit(X_train[:, 1:3])
# print(X_train[:, 1:3])
# X_train[:, 1:3] = imp_mean.transform(X_train[:, 1:3])
# imp_mean = imp_mean.fit(X_test[:, 1:3])
# X_test[:, 1:3] = imp_mean.transform(X_test[:, 1:3])
# print(X_train)
# print(X_train[:, 1:3])
# # print(X_test)
# print(X_test[:, 1:3])
labelencoder_x = LabelEncoder()
X_train[:, 1] = labelencoder_x.fit_transform(X_train[:, 1].astype(str))
#X_train[:, 3] = labelencoder_x.fit_transform(X_train[:, 3].astype(str))

X_test[:, 1] = labelencoder_x.fit_transform(X_test[:, 1].astype(str))
#X_test[:, 3] = labelencoder_x.fit_transform(X_test[:, 3].astype(str))
#___________________________ Linear Regression_______________________
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
final_pred = np.around(y_pred)
acc_log = round(regressor.score(X_train, y_train) * 100, 2)
print(acc_log)
#_____________________________________________________________________

#___________________________ Logistic Regression_______________________

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l1',C=2.0)
logreg.fit(X_train, y_train)
L_pred = logreg.predict(X_test)
logi_final = np.around(L_pred)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
lf = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': final_pred})
lf.to_csv('logisticSample.csv',index = False)
print('The score of the Logistic regression is : ',acc_log)
#_____________________________________________________________________


#___________________________  Support Vector Machines ___________________________ 
from sklearn.svm import SVC
suporrtVC = SVC()
suporrtVC.fit(X_train, y_train)
Y_pred = suporrtVC.predict(X_test)
acc_svc = round(suporrtVC.score(X_train, y_train) * 100, 2)
acc_svc
sf = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': final_pred})
sf.to_csv('supportvector.csv',index = False)
print('The score of the Logistic regression is : ',acc_svc)
#_____________________________________________________________________
# To check which all the the useful independent variables
# It is found that PClass, sex and age are useful variables
# import statsmodels.formula.api as sm
# import statsmodels.api as sm
# X_train = np.append(arr = np.ones((len(X_train), 1)).astype(int), values = X_train, axis = 1)

# X_opt = X_train[:, [0, 1, 2, 3]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# regressor_OLS.summary()
# print(regressor_OLS.summary())
# X_opt = X_train[:, [0, 1, 2]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# regressor_OLS.summary()

# X_opt = X_train[:, [0, 1]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# regressor_OLS.summary()

df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': final_pred})
df.to_csv('output.csv', index = False)