
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/train.csv')
test_data = pd.read_csv('E:/DataScienceTrials/Vijay/Titanic Dataset/INPUT/test.csv')

y_train = dataset.iloc[:, 1].values
X_train = dataset.iloc[:, [2, 4, 5, 6]].values
X_test = test_data.iloc[:, [1, 3, 4, 5]].values

imp_mean = Imputer()
imp_mean = imp_mean.fit(X_train[:, 2:4])
X_train[:, 2:4] = imp_mean.transform(X_train[:, 2:4])

imp_mean = imp_mean.fit(X_test[:, 2:4])
X_test[:, 2:4] = imp_mean.transform(X_test[:, 2:4])

labelencoder_x = LabelEncoder()
X_train[:, 1] = labelencoder_x.fit_transform(X_train[:, 1].astype(str))
#X_train[:, 3] = labelencoder_x.fit_transform(X_train[:, 3].astype(str))

X_test[:, 1] = labelencoder_x.fit_transform(X_test[:, 1].astype(str))
#X_test[:, 3] = labelencoder_x.fit_transform(X_test[:, 3].astype(str))

#####################################Start Of Linear Regression#######################################################


# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# final_pred = np.around(y_pred)
# acc_log = round(regressor.score(X_train, y_train) * 100, 2)
# print('The score of the Linear regression is : ',acc_log)


#######################################End Of Linear Regression##########################################################


#####################################Start Of Logistic Regression#######################################################

# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression(penalty='l1',C=2.0,solver ='liblinear',max_iter=90)
# logreg.fit(X_train, y_train)
# Y_pred = logreg.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_log = round(logreg.score(X_train, y_train) * 100, 2)
# print('The score of the Logistic regression is : ',acc_log)


#####################################End Of Logistic Regression#######################################################


#####################################Start Of Support Vector Machines#######################################################

# from sklearn.svm import SVC
# svc = SVC(probability=True,gamma = 'scale',kernel='rbf')
# svc.fit(X_train, y_train)
# Y_pred = svc.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_svc = round(svc.score(X_train, y_train) * 100, 2)
# print('The score of the Support Vector Machine is : ',acc_svc)

#####################################End Of Support Vector Machines#######################################################


#####################################Start Of K-Nearest Neighbours#######################################################

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
final_pred=np.around(Y_pred)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print('The score of K-Nearest Neighbours is : ',acc_knn)


#####################################Start Of K-Nearest Neighbours#######################################################

# To check which all the the useful independent variables
# It is found that PClass, sex and age are useful variables
import statsmodels.formula.api as sm
import statsmodels.api as sm
X_train = np.append(arr = np.ones((len(X_train), 1)).astype(int), values = X_train, axis = 1)

X_opt = X_train[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 1, 2]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 1]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
regressor_OLS.summary()

df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': final_pred})
df.to_csv('E:/DataScienceTrials/Vijay/Titanic Dataset/output.csv', index = False)


