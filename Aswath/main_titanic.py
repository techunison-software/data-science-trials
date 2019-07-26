import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import inspect, os.path
from sklearn.linear_model import LinearRegression

filename = inspect.getframeinfo(inspect.currentframe()).filename
trainpath = os.path.dirname(os.path.abspath(filename))+"/datasets/titanic/train.csv"
testpath = os.path.dirname(os.path.abspath(filename))+"/datasets/titanic/test.csv"
outputpath = os.path.dirname(os.path.abspath(filename))+"/datasets/titanic/output.csv"
dataset = pd.read_csv(trainpath)
test_data = pd.read_csv(testpath)

y_train = dataset.iloc[:, 1].values
X_train = dataset.iloc[:, [2, 4, 5]].values
X_test = test_data.iloc[:, [1, 3, 4]].values

imp_mean = SimpleImputer()
imp_mean = imp_mean.fit(X_train[:, 2:3])

X_train[:, 2:3] = imp_mean.transform(X_train[:, 2:3])

imp_mean = imp_mean.fit(X_test[:, 2:3])
X_test[:, 2:3] = imp_mean.transform(X_test[:, 2:3])

test_data['Age']=X_test[:,2]
labelencoder_x = LabelEncoder()
X_train[:, 1] = labelencoder_x.fit_transform(X_train[:, 1].astype(str))
#X_train[:, 3] = labelencoder_x.fit_transform(X_train[:, 3].astype(str))

X_test[:, 1] = labelencoder_x.fit_transform(X_test[:, 1].astype(str))
#X_test[:, 3] = labelencoder_x.fit_transform(X_test[:, 3].astype(str))
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
final_pred = np.around(y_pred)

#To check which all the the useful independent variables
#It is found that PClass, sex and age are useful variables
from sklearn import linear_model

X_train = np.append(arr = np.ones((len(X_train), 1)).astype(int), values = X_train, axis = 1)
X_opt = X_train[:, [0, 1, 2, 3]]
reg = linear_model.LinearRegression()
reg.fit(y_train.reshape(-1, 1), X_opt)

# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# regressor_OLS.summary()

# X_opt = X_train[:, [0, 1, 2]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# regressor_OLS.summary()

# X_opt = X_train[:, [0, 1]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# regressor_OLS.summary()
test_data = pd.read_csv(testpath)
test_data['Survived']=final_pred
test_data=test_data.drop('Cabin',axis=1)
df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': final_pred})
print("Subset of the predicted outcomes")
print(df.head())
test_data.to_csv(outputpath, index = False)
