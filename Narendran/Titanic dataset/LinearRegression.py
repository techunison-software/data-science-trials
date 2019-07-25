import numpy as np
import pandas as pd
import inspect, os.path
import matplotlib.pyplot as plt
import seaborn as  sns

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

filename = inspect.getframeinfo(inspect.currentframe()).filename
path     = os.path.dirname(os.path.abspath(filename))

# print(os.listdir(path+"/input"))

dataset=pd.read_csv(path+"/input/train.csv")
test_data=pd.read_csv(path+"/input/test.csv")

# print(dataset.head())

y_train = dataset.iloc[:, 1].values
X_train = dataset.iloc[:, [2, 4, 5]].values
X_test = test_data.iloc[:, [1, 3, 4]].values

# print(X_train)
imp_mean = SimpleImputer()
imp_mean = imp_mean.fit(X_train[:, 2:3])
X_train[:, 2:3] = imp_mean.transform(X_train[:, 2:3])

imp_mean = imp_mean.fit(X_test[:, 2:3])
X_test[:, 2:3] = imp_mean.transform(X_test[:, 2:3])


labelencoder_x = LabelEncoder()
X_train[:, 1] = labelencoder_x.fit_transform(X_train[:, 1].astype(str))
# X_train[:, 3] = labelencoder_x.fit_transform(X_train[:, 3].astype(str))

X_test[:, 1] = labelencoder_x.fit_transform(X_test[:, 1].astype(str))
# X_test[:, 3] = labelencoder_x.fit_transform(X_test[:, 3].astype(str))

# # ------------------------------------ Feature selection to Find what Feature contribution are Valid---------------------------------
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV
# from sklearn.datasets import make_classification
# import sklearn as sk
# # Create the RFE object and compute a cross-validated score.
# svc = SVC(kernel="linear",C=0.01) #Penalty parameter C of the error term.
# #nvb = MultinomialNB(alpha=0.5)
# # The "accuracy" scoring is proportional to the number of correct classifications
# rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(3),scoring='accuracy')

# rfecv.fit(X_train, y_train)
# rfopt = rfecv
# n_fest_opt = rfecv.n_features_

# # df_nn = dataset[0]
# rfopt_sup = rfecv.support_
# rfopt_grid = rfecv.grid_scores_

# print("RFOPT - ",rfopt_sup,"\t rfopt_grid - ",rfopt_grid)
# print("\nn_fest_opt - ",n_fest_opt)

# cBest = 0.1
# cScore = 0
# print('best c value:' )
# #set c
# for c in [0.001,0.01,0.1,1,10,100,1000]:
#     svc.C=c
#     rfecv.fit(X_train, y_train)
#     if(rfecv.score(X_train, y_train)>cScore):
#         print(c)
#         cBest = c
#         cScore = rfecv.score(X_train, y_train)

# svc.C = cBest

# # rfecv 10 times
# for k in range(10):
#     X_train = sk.utils.shuffle(X_train)
#     #df_n = df_n.reset_index()
#     #df_n = df_n.drop('index',axis=1)
#     print("\nIn",k)
#     rfecv.fit(X_train, y_train)
    
#     if(rfecv.n_features_<n_fest_opt and rfecv.score(X_train, y_train)>0.7):
#         print("nfeat:")
#         print(rfecv.n_features_)
#         df_nn = X_train
#         rfopt_sup = rfecv.support_
#         rfopt_grid = rfecv.grid_scores_      
#         n_fest_opt = rfecv.n_features_

# #best features 
# colsXopt = []
# mask = rfopt_sup
# print(mask)
# print(mask.shape)

# for i in range(len(mask)):
#     if (mask[i] == True):
#         colsXopt.append(colsX_[i])

# print("Optimal number of features : %d" % n_fest_opt)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
final_pred = np.around(y_pred)


# import statsmodels.api as sm

# X_train = np.append(arr = np.ones((len(X_train), 1)).astype(int), values = X_train, axis = 1)

# X_opt = X_train[:, [0, 1, 2, 3]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# print(regressor_OLS.summary())

# X_opt = X_train[:, [0, 1, 2]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# print("\n\n",regressor_OLS.summary())

# X_opt = X_train[:, [0, 1]]
# regressor_OLS = sm.OLS(endog=y_train, exog=X_opt.astype(float)).fit()
# print("\n\n",regressor_OLS.summary())

df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': final_pred})
df.to_csv(path+"/input/linear_reg_output.csv", index = False)

print('----------------------------------------------END--------------------------------------------------')