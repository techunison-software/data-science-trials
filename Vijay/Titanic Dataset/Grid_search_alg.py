import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR



dataset = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/train.csv')
test_data = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/test.csv')

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

X_test[:, 1] = labelencoder_x.fit_transform(X_test[:, 1].astype(str))


# #Grid Search   

# Random Forest Classifier

grid_param = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2']
}

rfc = RandomForestClassifier(grid_param) 
svc = SVC(grid_param)

#,'kernel':['rbf','poly','linear'],'gamma':['scale','auto'],'probability':[True,False]

gd_sr = GridSearchCV(estimator=svc,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1
                     )
#,SVC(kernel='rbf')
gd_sr.fit(X_train, y_train)
best_parameters = gd_sr.best_params_
print('Best Parameter is : ',best_parameters)
best_result = gd_sr.best_score_
print('Best score is : ', best_result)




# Support Vector Machine

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_
print('The best set of parameter is : ', grid_search.best_params_)
print('The best score is : ', grid_search.best_score_)



# Cross Validation


from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2])
cv = KFold(n_splits=3, random_state=0)

for train_index, test_index in cv.split(X):
      print("TRAIN:", train_index, "TEST:", test_index)


# Shuffle Split

from sklearn.model_selection import ShuffleSplit
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2])
cv = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)

for train_index, test_index in cv.split(X):
...    print("TRAIN:", train_index, "TEST:", test_index)