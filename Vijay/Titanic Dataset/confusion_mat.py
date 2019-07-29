import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/train.csv')
test_data = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/test.csv')
gender_sub=pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/gender_submission.csv')

gender_sub = gender_sub[['PassengerId', 'Survived']]
X_gender_sub = gender_sub.iloc[:, [1]].values

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


##################################################################################################################################

# # Linear Regression

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# final_pred = np.around(y_pred)
# acc_lin = round(regressor.score(X_train, y_train) * 100, 2)
# print('The score of the Linear regression is : ',acc_lin)


# # Logistic Regression

# logreg = LogisticRegression(penalty='l1',C=2.0,solver ='liblinear',max_iter=90)
# logreg.fit(X_train, y_train)
# Y_pred = logreg.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_log = round(logreg.score(X_train, y_train) * 100, 2)
# print('The score of the Logistic regression is : ',acc_log)


# #Support Vector Machines

# from sklearn.svm import SVC
# svc = SVC(probability=True,gamma = 'scale',kernel='rbf')
# svc.fit(X_train, y_train)
# Y_pred = svc.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_svc = round(svc.score(X_train, y_train) * 100, 2)
# print('The score of the Support Vector Machine is : ',acc_svc)


# #K-Nearest Neighbours

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, y_train)
# Y_pred = knn.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_knn = round(knn.score(X_train, y_train) * 100, 2)
# print('The score of K-Nearest Neighbours is : ',acc_knn)

# #Gaussian Naive Bayes

# from sklearn.naive_bayes import GaussianNB
# gaussian = GaussianNB(priors=None, var_smoothing=1e-09)
# gaussian.fit(X_train, y_train)
# Y_pred = gaussian.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
# print('The score of Gaussian Naive Bayes Theorem is : ',acc_gaussian)

# #Perceptron

# from sklearn.linear_model import Perceptron
# perceptron = Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
#       fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
#       penalty=None, random_state=0, shuffle=True, tol=0.001,
#       validation_fraction=0.1, verbose=0, warm_start=False)
# perceptron.fit(X_train, y_train)
# Y_pred = perceptron.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
# print('The score of Perceptron is : ', acc_perceptron)


# #Linear SVM

# from sklearn.svm import LinearSVC
# linear_svc = LinearSVC(loss ='hinge',penalty='l2',C=2.0)
# linear_svc.fit(X_train, y_train)
# Y_pred = linear_svc.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
# print('The score of LinearSVC is : ', acc_linear_svc)


# #Stochastic Gradient Descent

# from sklearn.linear_model import SGDClassifier
# sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
# sgd.fit(X_train, y_train)
# Y_pred = sgd.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
# print('The score of Stochastic Gradient Descent is : ', acc_sgd)

# #Decision Tree

# from sklearn.tree import DecisionTreeClassifier
# decision_tree = DecisionTreeClassifier(criterion = 'entropy',max_features=3,min_impurity_decrease=0,presort =True)
# decision_tree.fit(X_train, y_train)
# Y_pred = decision_tree.predict(X_test)
# final_pred=np.around(Y_pred)
# acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
# print('The score of Decision Tree is : ', acc_decision_tree)


#Random Forest

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,criterion = 'entropy',min_samples_leaf=2,
                                       min_samples_split=3,max_leaf_nodes=3,max_depth =10,min_weight_fraction_leaf=0.5,
                                       max_features=3,min_impurity_decrease=4,n_jobs =1,random_state=2)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
final_pred=np.around(Y_pred)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print('The score of Random Forest is : ',acc_random_forest)





# Results

results = confusion_matrix(X_gender_sub, final_pred) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(X_gender_sub, final_pred)) 
print ('Classification Report : ')
print (classification_report(X_gender_sub, final_pred))



# imp_mean=Imputer(missing_values='NaN', strategy='mean',axis=1 ) #specify axis
# imp_mean = imp_mean.fit(actual[:, [0, 1, 2, 3, 4]])
# actualdataset = imp_mean.transform(actual[:, [0, 1, 2, 3, 4]])

# # imp_means=Imputer(missing_values='NaN', strategy='mean',axis=1 )
# imp_mean = imp_mean.fit(predicted[:, [0, 1, 2]])
# predicteddataset = imp_mean.transform(predicted[:, [0, 1, 2]])


