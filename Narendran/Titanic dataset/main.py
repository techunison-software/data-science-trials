import numpy as np
import pandas as pd
import inspect, os.path
import matplotlib.pyplot as plt
import seaborn as  sns
import re
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

filename = inspect.getframeinfo(inspect.currentframe()).filename
path     = os.path.dirname(os.path.abspath(filename))

# print(os.listdir(path+"/input"))

train_df=pd.read_csv(path+"/input/train.csv")
test_df=pd.read_csv(path+"/input/test.csv")

datasets=[train_df,test_df]
dataset = datasets.copy()

# print(train_df.columns.values)
# print(test_df.columns.values)

# print(train_df.head(20))

# ========================================== Finding Missing Data =============================================
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

# print(missing_values_table(test_df))

# ======================================== Finding Correlation between Attributes using PLOTS ======================================

# # ----------------------------------------Barchart Representation of Survived by Pclass-------------------------------------
# sns.barplot(x='Pclass', y='Survived', data=train_df)

# # ----------------------------------------Histogram Representation of Survived by Age-------------------------------------
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# # -----------------------------------Common Function to get Grouping sum value by category---------------------- 
# def get_df_by_group(df, group):
#     df_groupedby = df.groupby(group).agg({'PassengerId':'count', 'Survived': 'sum'}).rename(columns={'PassengerId': 'NumPassengers'})
#     df_groupedby['Rate'] = df_groupedby['Survived'] / df_groupedby['NumPassengers'] 
#     return df_groupedby

# train_groupby_sex=get_df_by_group(train_df,['Sex'])
# print(train_groupby_sex)

# # ----------------------------------------Barchart Representation Survived by sex-------------------------------------
# f, (ax1, ax2) = plt.subplots(1, 2)
# f.set_figwidth(12)
# f.set_figheight(6)
# sns.barplot(x=train_groupby_sex.index, y='Survived', data=train_groupby_sex, ax=ax1)
# sns.barplot(x=train_groupby_sex.index, y='NumPassengers', data=train_groupby_sex, ax=ax2)
# ax1.set_title('Passengers Survived Per Sex')
# ax2.set_title('Passengers Embarked Per Sex') 
# ax1.plot()
# ax2.plot()

# train_groupby_pclass=get_df_by_group(train_df,['Pclass'])
# print(train_groupby_pclass)

# # ----------------------------------------Barchart Representation of Survived by Pclass-------------------------------------
# f, (ax1, ax2) = plt.subplots(1, 2)
# f.set_figwidth(12)
# f.set_figheight(6)
# sns.barplot(x=train_groupby_pclass.index, y='Survived', data=train_groupby_pclass, ax=ax1)
# sns.barplot(x=train_groupby_pclass.index, y='NumPassengers', data=train_groupby_pclass, ax=ax2)
# ax1.set_title('Passengers Survived Per Class')
# ax2.set_title('Passengers Embarked Per Class') 
# ax1.plot()
# ax2.plot()

# train_groupby_age=get_df_by_group(train_df,['Age'])
# # print(train_groupby_age)

# # ----------------------------------------Barchart Representation of Survived by Age-------------------------------------
# f, (ax1, ax2) = plt.subplots(1, 2)
# f.set_figwidth(20)
# f.set_figheight(10)

# p=sns.barplot(x=train_groupby_age.index, y='Survived', data=train_groupby_age, ax=ax1)
# q=sns.barplot(x=train_groupby_age.index, y='NumPassengers', data=train_groupby_age, ax=ax2)
# plt.setp(p.get_xticklabels(), rotation=90)
# plt.setp(q.get_xticklabels(), rotation=90)
# ax1.set_title('Passengers Survived Per Age')
# ax2.set_title('Passengers Embarked Per Age') 
# ax1.plot()
# ax2.plot()

# # ----------------------------------------displot Representation Survived by sex and AGE-------------------------------------
# survived = 'survived'
# not_survived = 'not survived'
# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
# women = train_df[train_df['Sex']=='female']
# men = train_df[train_df['Sex']=='male']
# ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
# ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
# ax.legend()
# ax.set_title('Female')
# ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
# ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
# ax.legend()
# _ = ax.set_title('Male') 

# # ----------------------------------------Histogram Representation of Pclass by AGE Survived-------------------------------------
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();

# # ----------------------------------------Histogram Representation of Embarked vs Pclass,Survivied,sex -------------------------------------
# grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()

# # ----------------------------------------Histogram Representation of Embarked ,survived on sex and fare -------------------------------------
# grid = sns.FacetGrid(train_df, row='Embarked',  height=2.2, aspect=1.6)
# grid.map(sns.barplot,'Sex', 'Fare', 'Survived', alpha=.5, ci=None,palette='deep')
# grid.add_legend()

plt.show()

# ============================================= Data cleaning and formatting ===========================================================

deck={"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

for data in dataset:
    # --------------------For Identifing Deck pasange belog to using Cabin no------------
    data['Cabin']=data['Cabin'].fillna('U0')
    data["Deck"]= data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data["Deck"]=data["Deck"].map(deck)
    data['Deck'] = data['Deck'].fillna(0)
    data["Deck"]=data["Deck"].astype(int)
    # --------------------For fulling null Age using mean and standard value------------
    mean=data["Age"].mean()
    std=data["Age"].std()
    is_null=data["Age"].isnull().sum()
    rand_age=np.random.randint(mean-std,mean+std,size=is_null)

    age_slice=data["Age"].copy()
    age_slice[np.isnan(age_slice)]=rand_age
    data["Age"]=age_slice
    data["Age"]=data["Age"].astype(int)
    # --------------------For fulling null Embarked with mode value of attribute------------
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

genders = {"male": 0, "female": 1}
embark={"S":0,"C":1,"Q":2}

#  -------------------------------------------- To get segament Seperated value --------------------------- 
# print(pd.cut(dataset[0]["Age"],8).unique())

for data in dataset:
    data["Sex"]=data["Sex"].map(genders)
    data["Embarked"]=data["Embarked"].map(embark)

    data['Age'] = data['Age'].astype(int)
    data.loc[ data['Age'] <= 10, 'Age'] = 0
    data.loc[(data['Age'] > 10) & (data['Age'] <= 20), 'Age'] = 1
    data.loc[(data['Age'] > 20) & (data['Age'] <= 30), 'Age'] = 2
    data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'Age'] = 3
    data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'Age'] = 4
    data.loc[(data['Age'] > 50) & (data['Age'] <= 60), 'Age'] = 5
    data.loc[(data['Age'] > 60) & (data['Age'] <= 70), 'Age'] = 6
    data.loc[ data['Age'] > 70, 'Age'] = 6
    
dataset[0] = dataset[0].drop(['Ticket','Fare','PassengerId','Cabin','Name'], axis=1)
dataset[1] = dataset[1].drop(['Ticket','Fare','PassengerId','Cabin','Name'], axis=1)

# print(dataset[0].head(10))
# print("\n\n",dataset[1].head(10))

Y_train = dataset[0]["Survived"]
X_train = dataset[0].drop("Survived",axis=1)
X_test = dataset[1].copy()


# ============================================== Feature selection to Find what Feature contribution are Valid========================================

# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV
# from sklearn.datasets import make_classification
# from sklearn.naive_bayes import MultinomialNB

# # Create the RFE object and compute a cross-validated score.
# est = SVC(C= 0.7,gamma=.07,  kernel= 'linear') #Penalty parameter C of the error term.
# # est=LogisticRegression(solver='lbfgs')
# # est = MultinomialNB(alpha=0.5)
# # The "accuracy" scoring is proportional to the number of correct classifications
# rfecv = RFECV(estimator=est, step=1, cv=StratifiedKFold(3),scoring='accuracy')

# rfecv.fit(X_train,Y_train)
# rfopt = rfecv

# n_fest_opt = rfecv.n_features_
# rfopt_sup = rfecv.support_
# rfopt_rank = rfecv.ranking_
# rfopt_grid = rfecv.grid_scores_

# print("rfopt_sup - ",rfopt_sup,"\nrfopt_grid - ",rfopt_grid,"\nrfopt_rank - ",rfopt_rank)
# print("\nn_fest_opt - ",n_fest_opt)

# feature_plt=pd.Series(rfopt_grid,X_train.columns)
# feature_plt.plot('barh')
# plt.show()

# ================================================ Grid Search Common Function to Find Best Param Given Model and Params====================================

# ------------------------------------------------- Common Function to Find Best Param Given Model and Param -----------------------------------------------------
def grid_search_fn(model,param_grid):  
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)

    print("\n-----------------------------------------------------------\n Grid Search :",model)
    print("  Best param :",grid_search.best_params_)
    print("  Best score :",grid_search.best_score_,'\n')

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean* 100, std * 2, params))
    print("\n-----------------------------------------------------------\n")

# # ------------------------------------ Grid Search to Find Best Param for Random Forest Classifier---------------------------------
# param_grid = {"max_depth": [3, None],
#               "min_samples_split": [2, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}
# grid_search_fn(RandomForestClassifier(),param_grid)

#------------------------------------ Grid Search to Find Best Param for SVC---------------------------------
# param_grid = [
#     {'kernel':['rbf'], 'C':[0.7,0.8,1], 'gamma':[0.07, 0.08, 0.09]},
#     {'kernel':['linear'], 'C':[0.7,0.8,1], 'gamma':[0.07, 0.08, 0.09]},
#     {'kernel':['poly'],'C':[0.1,1,10,100], 'gamma':['auto']}
#   ]
# grid_search_fn(SVC(),param_grid)

# # ------------------------------------ Grid Search to Find Best Param for KNeighborsClassifier---------------------------------
# param_grid = {
#     'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# }
# grid_search_fn(KNeighborsClassifier(),param_grid)

# ================================================== Applying MODELS ======================================================

# # ------------------------------------Applying Simple Linear Regression Model for Prediction ---------------------------------
# linear_model=LinearRegression(fit_intercept=True)
# linear_model.fit(X_train,Y_train)
# Y_pred = linear_model.predict(X_test)
# acc_linear_regression = round(linear_model.score(X_train, Y_train) * 100, 2)
# print('Linear_Regression - ',acc_linear_regression)

# # ------------------------------------Applying KNN Model for Prediction ---------------------------------
# knn = KNeighborsClassifier(n_neighbors = 3)                                             # Based on n_neighbors value scores with best fit changes
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)

# df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': Y_pred})     # Writing predicted values to KNN_output.csv output file 
# df.to_csv(path+"/output/KNN_output.csv", index = False)

# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# print("KNN - ",acc_knn)

# # ------------------------------------Applying SVC Model for Prediction ---------------------------------
# svc = SVC(C= 1, gamma= 0.09, kernel= 'rbf')                                             # Based on Gamma value score changes {'auto','scale'}
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)

# df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': Y_pred})      # Writing predicted values to SVC_output.csv output file 
# df.to_csv(path+"/output/SVC_output.csv", index = False)

# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print('SVC - ',acc_svc)

# ------------------------------------Applying LogisticRegression Model for Prediction ---------------------------------
# logreg = LogisticRegression(solver='lbfgs')
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)

# df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': Y_pred})      # Writing predicted values to Logistic_Regression_output.csv output file 
# df.to_csv(path+"/output/Logistic_regression_output.csv", index = False)

# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print('Logistic Regression - ',acc_log)

# ================================================== Applying confusion_matrix to Results ======================================================

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score 
# from sklearn.metrics import classification_report 

# cm=confusion_matrix(Y_test, Y_pred)
# plt.imshow(cm, cmap='binary')
# plt.show()

# print ('Accuracy Score : ',accuracy_score(actual, predicted)) 
# print ('Report : ')
# print classification_report(actual, predicted) 

print("-----------------------------------End------------------------------")

