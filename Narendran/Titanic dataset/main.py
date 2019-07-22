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

filename = inspect.getframeinfo(inspect.currentframe()).filename
path     = os.path.dirname(os.path.abspath(filename))

# print(os.listdir(path+"/input"))

train_df=pd.read_csv(path+"/input/train.csv")
test_df=pd.read_csv(path+"/input/test.csv")
datasets=[train_df,test_df]
# print(train_df.head(5))

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

def get_df_by_group(df, group):
    df_groupedby = df.groupby(group).agg({'PassengerId':'count', 'Survived': 'sum'}).rename(columns={'PassengerId': 'NumPassengers'})
    df_groupedby['Rate'] = df_groupedby['Survived'] / df_groupedby['NumPassengers'] 
    return df_groupedby

dataset = datasets.copy()

deck={"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

for data in dataset:
    data['Cabin']=data['Cabin'].fillna('U0')
    data["Deck"]= data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data["Deck"]=data["Deck"].map(deck)
    data['Deck'] = data['Deck'].fillna(0)
    data["Deck"]=data["Deck"].astype(int)

# print(train_df.columns.values)
# print(test_df.columns.values)

for data in dataset:
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

for data in dataset:
    data["Sex"]=data["Sex"].map(genders)
    data["Embarked"]=data["Embarked"].map(embark)

    data['Age'] = data['Age'].astype(int)
    data.loc[ data['Age'] <= 11, 'Age'] = 0
    data.loc[(data['Age'] > 11) & (data['Age'] <= 18), 'Age'] = 1
    data.loc[(data['Age'] > 18) & (data['Age'] <= 22), 'Age'] = 2
    data.loc[(data['Age'] > 22) & (data['Age'] <= 27), 'Age'] = 3
    data.loc[(data['Age'] > 27) & (data['Age'] <= 33), 'Age'] = 4
    data.loc[(data['Age'] > 33) & (data['Age'] <= 40), 'Age'] = 5
    data.loc[(data['Age'] > 40) & (data['Age'] <= 66), 'Age'] = 6
    data.loc[ data['Age'] > 66, 'Age'] = 6
    
dataset[0] = dataset[0].drop(['Ticket','Fare','PassengerId','Cabin','Name'], axis=1)
dataset[1] = dataset[1].drop(['Ticket','Fare','PassengerId','Cabin','Name'], axis=1)

# print(dataset[0].head())
# print(dataset[1].head())

Y_train = dataset[0]["Survived"]
X_train = dataset[0].drop("Survived",axis=1)
X_test = dataset[1].copy()

# ------------------------------------Applying Simple Linear Regression Model for Prediction ---------------------------------
linear_model=LinearRegression(fit_intercept=True)
linear_model.fit(X_train,Y_train)
Y_pred = linear_model.predict(X_test)
acc_linear_regression = round(linear_model.score(X_train, Y_train) * 100, 2)
print('Linear_Regression - ',acc_linear_regression)

# ------------------------------------Applying KNN Model for Prediction ---------------------------------
knn = KNeighborsClassifier(n_neighbors = 3)                                             # Based on n_neighbors value scores with best fit changes
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': Y_pred})     # Writing predicted values to KNN_output.csv output file 
df.to_csv(path+"/input/KNN_output.csv", index = False)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN - ",acc_knn)

# ------------------------------------Applying SVC Model for Prediction ---------------------------------
svc = SVC(C= 1, gamma= 0.09, kernel= 'rbf')                                             # Based on Gamma value score changes {'auto','scale'}
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': Y_pred})      # Writing predicted values to SVC_output.csv output file 
df.to_csv(path+"/input/SVC_output.csv", index = False)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print('SVC - ',acc_svc)

# ------------------------------------ Grid Search Common Function to Find Best Param Given Model and Params---------------------------------
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

# ------------------------------------ Grid Search to Find Best Param for SVC---------------------------------
# param_grid = [
#     {'kernel':['rbf'], 'C':[0.7,0.8,1], 'gamma':[0.07, 0.08, 0.09]},
#     {'kernel':['poly'],'C':[0.1,1,10,100], 'gamma':['auto']}
#   ]
# grid_search_fn(SVC(),param_grid)

# # ------------------------------------ Grid Search to Find Best Param for KNeighborsClassifier---------------------------------
# param_grid = {
#     'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# }
# grid_search_fn(KNeighborsClassifier(),param_grid)

print("-----------------------------------End------------------------------")

# print(df.head())

# train_groupby_sex=get_df_by_group(df,['Sex'])
# print(train_groupby_sex)

# f, (ax1, ax2) = plt.subplots(1, 2)
# f.set_figwidth(12)
# f.set_figheight(6)
# sns.barplot(x=train_groupby_sex.index, y='Survived', data=train_groupby_sex, ax=ax1)
# sns.barplot(x=train_groupby_sex.index, y='NumPassengers', data=train_groupby_sex, ax=ax2)
# ax1.set_title('Passengers Survived Per Sex')
# ax2.set_title('Passengers Embarked Per Sex') 
# ax1.plot()
# ax2.plot()


# train_groupby_pclass=get_df_by_group(df,['Pclass'])
# print(train_groupby_pclass)

# # ----------------------------------------Barchart Representation of Pclass Survived-------------------------------------
# f, (ax1, ax2) = plt.subplots(1, 2)
# f.set_figwidth(12)
# f.set_figheight(6)
# sns.barplot(x=train_groupby_pclass.index, y='Survived', data=train_groupby_pclass, ax=ax1)
# sns.barplot(x=train_groupby_pclass.index, y='NumPassengers', data=train_groupby_pclass, ax=ax2)
# ax1.set_title('Passengers Survived Per Class')
# ax2.set_title('Passengers Embarked Per Class') 
# ax1.plot()
# ax2.plot()

# train_groupby_age=get_df_by_group(df,['Age'])
# print(train_groupby_age)

# # ----------------------------------------Histogram Representation of Age Survived-------------------------------------
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

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


# plt.show()
# print('=================================End=================================')