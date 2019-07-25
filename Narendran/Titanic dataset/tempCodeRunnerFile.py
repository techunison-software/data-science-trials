-----------------------------------Applying LogisticRegression Model for Prediction ---------------------------------
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)

# df = pd.DataFrame({'PassengerId': pd.Series(range(892, 1310)), 'Survived': Y_pred})      # Writing predicted values to Logistic_Regression_output.csv output file 
# df.to_csv(path+"/output/Logistic_regression_output.csv", index = False)

# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print('Logistic Reg - ',acc_log)
