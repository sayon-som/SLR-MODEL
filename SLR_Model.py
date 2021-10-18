# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:19:48 2021

@author: Sayon Som

TITLE: A  SIMPLE LINEAR REGRESSION MODEL TO PREDICT THE SALARY OF AN EMPLOYEE BEASED ON YEARS OF EXP 
"""
    
    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Importing the dataset
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    
    # Training the Simple Linear Regression model on the Training set
    from sklearn.linear_model import LinearRegression
    Regression=LinearRegression()
    
    Regression.fit(X_train, y_train)
    
    
    
    
    # Predicting the Test set results
    y_pred=Regression.predict(X_test)
    predict_salary=Regression.predict([[8]])
    
    print("{} is the salary is your are expecting".format(predict_salary[0]))
    
    # Visualising the Training set results
    plt.scatter(X_train,y_train,color="red")
    plt.plot(X_train,Regression.predict(X_train),color="black")
    plt.title("Linear model")
    plt.xlabel("exp")
    plt.ylabel("salary")
    plt.show()
    # Visualising the Test set results
    plt.scatter(X_test,y_test,color="red")
    plt.plot(X_test,y_pred,color="black")
    plt.title("Linear model test")
    plt.xlabel("exp")
    plt.ylabel("salary")
    plt.show()


