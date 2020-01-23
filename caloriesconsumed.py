# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:08:42 2019

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('D:/excelr/Excelr Data/Assignments/Simple Linear Regression/calories_consumed.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/2)
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(regressor.score(X_test,Y_test))
##############