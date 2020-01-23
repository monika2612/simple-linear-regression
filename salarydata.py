# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 01:26:22 2019

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
salary=pd.read_csv('E:/excelr/Excelr Data/Assignments/Simple Linear Regression/Salary_Data.csv')
print(salary)
#######################################################################
X=salary.iloc[:,:-1].values


Y=salary.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3)
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(regressor.score(X_test,Y_test))
#################0.93##########################################################


salary.columns

plt.hist(salary.Salary)
plt.boxplot(salary.Salary)
plt.boxplot(salary.Salary,0,"rs",1)
?plt.boxplot
plt.plot(salary.Salary,salary.YearsExperience,"ro");plt.xlabel("Salary");plt.ylabel("YearsExperience")
plt.plot(np.arange(100000),salary.Salary,"ro-")
plt.hist(salary.YearsExperience)
plt.boxplot(salary.YearsExperience)

##plt.plot(wcat.Waist,wcat.AT,"bo");plt.xlabel("Waist");plt.ylabel("AT")

#wcat.corr()
salary.YearsExperience.corr(salary.Salary) # # correlation value between X and Y
#np.corrcoef(wcat.AT,wcat.Waist)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=salary).fit()
##type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()
model.conf_int(0.05) # 95% confidence interval

pred = model.predict(salary) # Predicted values of salry using the model

# Visualization of regresion line over the scatter plot of salary and ye
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=salary['YearsExperience'],y=salary['Salary'],color='red');plt.plot(salary['YearsExperience'],pred,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')


# Transforming variables for accuracy
model2 = smf.ols('Salary~np.log(YearsExperience)',data=salary).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(salary)
pred2.corr(salary.Salary)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=salary['YearsExperience'],y=salary['Salary'],color='green');plt.plot(salary['YearsExperience'],pred2,color='blue');plt.xlabel('YearsExperience');plt.ylabel('Salary')

# Exponential transformation
model3 = smf.ols('np.log(YearsExperience)~Salary',data=salary).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(salary)
pred_log
pred3=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
pred3
pred3.corr(salary.YearsExperience)
plt.scatter(x=salary['YearsExperience'],y=salary['Salary'],color='green');plt.plot(salary.Salary,np.exp(pred_log),color='blue');plt.xlabel('YearsExperience');plt.ylabel('Salary')
resid_3 = pred3-salary.YearsExperience
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=salary.YearsExperience);plt.xlabel("Predicted");plt.ylabel("Actual")

# Quadratic model
salary["Salary_Sq"] = salary.Salary*salary.Salary
model_quad = smf.ols("YearsExperience~Salary+Salary_Sq",data=salary).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(salary)

model_quad.conf_int(0.05) # 
plt.scatter(salary.Salary,salary.YearsExperience,c="b");plt.plot(salary.Salary,pred_quad,"r")

plt.scatter(np.arange(10000),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 


