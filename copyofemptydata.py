# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:08:40 2019

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
coed=pd.read_csv('E:/excelr/Excelr Data/Assignments/Simple Linear Regression/Copy of emp_data.csv')
print(coed)
coed.columns
#########co####################################

plt.hist(coed.Salary)
plt.hist(coed.Churn)
plt.boxplot(coed.Salary)
plt.boxplot(coed.Churn)
plt.boxplot(coed.Salary,0,"rs",1)
#####plt.plot(wcat.Waist,wcat.AT,"ro");plt.xlabel("waist");plt.ylabel("AT")
plt.plot(coed.Salary,coed.Churn,"ro");
plt.xlabel("salary");
plt.ylabel("Churn")
#plt.plot(np.arange(109),wcat.Waist,"ro-")
#plt.hist(wcat.AT)
#plt.boxplot(wcat.AT)
plt.plot(np.arange(1800),coed.Salary,"ro-")



#wcat.corr()
##wcat.AT.corr(wcat.Waist) # # correlation value between X and Y
#np.corrcoef(wcat.AT,wcat.Waist)
coed.Salary.corr(coed.Churn)
coed.Churn.corr(coed.Salary)

np.corrcoef(coed.Salary,coed.Churn)
# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Salary~Churn",data=coed).fit()
type(model)
#model=smf.ols("AT~Waist",data=wcat).fit()
##type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval
pred = model.predict(coed) # Predicted values of AT using the model


import matplotlib.pylab as plt
##plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='red');
#plt.plot(wcat['Waist'],pred,color='black');plt.xlabel('WAIST');plt.ylabel('TISSUE')

plt.scatter(y=coed['Salary'],x=coed['Churn'],color='red');
plt.plot(coed['Churn'],pred,color='black');
plt.ylabel('Salary');
plt.xlabel('Churn')

# Transforming variables for accuracy
model2 = smf.ols('Salary~np.log(Churn)',data=coed).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(coed)
pred2.corr(coed.Salary)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
###plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green');plt.plot(wcat['Waist'],pred2,color='blue');plt.xlabel('WAIST');plt.ylabel('TISSUE')
plt.scatter(y=coed['Salary'],x=coed['Churn'],color='green');
plt.plot(coed['Churn'],pred2,color='black');
plt.ylabel('Salary');
plt.xlabel('Churn')

# Exponential transformation
#model3 = smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3=smf.ols('np.log(Salary)~Churn',data=coed).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(coed)
pred_log
pred3=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
pred3
pred3.corr(coed.Salary)
plt.scatter(y=coed['Salary'],x=coed['Churn'],color='green');
plt.plot(coed.Churn,np.exp(pred_log),color='black');
plt.ylabel('Salary');
plt.xlabel('Churn')

#plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green');
#plt.plot(wcat.Waist,np.exp(pred_log),color='blue');plt.xlabel('WAIST');plt.ylabel('TISSUE')
resid_3 = pred3-coed.Salary
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=coed.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")


# Quadratic model


coed["Salary_sq"]=coed.Salary*coed.Salary
#wcat["Waist_Sq"] = wcat.Waist*wcat.Waist
#model_quad = smf.ols("AT~Waist+Waist_Sq",data=wcat).fit()
model_quad=smf.ols("Churn~Salary+Salary_sq",data=coed).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(coed)

model_quad.conf_int(0.05) # 
#plt.scatter(wcat.Waist,wcat.AT,c="b");plt.plot(wcat.Waist,pred_quad,"r")
plt.scatter(coed.Salary,coed.Churn,c='b');
plt.plot(coed.Salary,pred_quad,"r")
#plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');
plt.scatter(np.arange(1800),model_quad.resid_peason);
plt.axhline(y=0,color='red');
plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


plt.hist(model_quad.resid_pearson) # histogram for residual values 





