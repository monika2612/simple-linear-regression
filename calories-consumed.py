# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:13:56 2019

@author: admin
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#import linear_model
ccd=pd.read_csv('E:/excelr/Excelr Data/Assignments/Simple Linear Regression/calories_consumed.csv')

print(ccd)
print(ccd.shape)
plt.hist(ccd.Wei)
plt.hist(ccd.Calories)
plt.boxplot(ccd.Wei)
plt.boxplot(ccd.Wei,0,"rs",1)
plt.boxplot(ccd.Wei,ccd.Calories,"ro");
plt.xlabel("Wei");
plt.ylabel("Calories")

plt.boxplot(ccd.Calories)


import statsmodels.formula.api as smf
model=smf.ols("Wei~Calories",data=ccd).fit()
##type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(ccd) # Predicted values of AT using the model

import matplotlib.pylab as plt
plt.scatter(x=ccd['Calories'],y=ccd['Wei'],color='red');plt.plot(ccd['Calories'],pred,color='black');plt.xlabel('CALORIES');plt.ylabel('WEIGHT')


# Transforming variables for accuracy
model2 = smf.ols('Wei~np.log(Calories)',data=ccd).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(ccd)
pred2.corr(ccd.Wei)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=ccd['Calories'],y=ccd['Wei'],color='red');plt.plot(ccd['Calories'],pred2,color='black');plt.xlabel('CALORIES');plt.ylabel('WEIGHT')

#plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green');plt.plot(wcat['Waist'],pred2,color='blue');plt.xlabel('WAIST');plt.ylabel('TISSUE')

# Exponential transformation
model3 = smf.ols('np.log(Wei)~Calories',data=ccd).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(ccd)
pred_log
pred3=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
pred3
pred3.corr(ccd.Wei)
plt.scatter(x=ccd['Calories'],y=ccd['Wei'],color='green');plt.plot(ccd.Calories,np.exp(pred_log),color='blue');plt.xlabel('Calories');plt.ylabel('WEIGHT')
resid_3 = pred3-ccd.Wei
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=ccd.wei);plt.xlabel("Predicted");plt.ylabel("Actual")

# Quadratic model
ccd["Wei_Sq"] = ccd.Wei*ccd.Wei
model_quad = smf.ols("Calories~Wei+Wei_Sq",data=ccd).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(ccd)

model_quad.conf_int(0.05) # 
plt.scatter(ccd.Wei,ccd.Calories,c="b");plt.plot(ccd.Wei,pred_quad,"r")


plt.hist(model_quad.resid_pearson) # histogram for residual values 



