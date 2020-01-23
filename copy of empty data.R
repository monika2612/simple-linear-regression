copy_of_empty_data<- read.csv(file.choose())
View(copy_of_empty_data)
attach(copy_of_empty_data)
plot(Salary,Churn)

cor(Churn,Salary)
############3[1]- 0.9117
reg <- lm(Churn~Salary)
summary(reg)
confint(reg,level = 0.95)

predict(reg,inteval="predict")

predict(reg,inteval="predict")
# R-squared value is 0.81 for the above model so strong corelation
# we may have to do transformation of variables for better R-squared value
# Applying transformations

# Logarthmic transformation
reg <- lm(Churn~log(Salary))
summary(reg)
confint(reg,level = 0.95)

predict(reg,inteval="predict")

# R-squared value for the above model is 0.82. so strong corelation
# we may have to do transformation of variables for good R-squared value
# Applying  different transformations
reg_exp<-lm(log(Churn)~Salary) # regression using Exponential model
summary(reg)
# R-squared value for the above model is 0.82. so strong corelation
# we may have to do transformation of variables for good R-squared value

# Quadratic model
quad_mod <- lm(Churn~Salary+Salary*Salary,data=copy_of_empty_data)
summary(quad_mod)
copy_of_empty_data[,"salary_sq"] = Salary*Salary
# Quadratic model
qud_mod <- lm(Churn~Salary+I(Salary^2),data=copy_of_empty_data)
summary(qud_mod)
# R-squared value 0.96 for the above model is 0.89. so strong corelation

# Quadratic model
attach(copy_of_empty_data)
qd_model <- lm(Churn~Salary+salary_sq,data=copy_of_empty_data)
summary(qd_model)



