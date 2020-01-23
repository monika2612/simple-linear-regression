Salary_data<- read.csv(file.choose())
View(Salary_data)
attach(Salary_data)
plot(YearsExperience,Salary)

cor(Salary,YearsExperience)
############3[1]- 0.97
reg <- lm(Salary~YearsExperience)
summary(reg)
confint(reg,level = 0.95)

predict(reg,inteval="predict")

predict(reg,inteval="predict")
# R-squared value is 0.955 for the above model so strong corelation
# we may have to do transformation of variables for better R-squared value
# Applying transformations

# Logarthmic transformation
reg <- lm(Salary~log(YearsExperience))
summary(reg)
confint(reg,level = 0.95)

predict(reg,inteval="predict")


# R-squared value for the above model is 0.84. so strong corelation
# we may have to do transformation of variables for good R-squared value
# Applying  different transformations
reg_exp<-lm(log(Salary)~YearsExperience) # regression using Exponential model
summary(reg)
# R-squared value for the above model is 0.84. so strong corelation
# we may have to do transformation of variables for good R-squared value

