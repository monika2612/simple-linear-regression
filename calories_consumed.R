calories_consumed <- read.csv(file.choose())
View(calories_consumed)
attach(calories_consumed)
plot(Weight,Calories)

cor(Calories,Weight)
############3[1] 0.946991
reg <- lm(Calories~Weight)
summary(reg)
confint(reg,level = 0.95)

predict(reg,inteval="predict")

predict(reg,inteval="predict")
# R-squared value for the above model is 0.8. so strong corelation
# we may have to do transformation of variables for better R-squared value
# Applying transformations

# Logarthmic transformation
reg <- lm(Calories~log(Weight))
summary(reg)
confint(reg,level = 0.95)

predict(reg,inteval="predict")

predict(reg,inteval="predict")
# R-squared value for the above model is 0.87. so strong corelation
# we may have to do transformation of variables for good R-squared value
# Applying  different transformations
reg_exp<-lm(log(Calories)~Weight) # regression using Exponential model
summary(reg)
# R-squared value for the above model is 0.87. so strong corelation
# we may have to do transformation of variables for good R-squared value

# Quadratic model
quad_mod <- lm(Calories~Weight+Weight*Weight,data=calories_consumed)
summary(quad_mod)
calories_consumed[,"weight_sq"] = Weight*Weight
# Quadratic model
qud_mod <- lm(Calories~Weight+I(Weight^2),data=calories_consumed)
summary(qud_mod)
# R-squared value for the above model is 0.89. so strong corelation

# Quadratic model
attach(calories_consumed)
qd_model <- lm(Calories~Weight+weight_sq,data=calories_consumed)
summary(qd_model)



