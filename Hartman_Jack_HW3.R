##################################################
# ECON 418-518 Homework 3
# Jack Hartman
# The University of Arizona
# jhartman1@arizona.edu 
# 29 November 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages /preliminary
pacman::p_load(data.table)
# Set working directory to the folder containing your CSV file
setwd("/Users/johnhartman/Desktop")
getwd()
# Load the data set into R
data <- read.csv("ECON_418-518_HW3_Data.csv")
# Check if the working directory is set correctly

data <- as.data.table(data)
names(data)
head(data)
# Set the seed
set.seed(418518)

#####################
# Problem 1
#####################
library(dplyr)

#################
# Question (i)
#################
colnames(data)
data <- data %>% select(-fnlwgt, -occupation, -relationship,-capital.gain, -capital.loss, -educational.num)
colnames(data)


##########
#question (ii)
#########

##############
# Part (a)
##############

# Code
# Convert 'income' column to a binary indicator (1 for ">50K", 0 otherwise)
data$income <- ifelse(data$income == ">50K", 1, 0)

# confirm change
head(data$income)

##############
# Part (b)
##############

# Code
# Convert 'race' column to a binary indicator (1 for "White", 0 otherwise)
data$race <- ifelse(data$race == "White", 1, 0)

# confirm  changes
head(data$race)

####
#part (c)
####

# Convert 'gender' column to a binary indicator (1 for "Male", 0 otherwise)
data$gender <- ifelse(data$gender == "Male", 1, 0)

# confirm  change
head(data$gender)

#####
#part (d)
#####

# Convert 'workclass' column to a binary indicator (1 for "Private", 0 otherwise)
data$workclass <- ifelse(data$workclass == "Private", 1, 0)

# confirm changes
head(data$workclass)

#####
#part (e)
#####

# Convert 'native.country' column to a binary indicator (1 for "United-States", 0 otherwise)
data$native.country <- ifelse(data$native.country == "United-States", 1, 0)

#  confirm  change
head(data$native.country)

#####
#part (f)
######

# Convert 'marital.status' column to a binary indicator (1 for "Married-civ-spouse", 0 otherwise)
data$marital.status <- ifelse(data$marital.status == "Married-civ-spouse", 1, 0)

#  confirm  changes
head(data$marital.status)

######
#part (g)
######

# Convert 'education' column to a binary indicator (1 for "Bachelors", "Masters", or "Doctorate", 0 otherwise)
data$education <- ifelse(data$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

#  confirm  changes
head(data$education)

#####
#part (h)
#####
# Create the 'age_sq' variable as the square of 'age'
data$age_sq <- data$age^2

#  confirm  changes
head(data$age_sq)

#####
#part(i)
#####

# Standardize 'age', 'age_sq', and 'hours.per.week'
data$age_std <- (data$age - mean(data$age, na.rm = TRUE)) / sd(data$age, na.rm = TRUE)
data$age_sq_std <- (data$age_sq - mean(data$age_sq, na.rm = TRUE)) / sd(data$age_sq, na.rm = TRUE)
data$hours_per_week_std <- (data$hours.per.week - mean(data$hours.per.week, na.rm = TRUE)) / sd(data$hours.per.week, na.rm = TRUE)

#  confirm  changes
head(data[, c("age_std", "age_sq_std", "hours_per_week_std")])


#################
# Question (iii)
#################

####
#part(a)
####

# Code
# Calculate the proportion of individuals with income > 50K
proportion_income_gt_50k <- mean(data$income == 1)

#  result
proportion_income_gt_50k

#####
#part (b)
######

# Calculate the proportion of individuals in the private sector
proportion_private_sector <- mean(data$workclass == 1)

# result
proportion_private_sector

######
#part(c)
######

# Calculate the proportion of married individuals
proportion_married <- mean(data$marital.status == 1)

#result
proportion_married

#######
#part(d)
#######
# Calculate the proportion of females
proportion_females <- mean(data$gender == 0)

# result
proportion_females


#######
#part(e)
########
# Calculate the total number of NAs in the entire dataset
total_nas <- sum(is.na(data))

# result
total_nas


#######
#part(f)
########

# Convert 'income' variable to a factor
data$income <- factor(data$income)

str(data$income)
# View the unique levels of the 'income' factor
levels(data$income)

#######
# Question (iv)
#######

######
#part (a)
######
# Calculate the index of the last observation in the training set
train_index <- floor(nrow(data) * 0.70)

#  result 
train_index

######
#part (b)
#######
# Create the training dt
train_data <- data[1:train_index, ]

# confirm
head(train_data)

######
#part (c)
######

# Create the testing dt
test_data <- data[(train_index + 1):nrow(data), ]

#  confirm
head(test_data)

#########
# Question (v)
#########

#####
#part(a)
#####
#see pdf

#####
#part(b)
######

#preliminaries
colnames(data)
install.packages("glmnet")
library(glmnet)
install.packages("caret")
library(caret)

#prepare for training
X <- model.matrix(income ~ age + workclass + education + marital.status + race + gender + hours.per.week + native.country + age_sq + age_std + age_sq_std + hours_per_week_std - 1, train_data)
y <- train_data$income

# seed
set.seed(418518)

#   grid of lambda values over 50 values from 10^5 to 10^-2
lambda_grid <- 10^seq(5, -2, length = 50)

#  train control for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Fit lasso regression model using the train() function
lasso_model <- train(
  x = X,              # Explanatory variables
  y = y,              # Outcome variable
  method = "glmnet",   # Method for lasso regression
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid), # Lasso has alpha = 1
  trControl = train_control,  # Cross-validation control
  metric = "Accuracy" # You can use "Accuracy" or another performance metric
)

# Print the best model's details
print(lasso_model)


#####
#part(c)
#####
#given by code above

####
#part (d)
#####
# Extract the coefficients for the best lambda value
lasso_coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)

# Print
print(lasso_coefficients)


######
#part(e)
#######

# reduced dataset with selected variables
selected_vars <- c("age", "education", "marital.status", "hours.per.week", "income")
reduced_data <- train_data[, ..selected_vars]

# Define lambda values
lambda_grid <- 10^seq(5, -2, length = 50)

# preliminary
library(caret)
library(glmnet)

# cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train lasso  
lasso_model <- train(
  income ~ ., 
  data = reduced_data, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),
  trControl = train_control
)
# Train ridge regression model
ridge_model <- train(
  income ~ ., 
  data = reduced_data, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid),
  trControl = train_control
)
#  best accuracy for lasso
lasso_best_accuracy <- max(lasso_model$results$Accuracy)
lasso_best_lambda <- lasso_model$bestTune$lambda

#  best accuracy for ridge
ridge_best_accuracy <- max(ridge_model$results$Accuracy)
ridge_best_lambda <- ridge_model$bestTune$lambda

#  results
cat("Lasso Best Accuracy:", lasso_best_accuracy, "with lambda =", lasso_best_lambda, "\n")
cat("Ridge Best Accuracy:", ridge_best_accuracy, "with lambda =", ridge_best_lambda, "\n")

#  better model
if (lasso_best_accuracy > ridge_best_accuracy) {
  cat("Lasso regression performs better.\n")
} else {
  cat("Ridge regression performs better.\n")
}

######
#question  (vi)
######

####
#part (a)
####

####
#part(b)
#####

#preliminaries
colnames(data)
library(caret)
install.packages("randomForest")
library(randomForest)

# seed
set.seed(418518)

# Define control for 5-fold cross-validation
control <- trainControl(method = "cv", number = 5)

# Define the tuning grid for the random forest model
tuning_grid <- expand.grid(
  .mtry = c(2, 5, 9)  # Number of random possible features
)

# Prepare the training data
# Exclude the target variable 'income' from predictors
predictors <- c("age", "workclass", "education", "marital.status", 
                "race", "gender", "hours.per.week", "native.country",
                "age_sq", "age_std", "age_sq_std", "hours_per_week_std")
outcome <- "income"

# Train the random forest models with 100, 200, and 300 trees
rf_100 <- train(
  form = as.formula(paste(outcome, "~", paste(predictors, collapse = " + "))),
  data = train_data,
  method = "rf",
  trControl = control,
  tuneGrid = tuning_grid,
  ntree = 100
)

rf_200 <- train(
  form = as.formula(paste(outcome, "~", paste(predictors, collapse = " + "))),
  data = train_data,
  method = "rf",
  trControl = control,
  tuneGrid = tuning_grid,
  ntree = 200
)

rf_300 <- train(
  form = as.formula(paste(outcome, "~", paste(predictors, collapse = " + "))),
  data = train_data,
  method = "rf",
  trControl = control,
  tuneGrid = tuning_grid,
  ntree = 300
)

# Output for each model
cat("Results for 100 Trees:\n")
print(rf_100$results)

cat("\nResults for 200 Trees:\n")
print(rf_200$results)

cat("\nResults for 300 Trees:\n")
print(rf_300$results)


#####
#Part (C)
#####

# code used in part (b)

####
#part (d)
#####

#####
#part (e)
#####

#  predictions using the training model
predictions <- predict(rf_300, newdata = train_data)

#   confusion matrix
conf_matrix <- confusionMatrix(predictions, train_data$income)

# Print 
print(conf_matrix)

# Extract false positives and false negatives
false_positives <- conf_matrix$table[2, 1]  # False positives (predicted 1, actual 0)
false_negatives <- conf_matrix$table[1, 2]  # False negatives (predicted 0, actual 1)

# Print 
cat("False Positives:", false_positives, "\n")
cat("False Negatives:", false_negatives, "\n")


######
# Question (vii)
######

# test Lasso from pt. v

#prepare for testing
X <- model.matrix(income ~ age + workclass + education + marital.status + race + gender + hours.per.week + native.country + age_sq + age_std + age_sq_std + hours_per_week_std - 1, test_data)
y <- test_data$income

# seed
set.seed(418518)

#   grid of lambda values over 50 values from 10^5 to 10^-2
lambda_grid <- 10^seq(5, -2, length = 50)

#  train control for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Fit lasso regression model using the train() function
lasso_model <- train(
  x = X,              # Explanatory variables
  y = y,              # Outcome variable
  method = "glmnet",   # Method for lasso regression
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid), # Lasso has alpha = 1
  trControl = train_control,  # Cross-validation control
  metric = "Accuracy" # You can use "Accuracy" or another performance metric
)

# Print the best model's details
print(lasso_model)


# test updated Lasso and ridge

# reduced dataset with selected variables
selected_vars <- c("age", "education", "marital.status", "hours.per.week", "income")
reduced_data <- test_data[, ..selected_vars]

# Define lambda values
lambda_grid <- 10^seq(5, -2, length = 50)

# preliminary
library(caret)
library(glmnet)

# cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train lasso  
lasso_model <- train(
  income ~ ., 
  data = reduced_data, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),
  trControl = train_control
)
# Train ridge regression model
ridge_model <- train(
  income ~ ., 
  data = reduced_data, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid),
  trControl = train_control
)
#  best accuracy for lasso
lasso_best_accuracy <- max(lasso_model$results$Accuracy)
lasso_best_lambda <- lasso_model$bestTune$lambda

#  best accuracy for ridge
ridge_best_accuracy <- max(ridge_model$results$Accuracy)
ridge_best_lambda <- ridge_model$bestTune$lambda

#  results
cat("Lasso Best Accuracy:", lasso_best_accuracy, "with lambda =", lasso_best_lambda, "\n")
cat("Ridge Best Accuracy:", ridge_best_accuracy, "with lambda =", ridge_best_lambda, "\n")

#  better model
if (lasso_best_accuracy > ridge_best_accuracy) {
  cat("Lasso regression performs better.\n")
} else {
  cat("Ridge regression performs better.\n")
}

# evaluate forest models on testing data

#preliminaries
colnames(data)
library(caret)
install.packages("randomForest")
library(randomForest)

# seed
set.seed(418518)

# Define control for 5-fold cross-validation
control <- trainControl(method = "cv", number = 5)

# Define the tuning grid for the random forest model
tuning_grid <- expand.grid(
  .mtry = c(2, 5, 9)  # Number of random possible features
)

# Prepare the training data
# Exclude the target variable 'income' from predictors
predictors <- c("age", "workclass", "education", "marital.status", 
                "race", "gender", "hours.per.week", "native.country",
                "age_sq", "age_std", "age_sq_std", "hours_per_week_std")
outcome <- "income"

# Train the random forest models with 100, 200, and 300 trees
rf_100 <- train(
  form = as.formula(paste(outcome, "~", paste(predictors, collapse = " + "))),
  data = test_data,
  method = "rf",
  trControl = control,
  tuneGrid = tuning_grid,
  ntree = 100
)

rf_200 <- train(
  form = as.formula(paste(outcome, "~", paste(predictors, collapse = " + "))),
  data = test_data,
  method = "rf",
  trControl = control,
  tuneGrid = tuning_grid,
  ntree = 200
)

rf_300 <- train(
  form = as.formula(paste(outcome, "~", paste(predictors, collapse = " + "))),
  data = test_data,
  method = "rf",
  trControl = control,
  tuneGrid = tuning_grid,
  ntree = 300
)

# Output for each model
cat("Results for 100 Trees:\n")
print(rf_100$results)

cat("\nResults for 200 Trees:\n")
print(rf_200$results)

cat("\nResults for 300 Trees:\n")
print(rf_300$results)


