# Fraud Detection in R
# Author: Tim Hopp
# 11/10/2022

# Packages:
install.packages('caTools')
install.packages('DMwR')
install.packages('ROSE')
install.packages('Rborist')
install.packages('rpart')
install.packages('Rtsne')
library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(DMwR) # for smote implementation
library(ROSE)# for ROSE sampling
library(rpart)# for decision tree model
library(Rborist)# for random forest model
library(xgboost) # for xgboost model
# read in dataset from Kaggle
df <- read.csv("C:\\Users\\timho\\OneDrive\\Desktop\\ML\\R_Fraud_Detection\\creditcard.csv")

# what is in dataset?

dim(df)

table(df$Class)
colnames(df)
totalamount_dollars <- sum(df$Amount)

# how much in fraud?
fraudtrans <- subset(df, Class == '1')
sum(fraudtrans$Amount)
fraudamount_dollars <- 60127.97/25162590

# Data set should be clean due to prior PCA but check for NA's

colSums(is.na(df))

# remove time or not?
df <- df[,-1]

# Change class column to factor
df$Class <- as.factor(df$Class)
levels(df$Class) <- c("Not_Fraud", "Fraud")
head(df)

# split into test and training data with 70% training split
set.seed(123)
split <- sample.split(df$Class, SplitRatio = 0.7)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE) 

# what do testing and training splits look like?
dim(train)
dim(test)
table(train$Class)
# very imbalanced data set - how to sample?
# first approach - undersampling to avoid overfitting future models
set.seed(456)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Class)
table(down_train$Class)

# also try upsampling but check for overfitting especially w NN & trees
set.seed(789)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Class)
table(up_train$Class)

# look at smote and rose sampling next?

# fit down sample to decision tree model
set.seed(3452)
down_sample_fit <- rpart(Class ~ ., data = down_train)

# use AUC for accuracy
down_sample_pred <- predict(down_sample_fit, newdata = test)
roc.curve(test$Class, down_sample_pred[,2], plotit = FALSE)

# fit up sample to decision tree model
set.seed(9837)
up_sample_fit <- rpart(Class ~ ., data = up_train)

# use AUC for accuracry of up train
up_sample_pred <- predict(up_sample_fit, newdata = test)
roc.curve(test$Class, up_sample_pred[,2], plotit = TRUE)
