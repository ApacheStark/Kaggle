##################################
# Kaggle Comp - Titatnic
##################################

library(dplyr)

setwd("C:/Users/samgh/Desktop/Kaggle/Titanic")

train <- read.csv('train.csv')
head(train)
summary(train)

test <- read.csv('test.csv')
head(test)
summary(test)
str(test)

gend <- read.csv('gender_submission.csv')
str(gend)

#######
# Logistic Regression
#######

# grab training data required for log reg (id, name, ticket, cabin?? not important)
train2 <- train[c(2,3,5:8,10,12)]
head(train2)
summary(train2)
str(train2)

# match columns for test
test3 <- test[c(2,4,5:7,9,11:12)]
colnames(test3)
colnames(train2) # ensure columns matchstr
summary(test3)

# Deal with the many NA values in age and one NA in Fare

# See how a basic mean replacement works

train2$Age[is.na(train2$Age)] <- mean(train2$Age, na.rm = TRUE)
test3$Age[is.na(test3$Age)] <- mean(train2$Age, na.rm = TRUE)

test3$Fare[is.na(test3$Fare)] <- mean(test3$Fare, na.rm = TRUE)

# feature scaling for int/num values not considered due to small dataset size (optimisation not required)

# fit logistic reg classifier
classifier <- glm(formula = Survived ~ .,
                  data = train2,
                  family = binomial)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test3[-8])

# Test with a new vector 
y_pred = ifelse(prob_pred > 0.50, 1, 0) 

# Export CSV findings
test_final <- cbind(test[1], y_pred)

colnames(test_final) <- c('PassengerID','Survived')

# The silliest thing is... all males died and all females lived in the test set, not the same as the train set though
# maybe try a different type of NA replacement?
write.csv(test_final,
          file = 'titantic_final.csv',
          row.names = FALSE)

# terrible... 0.765 from Kaggle

# improvements:
# new way of replacing NAs, use regression, separately for both train and test
# include cabin variable

#######
# TAKE TWO
#######
library(dplyr)

setwd("C:/Users/samgh/Desktop/Kaggle/Titanic")

train <- read.csv('train.csv')
head(train)
summary(train)

test <- read.csv('test.csv')
head(test)
summary(test)
str(test)

# use rand forest reg for age prediction
library(randomForest)
str(train)
train2 <- train[c(2,3,5:8,10,12)]
str(train2)
train2 <- train2[complete.cases(train2),]
train_rf <- randomForest(x = train2[-4],
                          y = train2$Age,
                          ntree = 100)

x <- predict(train_rf, train[-6])
train$Age %>% summary
train$Age <- replace(train$Age, is.na(train$Age), x)

test3 <- test2[complete.cases(test2),]
test_rf <- randomForest(x = test3[-3],
                        y = test3$Age,
                        ntree = 100)

test$Fare[is.na(test$Fare)] <- mean(test$Fare, na.rm = TRUE)
pred_test <- predict(test_rf, test[-5])
test2$Age <- replace(test2$Age, is.na(test2$Age), pred_test)
View(train)
colnames(train)
train2 <- train[c(2,3,5,6,7,8,10,12)]
colnames(train2)
summary(train2)
colnames(test)
test2 <- test[c(2,4,5,6,7,9,11)]
colnames(test2)
summary(test2)

test2$Fare[is.na(test2$Fare)] <- mean(test2$Fare, na.rm = TRUE)
# fit logistic reg classifier
classifier <- glm(formula = Survived ~ .,
                  data = train2,
                  family = binomial)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test2)

# Test with a new vector 
y_pred = ifelse(prob_pred > 0.50, 1, 0) 

# Export CSV findings
test_final <- cbind(test[1], y_pred)
summary(test_final)
colnames(test_final) <- c('PassengerID','Survived')

write.csv(test_final,
          file = 'titantic_final2.csv',
          row.names = FALSE)

# this is worse by 0.5%...

# look into properly factorising cabin variable, maybe that had an impact, maybe look into the different sections in general (A,B,C...)??
# maybe look into Rand Forest classification? or decision trees?
