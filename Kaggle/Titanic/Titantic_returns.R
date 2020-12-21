# Titanic 2020-12-01

# Revamped and hopefully an improvement!

# Let's try this shall we?
# Desc stats and visualise training/testing data with respect to survived,
# look at NAs and visualise surrounding variables to create imputing models (rf/xgboost)
# Model as follows: rf, xgb, superlearner with CV

library(randomForest)
library(xgboost)
library(tidyverse)
library(caret)


train <- read_csv('train.csv')

test <- read_csv('test.csv')

submission_format <- read_csv('gender_submission.csv')

# NAs # 
sapply(train, function(x)sum(is.na(x)))
sapply(test, function(x)sum(is.na(x)))

# Notes on NA:
# - NA in Fare for test only, need to impute that before predictions
# - Cabin can be mutated into just hasCabin or not
# - Age needs to be imputed

### DESCRIPTIVE STATISTICS ###
# Quantitative  #
 
groupby_survived <- function(df, ...) {
  df %>% 
    group_by(...) %>% 
    count(Survived) %>% 
    mutate(pp = n/sum(n)) %>% 
    arrange(desc(pp))
}

# Sex
groupby_survived(train, Sex)
# Male has a massive impact on survival rates

# Cabin
groupby_survived(train, Cabin)
# Q: What do the cabin numbers and letters mean? Is there an interaction here?
train <- train %>% 
  mutate(CabinClass = str_remove_all(Cabin, '[:digit:]'),
         CabinNo = str_trim(str_remove_all(Cabin, '[:alpha:]')),
         HasCabin = ifelse(is.na(Cabin), F, T))
train %>% count(CabinClass, sort = T)

groupby_survived(train, CabinClass)
groupby_survived(train, CabinNo)
groupby_survived(train, HasCabin)
# simply having a cabin means a better survival rate

# Embarked
groupby_survived(train, Embarked)

# Pclass
groupby_survived(train, Pclass)
# Sex, cabin, and class so far have a significant indicator of survival

# Quantitative against Quantitative #
groupby_survived(train, Pclass, HasCabin) %>% 
  ggplot(., aes(y = pp, x = HasCabin, fill = as.factor(Survived))) +
  geom_bar(stat = 'identity', position = 'dodge') +
  facet_wrap(~Pclass)

# Qualitative against Quant #
ggplot(train, aes(x = Age, y = Fare, col = as.factor(Survived))) +
  facet_grid(rows = vars(Sex), cols = vars(HasCabin)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = 'lm', se = F, size = 1) +
  theme_bw()

### IMPUTING ###
# Impute Age #
# grab title
train <- train %>% 
  mutate(Title = case_when(
    str_detect(Name, 'Mrs') ~ 'Mrs',
    str_detect(Name, 'Mr') ~ 'Mr',
    str_detect(Name, 'Master') ~ 'Master',
    str_detect(Name, 'Miss|Ms') ~ 'Miss',
    str_detect(Name, 'Rev') ~ 'Rev',
    str_detect(Name, 'Dr') ~ 'Dr',
    str_detect(Name, 'Major') ~ 'Major',
    T ~ 'Other'
  ))

train %>% count(Title, Sex, sort = T) # That has worked, still have 24 Others

train %>% 
  group_by(Title) %>% 
  summarise(Missing = sum(is.na(Age)),
            Min = min(Age, na.rm = T),
            Median = median(Age, na.rm = T),
            Mean = mean(Age, na.rm = T),
            Max = max(Age, na.rm = T))

train_sub <- train %>% select(PassengerId, Pclass, Sex, Age) %>% 
  mutate(
    # Title = as.factor(Title),
    WasNA = ifelse(is.na(train_sub$Age), T, F))

train_sub_rf_full <- train %>% select(PassengerId, Pclass, Sex, Age, HasCabin, Fare, Parch, SibSp) %>% 
  mutate(
    # Title = as.factor(Title),
         Sex = as.factor(Sex),
         WasNA = ifelse(is.na(train_sub_rf_full$Age), T, F),
         Pclass = as.factor(Pclass))

train_sub_rf <- train_sub_rf_full  %>% filter(!is.na(Age))

# Simple linear regression
impute_age_lm <- lm(Age~., data = train_sub)
summary(impute_age_lm)

# Random Forest
impute_age_rf <- randomForest::randomForest(Age ~ .,
                                            data = train_sub_rf %>% select(-PassengerId, -WasNA),
                                            strata = 'Sex',
                                            ntree = 100)
plot(impute_age_rf)
varImpPlot(impute_age_rf)

# Impute for Fare using a random forest model/strata by Pclass as this determines fare 
train_sub_rf_full %>% 
  group_by(Pclass) %>% 
  summarise(Median = median(Fare),
            Mean = mean(Fare))
impute_fare_rf <- randomForest::randomForest(Fare ~ .,
                                             data = train_sub_rf %>% select(-PassengerId, -WasNA),
                                             strata = 'Pclass',
                                             ntree = 100)

plot(impute_fare_rf)
varImpPlot(impute_fare_rf)

# Impute with RF models
train_sub$Age_lm <- round(replace(train_sub$Age, is.na(train_sub$Age), predict(impute_age_lm, train_sub)))

train_sub_rf_full$Age_rf <- round(replace(train_sub_rf_full$Age, 
                                          is.na(train_sub_rf_full$Age), 
                                          predict(impute_age_rf, train_sub_rf_full[-4])))
sum(is.na(train_sub_rf_full$Age_rf))

# Look at both models density
train_sub %>% 
  ggplot(., aes(x = Age)) +
  geom_density(alpha = 0.5, fill = 'green') +
  geom_density(mapping = aes(x = Age, fill = WasNA), data = train_sub, alpha = 0.3) +
  labs(fill = 'Was Missing')
train_sub_rf_full %>% 
  ggplot(., aes(x = Age_rf)) +
  geom_density(alpha = 0.5, fill = 'green') +
  geom_density(mapping = aes(x = Age, fill = WasNA), data = train_sub, alpha = 0.3) +
  labs(fill = 'Was Missing')

# Create model (rf then xgb)
train_full <- train %>% 
  select(PassengerId, Survived) %>% 
  left_join(train_sub_rf_full) %>% 
  select(-Age, -WasNA) %>% 
  rename(Age = Age_rf) %>% 
  mutate(Survived = as.factor(Survived))

RF <- randomForest(Survived ~ .,
                   data = train_full %>% select(-PassengerId),
                   strata = 'Sex',
                   ntree = 100,
                   predicted = T,
                   type = 'classification',
                   localImp = T,
                   proximity = T)
RF
plot(RF)
randomForest::varImpPlot(RF)


RF_pclass <- randomForest(Survived ~ .,
                   data = train_full %>% select(-PassengerId),
                   strata = 'Pclass',
                   ntree = 100,
                   predicted = T,
                   type = 'classification',
                   localImp = T,
                   proximity = T)

# SVM
library(e1071)
SVM <- svm(formula = Survived ~ .,
           data = train_full %>% select(-PassengerId),
           kernel = 'linear',
           # degree = 3,
           cost = 10)

# XGBoost
train_xg <-train_full %>% 
  select(-PassengerId, -Survived) %>% 
  mutate(Sex = as.numeric(Sex),
         HasCabin = as.numeric(HasCabin),
         Pclass = as.numeric(Pclass))%>% 
  as.matrix()
labels_xg <- train_full$Survived %>% 
  as.numeric() - 1
XGB <- xgboost(data = train_xg,
               label = labels_xg,
               max_depth = 3,
               eta = 1,
               nthread = 3,
               nrounds = 200,
               early_stopping_rounds = 20,
               objective = 'binary:logistic')

XGB_multi <- xgboost(data = train_xg,
               label = labels_xg,
               max_depth = 5,
               eta = 0.5,
               nthread = 3,
               nrounds = 200,
               early_stopping_rounds = 20,
               objective = 'binary:logistic')

XGB_multi$best_score

# What do we need to do to this here test data?
# Create HasCabin
# Impute Age and Fare
sapply(test, function(x)sum(is.na(x)))

test_prep <- test %>% 
  mutate(HasCabin = ifelse(is.na(Cabin), F, T),
         Sex = as.factor(Sex),
         Pclass = as.factor(Pclass)) 
test_prep$Fare <- replace(test_prep$Fare,
                          is.na(test_prep$Fare),
                          predict(impute_fare_rf, test_prep))
test_prep$Age <- replace(test_prep$Age,
                          is.na(test_prep$Age),
                          predict(impute_age_rf, test_prep))

sapply(test_prep, function(x)sum(is.na(x)))

# Pred with RF and XGB
# RF
RF_preds <- predict(RF, test_prep)
names(RF_preds) <- test_prep$PassengerId

table(RF_preds)

test_xg <- test_prep[XGB$feature_names] %>% 
  mutate(Pclass = as.numeric(Pclass),
         Sex = as.numeric(Sex),
         HasCabin = as.numeric(HasCabin))
# XG
XGB_probs <- predict(XGB, 
                     data.matrix(test_xg))
XGB_preds <- ifelse(XGB_probs > 0.5, 1, 0)
# XG_multi
XGB_probs_m <- predict(XGB_multi, 
                     data.matrix(test_xg))
XGB_preds_m <- ifelse(XGB_probs_m > 0.5, 1, 0)


# SVM 
SVM_preds <- predict(SVM, test_prep)


# Collect predictions
predictions <- tibble(PassengerID = test_prep$PassengerId,
                      RF = as.numeric(RF_preds) - 1,
                      XGB = XGB_preds,
                      XGB_m = XGB_preds_m,
                      Vote = ifelse(RF == XGB, RF, XGB_m))

predictions %>% count(RF, XGB, XGB_m)
predictions %>% count(Vote)
predictions_out <- predictions %>% 
  select(PassengerID,
         Survived = Vote)

# NNs?
# CHANGES

asdnsdubf <- 1

write_csv(predictions_out,
          'Titantic_out_returns.csv')
