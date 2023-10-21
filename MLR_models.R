library(glmnet)

# MLR model on original bag of words model --------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(bow_200), size = nrow(bow_200)*0.7, replace = F)
train_bow_200 <- bow_200[ind,-2] # 70% training set
train_bow_200$Pid <- as.factor(train_bow_200$Pid) # convert target variable to factor for training set
test_bow_200 <- bow_200[-ind,-2] # 30% training set
test_bow_200$Pid <- as.factor(test_bow_200$Pid) # convert target variable to factor for test set

# convert predictor variables into a matrix/array as glmnet only accepts this data type as input
X_train_bow_200 <- as.matrix(train_bow_200[,-1])
# convert response variable into a matrix/array as glmnet only accepts matrix data type as input
Y_train_bow_200 <- as.matrix(train_bow_200[,1])

X_test_bow_200 <- as.matrix(test_bow_200[,-1])

# Fit multinomial logistic regression
bow_200_mlr <- glmnet(X_train_bow_200, Y_train_bow_200, 
                    alpha = 1, # invoke lasso as regularisation method
                    standardize = T, 
                    family = 'multinomial')

# 10-fold CV results for logistic regression with LASSO penalty
set.seed(1)
cv_lasso_bow_200_mlr <- cv.glmnet(X_train_bow_200, Y_train_bow_200,
                      alpha = 1, # invoke lasso as regularisation method
                      nfolds = 10, # 10-folds cross validation
                      type.measure = 'class', # classification problem
                      standardize = T,
                      family = 'multinomial')

save(bow_200_mlr, cv_lasso_bow_200_mlr, file = "Rdata/MLR_bow_200.Rdata")
mlr_bow_200_train_accuracy <- mean(predict(bow_200_mlr, 
                                     newx = X_train_bow_200, 
                                     s = cv_lasso_bow_200_mlr$lambda.min, 
                                     type = 'class') == train_bow_200$Pid) # training accuracy

mlr_bow_200_test_accuracy <- mean(predict(bow_200_mlr, 
                                           newx = X_test_bow_200, 
                                           s = cv_lasso_bow_200_mlr$lambda.min, 
                                           type = 'class') == test_bow_200$Pid) # test accuracy

# Matthew's correlation coefficient
bow_200_mlr_mcc <- mcc(preds = as.numeric(predict(bow_200_mlr, 
                                       newx = X_test_bow_200, 
                                       s = cv_lasso_bow_200_mlr$lambda.min, 
                                       type = 'class')),
                       actuals = as.numeric(test_bow_200$Pid)) # MCC

##############################################################################################################################
# MLR model on upsampled bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(US_bow_200), size = nrow(US_bow_200)*0.7, replace = F)
train_US_bow_200 <- US_bow_200[ind,-c(2,203)] # 70% training set
train_US_bow_200$Pid <- as.factor(train_US_bow_200$Pid) # convert target variable to factor for training set
test_US_bow_200 <- US_bow_200[-ind,-c(2,203)] # 30% training set
test_US_bow_200$Pid <- as.factor(test_US_bow_200$Pid) # convert target variable to factor for test set

# convert predictor variables into a matrix/array as glmnet only accepts this data type as input
X_train_US_bow_200 <- as.matrix(train_US_bow_200[,-1])
# convert response variable into a matrix/array as glmnet only accepts matrix data type as input
Y_train_US_bow_200 <- as.matrix(train_US_bow_200[,1])

X_test_US_bow_200 <- as.matrix(test_US_bow_200[,-1])

# Fit multinomial logistic regression
US_bow_200_mlr <- glmnet(X_train_US_bow_200, Y_train_US_bow_200, 
                      alpha = 1, # invoke lasso as regularisation method
                      standardize = T, 
                      family = 'multinomial')

# 10-fold CV results for logistic regression with LASSO penalty
set.seed(1)
cv_lasso_US_bow_200_mlr <- cv.glmnet(X_train_US_bow_200, Y_train_US_bow_200,
                                  alpha = 1, # invoke lasso as regularisation method
                                  nfolds = 10, # 10-folds cross validation
                                  type.measure = 'class', # classification problem
                                  standardize = T,
                                  family = 'multinomial')

save(US_bow_200_mlr, cv_lasso_US_bow_200_mlr, file = "Rdata/MLR_US_bow_200.Rdata")

mlr_US_bow_200_train_accuracy <- mean(predict(US_bow_200_mlr, 
                                           newx = X_train_US_bow_200, 
                                           s = cv_lasso_US_bow_200_mlr$lambda.min, 
                                           type = 'class') == train_US_bow_200$Pid) # training accuracy

mlr_US_bow_200_test_accuracy <- mean(predict(US_bow_200_mlr, 
                                          newx = X_test_US_bow_200, 
                                          s = cv_lasso_US_bow_200_mlr$lambda.min, 
                                          type = 'class') == test_US_bow_200$Pid) # test accuracy

##############################################################################################################################
# MLR model on downsampled bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(DS_bow_200), size = nrow(DS_bow_200)*0.7, replace = F)
train_DS_bow_200 <- DS_bow_200[ind,-c(2,203)] # 70% training set
train_DS_bow_200$Pid <- as.factor(train_DS_bow_200$Pid) # convert target variable to factor for training set
test_DS_bow_200 <- DS_bow_200[-ind,-c(2,203)] # 30% training set
test_DS_bow_200$Pid <- as.factor(test_DS_bow_200$Pid) # convert target variable to factor for test set

# convert predictor variables into a matrix/array as glmnet only accepts this data type as input
X_train_DS_bow_200 <- as.matrix(train_DS_bow_200[,-1])
# convert response variable into a matrix/array as glmnet only accepts matrix data type as input
Y_train_DS_bow_200 <- as.matrix(train_DS_bow_200[,1])

X_test_DS_bow_200 <- as.matrix(test_DS_bow_200[,-1])

# Fit multinomial logistic regression
DS_bow_200_mlr <- glmnet(X_train_DS_bow_200, Y_train_DS_bow_200, 
                         alpha = 1, # invoke lasso as regularisation method
                         standardize = T, 
                         family = 'multinomial')

# 10-fold CV results for logistic regression with LASSO penalty
set.seed(1)
cv_lasso_DS_bow_200_mlr <- cv.glmnet(X_train_DS_bow_200, Y_train_DS_bow_200,
                                     alpha = 1, # invoke lasso as regularisation method
                                     nfolds = 10, # 10-folds cross validation
                                     type.measure = 'class', # classification problem
                                     standardize = T,
                                     family = 'multinomial')

save(DS_bow_200_mlr, cv_lasso_DS_bow_200_mlr, file = "Rdata/MLR_DS_bow_200.Rdata")

mlr_DS_bow_200_train_accuracy <- mean(predict(DS_bow_200_mlr, 
                                              newx = X_train_DS_bow_200, 
                                              s = cv_lasso_DS_bow_200_mlr$lambda.min, 
                                              type = 'class') == train_DS_bow_200$Pid) # training accuracy

mlr_DS_bow_200_test_accuracy <- mean(predict(DS_bow_200_mlr, 
                                             newx = X_test_DS_bow_200, 
                                             s = cv_lasso_DS_bow_200_mlr$lambda.min, 
                                             type = 'class') == test_DS_bow_200$Pid) # test accuracy


##############################################################################################################################
# MLR model on original tfidf bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(tfidf_200), size = nrow(tfidf_200)*0.7, replace = F)
train_tfidf_200 <- tfidf_200[ind,-2] # 70% training set
train_tfidf_200$Pid <- as.factor(train_tfidf_200$Pid) # convert target variable to factor for training set
test_tfidf_200 <- tfidf_200[-ind,-2] # 30% training set
test_tfidf_200$Pid <- as.factor(test_tfidf_200$Pid) # convert target variable to factor for test set

# convert predictor variables into a matrix/array as glmnet only accepts this data type as input
X_train_tfidf_200 <- as.matrix(train_tfidf_200[,-1])
# convert response variable into a matrix/array as glmnet only accepts matrix data type as input
Y_train_tfidf_200 <- as.matrix(train_tfidf_200[,1])

X_test_tfidf_200 <- as.matrix(test_tfidf_200[,-1])

# Fit multinomial logistic regression
tfidf_200_mlr <- glmnet(X_train_tfidf_200, Y_train_tfidf_200, 
                      alpha = 1, # invoke lasso as regularisation method
                      standardize = T, 
                      family = 'multinomial')

# 10-fold CV results for logistic regression with LASSO penalty
set.seed(1)
cv_lasso_tfidf_200_mlr <- cv.glmnet(X_train_tfidf_200, Y_train_tfidf_200,
                                  alpha = 1, # invoke lasso as regularisation method
                                  nfolds = 10, # 10-folds cross validation
                                  type.measure = 'class', # classification problem
                                  standardize = T,
                                  family = 'multinomial')

save(tfidf_200_mlr, cv_lasso_tfidf_200_mlr, file = "Rdata/MLR_tfidf_200.Rdata")
mlr_tfidf_200_train_accuracy <- mean(predict(tfidf_200_mlr, 
                                           newx = X_train_tfidf_200, 
                                           s = cv_lasso_tfidf_200_mlr$lambda.min, 
                                           type = 'class') == train_tfidf_200$Pid) # training accuracy

mlr_tfidf_200_test_accuracy <- mean(predict(tfidf_200_mlr, 
                                          newx = X_test_tfidf_200, 
                                          s = cv_lasso_tfidf_200_mlr$lambda.min, 
                                          type = 'class') == test_tfidf_200$Pid) # test accuracy

# Matthew's correlation coefficient
tfidf_200_mlr_mcc <- mcc(preds = as.numeric(predict(tfidf_200_mlr, 
                                                  newx = X_test_tfidf_200, 
                                                  s = cv_lasso_tfidf_200_mlr$lambda.min, 
                                                  type = 'class')),
                       actuals = as.numeric(test_tfidf_200$Pid)) # MCC


##############################################################################################################################
# MLR model on upsampled tfidf bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(US_tfidf_200), size = nrow(US_tfidf_200)*0.7, replace = F)
train_US_tfidf_200 <- US_tfidf_200[ind,-c(2,203)] # 70% training set
train_US_tfidf_200$Pid <- as.factor(train_US_tfidf_200$Pid) # convert target variable to factor for training set
test_US_tfidf_200 <- US_tfidf_200[-ind,-c(2,203)] # 30% training set
test_US_tfidf_200$Pid <- as.factor(test_US_tfidf_200$Pid) # convert target variable to factor for test set

# convert predictor variables into a matrix/array as glmnet only accepts this data type as input
X_train_US_tfidf_200 <- as.matrix(train_US_tfidf_200[,-1])
# convert response variable into a matrix/array as glmnet only accepts matrix data type as input
Y_train_US_tfidf_200 <- as.matrix(train_US_tfidf_200[,1])

X_test_US_tfidf_200 <- as.matrix(test_US_tfidf_200[,-1])

# Fit multinomial logistic regression
US_tfidf_200_mlr <- glmnet(X_train_US_tfidf_200, Y_train_US_tfidf_200, 
                         alpha = 1, # invoke lasso as regularisation method
                         standardize = T, 
                         family = 'multinomial')

# 10-fold CV results for logistic regression with LASSO penalty
set.seed(1)
cv_lasso_US_tfidf_200_mlr <- cv.glmnet(X_train_US_tfidf_200, Y_train_US_tfidf_200,
                                     alpha = 1, # invoke lasso as regularisation method
                                     nfolds = 10, # 10-folds cross validation
                                     type.measure = 'class', # classification problem
                                     standardize = T,
                                     family = 'multinomial')

save(US_tfidf_200_mlr, cv_lasso_US_tfidf_200_mlr, file = "Rdata/MLR_US_tfidf_200.Rdata")

mlr_US_tfidf_200_train_accuracy <- mean(predict(US_tfidf_200_mlr, 
                                              newx = X_train_US_tfidf_200, 
                                              s = cv_lasso_US_tfidf_200_mlr$lambda.min, 
                                              type = 'class') == train_US_tfidf_200$Pid) # training accuracy

mlr_US_tfidf_200_test_accuracy <- mean(predict(US_tfidf_200_mlr, 
                                             newx = X_test_US_tfidf_200, 
                                             s = cv_lasso_US_tfidf_200_mlr$lambda.min, 
                                             type = 'class') == test_US_tfidf_200$Pid) # test accuracy

##############################################################################################################################
# MLR model on downsampled tfidf bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(DS_tfidf_200), size = nrow(DS_tfidf_200)*0.7, replace = F)
train_DS_tfidf_200 <- DS_tfidf_200[ind,-c(2,203)] # 70% training set
train_DS_tfidf_200$Pid <- as.factor(train_DS_tfidf_200$Pid) # convert target variable to factor for training set
test_DS_tfidf_200 <- DS_tfidf_200[-ind,-c(2,203)] # 30% training set
test_DS_tfidf_200$Pid <- as.factor(test_DS_tfidf_200$Pid) # convert target variable to factor for test set

# convert predictor variables into a matrix/array as glmnet only accepts this data type as input
X_train_DS_tfidf_200 <- as.matrix(train_DS_tfidf_200[,-1])
# convert response variable into a matrix/array as glmnet only accepts matrix data type as input
Y_train_DS_tfidf_200 <- as.matrix(train_DS_tfidf_200[,1])

X_test_DS_tfidf_200 <- as.matrix(test_DS_tfidf_200[,-1])

# Fit multinomial logistic regression
DS_tfidf_200_mlr <- glmnet(X_train_DS_tfidf_200, Y_train_DS_tfidf_200, 
                           alpha = 1, # invoke lasso as regularisation method
                           standardize = T, 
                           family = 'multinomial')

# 10-fold CV results for logistic regression with LASSO penalty
set.seed(1)
cv_lasso_DS_tfidf_200_mlr <- cv.glmnet(X_train_DS_tfidf_200, Y_train_DS_tfidf_200,
                                       alpha = 1, # invoke lasso as regularisation method
                                       nfolds = 10, # 10-folds cross validation
                                       type.measure = 'class', # classification problem
                                       standardize = T,
                                       family = 'multinomial')

save(DS_tfidf_200_mlr, cv_lasso_DS_tfidf_200_mlr, file = "Rdata/MLR_DS_tfidf_200.Rdata")

mlr_DS_tfidf_200_train_accuracy <- mean(predict(DS_tfidf_200_mlr, 
                                                newx = X_train_DS_tfidf_200, 
                                                s = cv_lasso_DS_tfidf_200_mlr$lambda.min, 
                                                type = 'class') == train_DS_tfidf_200$Pid) # training accuracy

mlr_DS_tfidf_200_test_accuracy <- mean(predict(DS_tfidf_200_mlr, 
                                               newx = X_test_DS_tfidf_200, 
                                               s = cv_lasso_DS_tfidf_200_mlr$lambda.min, 
                                               type = 'class') == test_DS_tfidf_200$Pid) # test accuracy












