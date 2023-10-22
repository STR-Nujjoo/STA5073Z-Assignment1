library(e1071)
# SVM model on original bag of words model --------------------------------
set.seed(1)
ind <- sample(1:nrow(bow_200), size = nrow(bow_200)*0.7, replace = F)
train_bow_200 <- bow_200[ind,-2] # 70% training set
train_bow_200$Pid <- as.factor(train_bow_200$Pid) # convert target variable to factor for training set
test_bow_200 <- bow_200[-ind,-2] # 30% training set
test_bow_200$Pid <- as.factor(test_bow_200$Pid) # convert target variable to factor for test set

svm_bow_200 <- svm(Pid ~., 
    data = train_bow_200,
    type = 'C-classification',
    kernel = 'radial', 
    scale = T, 
    cost = 1)

save(svm_bow_200, file = "Rdata/SVM_bow_200.Rdata")

svm_bow_200_trainpred <- predict(svm_bow_200, newdata = train_bow_200)
svm_bow_200_testpred <- predict(svm_bow_200, newdata = test_bow_200)
svm_bow_200_train_accuracy <- mean(svm_bow_200_trainpred == train_bow_200$Pid)
svm_bow_200_test_accuracy <- mean(svm_bow_200_testpred == test_bow_200$Pid)


svm_bow_200_mcc <-  mcc(preds = as.numeric(svm_bow_200_testpred),
                        actuals = as.numeric(test_bow_200$Pid))

##############################################################################################################################
# SVM model on upsampled bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(US_bow_200), size = nrow(US_bow_200)*0.7, replace = F)
train_US_bow_200 <- US_bow_200[ind,-c(2,203)] # 70% training set
train_US_bow_200$Pid <- as.factor(train_US_bow_200$Pid) # convert target variable to factor for training set
test_US_bow_200 <- US_bow_200[-ind,-c(2,203)] # 30% training set
test_US_bow_200$Pid <- as.factor(test_US_bow_200$Pid) # convert target variable to factor for test set

svm_US_bow_200 <- svm(Pid ~., 
                   data = train_US_bow_200,
                   type = 'C-classification',
                   kernel = 'radial', 
                   scale = T, 
                   cost = 1)

save(svm_US_bow_200, file = "Rdata/SVM_US_bow_200.Rdata")

svm_US_bow_200_trainpred <- predict(svm_US_bow_200, newdata = train_US_bow_200)
svm_US_bow_200_testpred <- predict(svm_US_bow_200, newdata = test_US_bow_200)
svm_US_bow_200_train_accuracy <- mean(svm_US_bow_200_trainpred == train_US_bow_200$Pid)
svm_US_bow_200_test_accuracy <- mean(svm_US_bow_200_testpred == test_US_bow_200$Pid)

##############################################################################################################################
# SVM model on downsampled bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(DS_bow_200), size = nrow(DS_bow_200)*0.7, replace = F)
train_DS_bow_200 <- DS_bow_200[ind,-c(2,203)] # 70% training set
train_DS_bow_200$Pid <- as.factor(train_DS_bow_200$Pid) # convert target variable to factor for training set
test_DS_bow_200 <- DS_bow_200[-ind,-c(2,203)] # 30% training set
test_DS_bow_200$Pid <- as.factor(test_DS_bow_200$Pid) # convert target variable to factor for test set

svm_DS_bow_200 <- svm(Pid ~., 
                      data = train_DS_bow_200,
                      type = 'C-classification',
                      kernel = 'radial', 
                      scale = T, 
                      cost = 1)

save(svm_DS_bow_200, file = "Rdata/SVM_DS_bow_200.Rdata")

svm_DS_bow_200_trainpred <- predict(svm_DS_bow_200, newdata = train_DS_bow_200)
svm_DS_bow_200_testpred <- predict(svm_DS_bow_200, newdata = test_DS_bow_200)
svm_DS_bow_200_train_accuracy <- mean(svm_DS_bow_200_trainpred == train_DS_bow_200$Pid)
svm_DS_bow_200_test_accuracy <- mean(svm_DS_bow_200_testpred == test_DS_bow_200$Pid)

##############################################################################################################################
# SVM model on original tfidf bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(tfidf_200), size = nrow(tfidf_200)*0.7, replace = F)
train_tfidf_200 <- tfidf_200[ind,-2] # 70% training set
train_tfidf_200$Pid <- as.factor(train_tfidf_200$Pid) # convert target variable to factor for training set
test_tfidf_200 <- tfidf_200[-ind,-2] # 30% training set
test_tfidf_200$Pid <- as.factor(test_tfidf_200$Pid) # convert target variable to factor for test set

svm_tfidf_200 <- svm(Pid ~., 
                   data = train_tfidf_200,
                   type = 'C-classification',
                   kernel = 'radial', 
                   scale = T, 
                   cost = 1)

save(svm_tfidf_200, file = "Rdata/SVM_tfidf_200.Rdata")

svm_tfidf_200_trainpred <- predict(svm_tfidf_200, newdata = train_tfidf_200)
svm_tfidf_200_testpred <- predict(svm_tfidf_200, newdata = test_tfidf_200)
svm_tfidf_200_train_accuracy <- mean(svm_tfidf_200_trainpred == train_tfidf_200$Pid)
svm_tfidf_200_test_accuracy <- mean(svm_tfidf_200_testpred == test_tfidf_200$Pid)


svm_tfidf_200_mcc <-  mcc(preds = as.numeric(svm_tfidf_200_testpred),
                        actuals = as.numeric(test_tfidf_200$Pid))

##############################################################################################################################
# SVM model on upsampled tfidf bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(US_tfidf_200), size = nrow(US_tfidf_200)*0.7, replace = F)
train_US_tfidf_200 <- US_tfidf_200[ind,-c(2,203)] # 70% training set
train_US_tfidf_200$Pid <- as.factor(train_US_tfidf_200$Pid) # convert target variable to factor for training set
test_US_tfidf_200 <- US_tfidf_200[-ind,-c(2,203)] # 30% training set
test_US_tfidf_200$Pid <- as.factor(test_US_tfidf_200$Pid) # convert target variable to factor for test set

svm_US_tfidf_200 <- svm(Pid ~., 
                     data = train_US_tfidf_200,
                     type = 'C-classification',
                     kernel = 'radial', 
                     scale = T, 
                     cost = 1)

save(svm_US_tfidf_200, file = "Rdata/SVM_US_tfidf_200.Rdata")

svm_US_tfidf_200_trainpred <- predict(svm_US_tfidf_200, newdata = train_US_tfidf_200)
svm_US_tfidf_200_testpred <- predict(svm_US_tfidf_200, newdata = test_US_tfidf_200)
svm_US_tfidf_200_train_accuracy <- mean(svm_US_tfidf_200_trainpred == train_US_tfidf_200$Pid)
svm_US_tfidf_200_test_accuracy <- mean(svm_US_tfidf_200_testpred == test_US_tfidf_200$Pid)

##############################################################################################################################
# SVM model on downsampled tfidf bag of words model -------------------------------
# randomly splitting data
set.seed(1)
ind <- sample(1:nrow(DS_tfidf_200), size = nrow(DS_tfidf_200)*0.7, replace = F)
train_DS_tfidf_200 <- DS_tfidf_200[ind,-c(2,203)] # 70% training set
train_DS_tfidf_200$Pid <- as.factor(train_DS_tfidf_200$Pid) # convert target variable to factor for training set
test_DS_tfidf_200 <- DS_tfidf_200[-ind,-c(2,203)] # 30% training set
test_DS_tfidf_200$Pid <- as.factor(test_DS_tfidf_200$Pid) # convert target variable to factor for test set

svm_DS_tfidf_200 <- svm(Pid ~., 
                        data = train_DS_tfidf_200,
                        type = 'C-classification',
                        kernel = 'radial', 
                        scale = T, 
                        cost = 1)

save(svm_DS_tfidf_200, file = "Rdata/SVM_DS_tfidf_200.Rdata")

svm_DS_tfidf_200_trainpred <- predict(svm_DS_tfidf_200, newdata = train_DS_tfidf_200)
svm_DS_tfidf_200_testpred <- predict(svm_DS_tfidf_200, newdata = test_DS_tfidf_200)
svm_DS_tfidf_200_train_accuracy <- mean(svm_DS_tfidf_200_trainpred == train_DS_tfidf_200$Pid)
svm_DS_tfidf_200_test_accuracy <- mean(svm_DS_tfidf_200_testpred == test_DS_tfidf_200$Pid)


# save all predictions
save(svm_bow_200_trainpred, svm_bow_200_testpred, 
     svm_US_bow_200_trainpred, svm_US_bow_200_testpred, 
     svm_DS_bow_200_trainpred, svm_DS_bow_200_testpred, 
     svm_tfidf_200_trainpred, svm_tfidf_200_testpred,
     svm_US_tfidf_200_trainpred, svm_US_tfidf_200_testpred,
     svm_DS_tfidf_200_trainpred, svm_DS_tfidf_200_testpred, 
     file = "Rdata/SVM_predictions.Rdata")
