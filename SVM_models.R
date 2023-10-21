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

svm_bow_200_train_accuracy <- mean(predict(svm_bow_200, newdata = train_bow_200) == train_bow_200$Pid)
svm_bow_200_test_accuracy <- mean(predict(svm_bow_200, newdata = test_bow_200) == test_bow_200$Pid)


svm_bow_200_mcc <-  mcc(preds = as.numeric(predict(svm_bow_200, newdata = test_bow_200)),
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

svm_US_bow_200_train_accuracy <- mean(predict(svm_US_bow_200, newdata = train_US_bow_200) == train_US_bow_200$Pid)
svm_US_bow_200_test_accuracy <- mean(predict(svm_US_bow_200, newdata = test_US_bow_200) == test_US_bow_200$Pid)

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

svm_DS_bow_200_train_accuracy <- mean(predict(svm_DS_bow_200, newdata = train_DS_bow_200) == train_DS_bow_200$Pid)
svm_DS_bow_200_test_accuracy <- mean(predict(svm_DS_bow_200, newdata = test_DS_bow_200) == test_DS_bow_200$Pid)

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

svm_tfidf_200_train_accuracy <- mean(predict(svm_tfidf_200, newdata = train_tfidf_200) == train_tfidf_200$Pid)
svm_tfidf_200_test_accuracy <- mean(predict(svm_tfidf_200, newdata = test_tfidf_200) == test_tfidf_200$Pid)


svm_tfidf_200_mcc <-  mcc(preds = as.numeric(predict(svm_tfidf_200, newdata = test_tfidf_200)),
                        actuals = as.numeric(test_tfidf_200$Pid))

##############################################################################################################################
# MLR model on upsampled tfidf bag of words model -------------------------------
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

svm_US_tfidf_200_train_accuracy <- mean(predict(svm_US_tfidf_200, newdata = train_US_tfidf_200) == train_US_tfidf_200$Pid)
svm_US_tfidf_200_test_accuracy <- mean(predict(svm_US_tfidf_200, newdata = test_US_tfidf_200) == test_US_tfidf_200$Pid)

##############################################################################################################################
# MLR model on downsampled tfidf bag of words model -------------------------------
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

svm_DS_tfidf_200_train_accuracy <- mean(predict(svm_DS_tfidf_200, newdata = train_DS_tfidf_200) == train_DS_tfidf_200$Pid)
svm_DS_tfidf_200_test_accuracy <- mean(predict(svm_DS_tfidf_200, newdata = test_DS_tfidf_200) == test_DS_tfidf_200$Pid)
