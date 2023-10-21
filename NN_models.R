# Fit NN on original bag of words model -----------------------------------
# Preprocessing for NN model
bow_200_target <- bow_200$Pid
bow_200_features <- as.matrix(bow_200[,-c(1:2)])

# Split data into train and test ------------------------------------------
# Determine sample size
set.seed(1)
ind <-  sample(1:2, nrow(bow_200), replace=TRUE, prob=c(0.7, 0.3))

# Split features
x_train_bow_200 <- bow_200_features[ind==1, ]
x_test_bow_200 <- bow_200_features[ind==2, ]
str(x_train_bow_200)
# Split target
y_train_bow_200 <- bow_200_target[ind==1]
y_test_bow_200 <- bow_200_target[ind==2]

# Scale dataset -----------------------------------------------------------
x_train_bow_200 <- scale(x_train_bow_200)

# Scale test data based on training data means and std devs
x_test_bow_200 <- scale(x_test_bow_200, center = attr(x_train_bow_200, "scaled:center"), 
                scale = attr(x_train_bow_200, "scaled:scale"))

# One hot encoding --------------------------------------------------------
y_train_bow_200 <- to_categorical(y_train_bow_200)
y_test_bow_200_original <- y_test_bow_200
y_test_bow_200 <- to_categorical(y_test_bow_200)


# Neural Network Model ----------------------------------------------------
set.seed(1)
# create empty model
model_bow_200 <- keras_model_sequential()

# define model
model_bow_200 %>% 
  layer_dense(units = 500, activation = 'tanh', input_shape = c(200)) %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 300, activation = 'tanh') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model_bow_200)

# compile model
model_bow_200 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)
set.seed(1)

# Train the model
history_bow_200 <- model_bow_200 %>% fit(
  x_train_bow_200, y_train_bow_200, 
  epochs = 100, batch_size = 30, 
  validation_split = 0.2, shuffle = TRUE
)

save(history_bow_200, file = "Rdata/NN_bow_200.Rdata")

plot(history_bow_200)
NN_bow_200_train_accuracy <- mean(history_bow_200$metrics$accuracy) # training accuracy of NN on original BOW model
NN_bow_200_val_accuracy <- mean(history_bow_200$metrics$val_accuracy) # validation accuracy of NN on original BOW model


# model evaluation
NN_bow_200_test_accuracy <- model_bow_200 %>% evaluate(x_test_bow_200, y_test_bow_200) # test accuracy of NN on original BOW model

# confusion matrix
y_test_bow_200_hat <- model_bow_200 %>% predict(x_test_bow_200) %>% k_argmax() %>% as.numeric()
table(y_test_bow_200_original, y_test_bow_200_hat)

# Matthew's correlation coefficient
MCC_bow_200 <- mcc(preds = y_test_bow_200_hat, actuals = y_test_bow_200_original)

#######################################################################################################################
# Fit NN on upsampled bag of words model ----------------------------------
# Preprocessing for NN model
US_bow_200_target <- US_bow_200$Pid
US_bow_200_features <- as.matrix(US_bow_200[,-c(1:2,203)])

# Split data into train and test ------------------------------------------
# Determine sample size
set.seed(1)
ind <-  sample(1:2, nrow(US_bow_200), replace=TRUE, prob=c(0.7, 0.3))

# Split features
x_train_US_bow_200 <- US_bow_200_features[ind==1, ]
x_test_US_bow_200 <- US_bow_200_features[ind==2, ]
str(x_train_US_bow_200)
# Split target
y_train_US_bow_200 <- US_bow_200_target[ind==1]
y_test_US_bow_200 <- US_bow_200_target[ind==2]

# Scale dataset -----------------------------------------------------------
x_train_US_bow_200 <- scale(x_train_US_bow_200)

# Scale test data based on training data means and std devs
x_test_US_bow_200 <- scale(x_test_US_bow_200, center = attr(x_train_US_bow_200, "scaled:center"), 
                        scale = attr(x_train_US_bow_200, "scaled:scale"))

# One hot encoding --------------------------------------------------------
y_train_US_bow_200 <- to_categorical(y_train_US_bow_200)
y_test_US_bow_200_original <- y_test_US_bow_200
y_test_US_bow_200 <- to_categorical(y_test_US_bow_200)


# Neural Network Model ----------------------------------------------------
set.seed(1)
# create empty model
model_US_bow_200 <- keras_model_sequential()

# define model
model_US_bow_200 %>% 
  layer_dense(units = 500, activation = 'tanh', input_shape = c(200)) %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 300, activation = 'tanh') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model_US_bow_200)

# compile model
model_US_bow_200 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)
set.seed(1)

# Train the model
history_US_bow_200 <- model_US_bow_200 %>% fit(
  x_train_US_bow_200, y_train_US_bow_200, 
  epochs = 100, batch_size = 30, 
  validation_split = 0.2, shuffle = TRUE
)

save(history_US_bow_200, file = "Rdata/NN_US_bow_200.Rdata")

plot(history_US_bow_200)
NN_US_bow_200_train_accuracy <- mean(history_US_bow_200$metrics$accuracy) # training accuracy of NN on upsampled BOW model
NN_US_bow_200_val_accuracy <- mean(history_US_bow_200$metrics$val_accuracy) # validation accuracy of NN on upsampled BOW model

# model evaluation
NN_US_bow_200_test_accuracy <- model_US_bow_200 %>% evaluate(x_test_US_bow_200, y_test_US_bow_200) # test accuracy of NN on upsampled BOW model

# confusion matrix
y_test_US_bow_200_hat <- model_US_bow_200 %>% predict(x_test_US_bow_200) %>% k_argmax() %>% as.numeric()
table(y_test_US_bow_200_original, y_test_US_bow_200_hat)


#######################################################################################################################
# Fit NN on downsampled bag of words model ----------------------------------
# Preprocessing for NN model
DS_bow_200_target <- DS_bow_200$Pid
DS_bow_200_features <- as.matrix(DS_bow_200[,-c(1:2,203)])

# Split data into train and test ------------------------------------------
# Determine sample size
set.seed(1)
ind <-  sample(1:2, nrow(DS_bow_200), replace=TRUE, prob=c(0.7, 0.3))

# Split features
x_train_DS_bow_200 <- DS_bow_200_features[ind==1, ]
x_test_DS_bow_200 <- DS_bow_200_features[ind==2, ]
str(x_train_DS_bow_200)
# Split target
y_train_DS_bow_200 <- DS_bow_200_target[ind==1]
y_test_DS_bow_200 <- DS_bow_200_target[ind==2]

# Scale dataset -----------------------------------------------------------
x_train_DS_bow_200 <- scale(x_train_DS_bow_200)

# Scale test data based on training data means and std devs
x_test_DS_bow_200 <- scale(x_test_DS_bow_200, center = attr(x_train_DS_bow_200, "scaled:center"), 
                           scale = attr(x_train_DS_bow_200, "scaled:scale"))

# One hot encoding --------------------------------------------------------
y_train_DS_bow_200 <- to_categorical(y_train_DS_bow_200)
y_test_DS_bow_200_original <- y_test_DS_bow_200
y_test_DS_bow_200 <- to_categorical(y_test_DS_bow_200)


# Neural Network Model ----------------------------------------------------
set.seed(1)
# create empty model
model_DS_bow_200 <- keras_model_sequential()

# define model
model_DS_bow_200 %>% 
  layer_dense(units = 500, activation = 'tanh', input_shape = c(200)) %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 300, activation = 'tanh') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model_DS_bow_200)

# compile model
model_DS_bow_200 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)
set.seed(1)

# Train the model
history_DS_bow_200 <- model_DS_bow_200 %>% fit(
  x_train_DS_bow_200, y_train_DS_bow_200, 
  epochs = 100, batch_size = 30, 
  validation_split = 0.2, shuffle = TRUE
)

save(history_DS_bow_200, file = "Rdata/NN_DS_bow_200.Rdata")

plot(history_DS_bow_200)
NN_DS_bow_200_train_accuracy <- mean(history_DS_bow_200$metrics$accuracy) # training accuracy of NN on downsampled BOW model
NN_DS_bow_200_val_accuracy <- mean(history_DS_bow_200$metrics$val_accuracy)  # validation accuracy of NN on downsampled BOW model

# model evaluation
NN_DS_bow_200_test_accuracy <- model_DS_bow_200 %>% evaluate(x_test_DS_bow_200, y_test_DS_bow_200) # test accuracy of NN on downsampled BOW model

# confusion matrix
y_test_DS_bow_200_hat <- model_DS_bow_200 %>% predict(x_test_DS_bow_200) %>% k_argmax() %>% as.numeric()
table(y_test_DS_bow_200_original, y_test_DS_bow_200_hat)

############################################################################################################################
# Fit NN on original tfidf bag of words model -----------------------------
# Preprocessing for NN model
tfidf_200_target <- tfidf_200$Pid
tfidf_200_features <- as.matrix(tfidf_200[,-c(1:2)])

# Split data into train and test ------------------------------------------
# Determine sample size
set.seed(1)
ind <-  sample(1:2, nrow(tfidf_200), replace=TRUE, prob=c(0.7, 0.3))

# Split features
x_train_tfidf_200 <- tfidf_200_features[ind==1, ]
x_test_tfidf_200 <- tfidf_200_features[ind==2, ]

# Split target
y_train_tfidf_200 <- tfidf_200_target[ind==1]
y_test_tfidf_200 <- tfidf_200_target[ind==2]

# Scale dataset -----------------------------------------------------------
x_train_tfidf_200 <- scale(x_train_tfidf_200)

# Scale test data based on training data means and std devs
x_test_tfidf_200 <- scale(x_test_tfidf_200, center = attr(x_train_tfidf_200, "scaled:center"), 
                        scale = attr(x_train_tfidf_200, "scaled:scale"))

# One hot encoding --------------------------------------------------------
y_train_tfidf_200 <- to_categorical(y_train_tfidf_200)
y_test_tfidf_200_original <- y_test_tfidf_200
y_test_tfidf_200 <- to_categorical(y_test_tfidf_200)


# Neural Network Model ----------------------------------------------------
set.seed(1)
# create empty model
model_tfidf_200 <- keras_model_sequential()

# define model
model_tfidf_200 %>% 
  layer_dense(units = 500, activation = 'tanh', input_shape = c(200)) %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 300, activation = 'tanh') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model_tfidf_200)

# compile model
model_tfidf_200 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)
set.seed(1)

# Train the model
history_tfidf_200 <- model_tfidf_200 %>% fit(
  x_train_tfidf_200, y_train_tfidf_200, 
  epochs = 100, batch_size = 30, 
  validation_split = 0.2, shuffle = TRUE
)

save(history_tfidf_200, file = "Rdata/NN_tfidf_200.Rdata")

plot(history_tfidf_200)
NN_tfidf_200_train_accuracy <- mean(history_tfidf_200$metrics$accuracy) # training accuracy of NN on original tfidf model
NN_tfidf_200_val_accuracy <- mean(history_tfidf_200$metrics$val_accuracy) # validation accuracy of NN on original tfidf model


# model evaluation
NN_tfidf_200_test_accuracy <- model_tfidf_200 %>% evaluate(x_test_tfidf_200, y_test_tfidf_200) # test accuracy of NN on original tfidf model

# confusion matrix
y_test_tfidf_200_hat <- model_tfidf_200 %>% predict(x_test_tfidf_200) %>% k_argmax() %>% as.numeric()
table(y_test_tfidf_200_original, y_test_tfidf_200_hat)

# Matthew's correlation coefficient
MCC_tfidf_200 <- mcc(preds = y_test_tfidf_200_hat, actuals = y_test_tfidf_200_original)

#######################################################################################################################
# Fit NN on upsampled tfidf bag of words model ----------------------------
# Preprocessing for NN model
US_tfidf_200_target <- US_tfidf_200$Pid
US_tfidf_200_features <- as.matrix(US_tfidf_200[,-c(1:2,203)])

# Split data into train and test ------------------------------------------
# Determine sample size
set.seed(1)
ind <-  sample(1:2, nrow(US_tfidf_200), replace=TRUE, prob=c(0.7, 0.3))

# Split features
x_train_US_tfidf_200 <- US_tfidf_200_features[ind==1, ]
x_test_US_tfidf_200 <- US_tfidf_200_features[ind==2, ]
str(x_train_US_tfidf_200)
# Split target
y_train_US_tfidf_200 <- US_tfidf_200_target[ind==1]
y_test_US_tfidf_200 <- US_tfidf_200_target[ind==2]

# Scale dataset -----------------------------------------------------------
x_train_US_tfidf_200 <- scale(x_train_US_tfidf_200)

# Scale test data based on training data means and std devs
x_test_US_tfidf_200 <- scale(x_test_US_tfidf_200, center = attr(x_train_US_tfidf_200, "scaled:center"), 
                           scale = attr(x_train_US_tfidf_200, "scaled:scale"))

# One hot encoding --------------------------------------------------------
y_train_US_tfidf_200 <- to_categorical(y_train_US_tfidf_200)
y_test_US_tfidf_200_original <- y_test_US_tfidf_200
y_test_US_tfidf_200 <- to_categorical(y_test_US_tfidf_200)


# Neural Network Model ----------------------------------------------------
set.seed(1)
# create empty model
model_US_tfidf_200 <- keras_model_sequential()

# define model
model_US_tfidf_200 %>% 
  layer_dense(units = 500, activation = 'tanh', input_shape = c(200)) %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 300, activation = 'tanh') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model_US_tfidf_200)

# compile model
model_US_tfidf_200 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)
set.seed(1)

# Train the model
history_US_tfidf_200 <- model_US_tfidf_200 %>% fit(
  x_train_US_tfidf_200, y_train_US_tfidf_200, 
  epochs = 100, batch_size = 30, 
  validation_split = 0.2, shuffle = TRUE
)

save(history_US_tfidf_200, file = "Rdata/NN_US_tfidf_200.Rdata")

plot(history_US_tfidf_200)
NN_US_tfidf_200_train_accuracy <- mean(history_US_tfidf_200$metrics$accuracy) # training accuracy of NN on upsampled tfidf model
NN_US_tfidf_200_val_accuracy <- mean(history_US_tfidf_200$metrics$val_accuracy) # validation accuracy of NN on upsampled tfidf model

# model evaluation
NN_US_tfidf_200_test_accuracy <- model_US_tfidf_200 %>% evaluate(x_test_US_tfidf_200, y_test_US_tfidf_200) # test accuracy of NN on upsampled tfidf model

# confusion matrix
y_test_US_tfidf_200_hat <- model_US_tfidf_200 %>% predict(x_test_US_tfidf_200) %>% k_argmax() %>% as.numeric()
table(y_test_US_tfidf_200_original, y_test_US_tfidf_200_hat)

#######################################################################################################################
# Fit NN on downsampled tfidf bag of words model ----------------------------
# Preprocessing for NN model
DS_tfidf_200_target <- DS_tfidf_200$Pid
DS_tfidf_200_features <- as.matrix(DS_tfidf_200[,-c(1:2,203)])

# Split data into train and test ------------------------------------------
# Determine sample size
set.seed(1)
ind <-  sample(1:2, nrow(DS_tfidf_200), replace=TRUE, prob=c(0.7, 0.3))

# Split features
x_train_DS_tfidf_200 <- DS_tfidf_200_features[ind==1, ]
x_test_DS_tfidf_200 <- DS_tfidf_200_features[ind==2, ]
str(x_train_DS_tfidf_200)
# Split target
y_train_DS_tfidf_200 <- DS_tfidf_200_target[ind==1]
y_test_DS_tfidf_200 <- DS_tfidf_200_target[ind==2]

# Scale dataset -----------------------------------------------------------
x_train_DS_tfidf_200 <- scale(x_train_DS_tfidf_200)

# Scale test data based on training data means and std devs
x_test_DS_tfidf_200 <- scale(x_test_DS_tfidf_200, center = attr(x_train_DS_tfidf_200, "scaled:center"), 
                             scale = attr(x_train_DS_tfidf_200, "scaled:scale"))

# One hot encoding --------------------------------------------------------
y_train_DS_tfidf_200 <- to_categorical(y_train_DS_tfidf_200)
y_test_DS_tfidf_200_original <- y_test_DS_tfidf_200
y_test_DS_tfidf_200 <- to_categorical(y_test_DS_tfidf_200)


# Neural Network Model ----------------------------------------------------
set.seed(1)
# create empty model
model_DS_tfidf_200 <- keras_model_sequential()

# define model
model_DS_tfidf_200 %>% 
  layer_dense(units = 500, activation = 'tanh', input_shape = c(200)) %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 300, activation = 'tanh') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model_DS_tfidf_200)

# compile model
model_DS_tfidf_200 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)
set.seed(1)

# Train the model
history_DS_tfidf_200 <- model_DS_tfidf_200 %>% fit(
  x_train_DS_tfidf_200, y_train_DS_tfidf_200, 
  epochs = 100, batch_size = 30, 
  validation_split = 0.2, shuffle = TRUE
)

save(history_DS_tfidf_200, file = "Rdata/NN_DS_tfidf_200.Rdata")

plot(history_DS_tfidf_200)
NN_DS_tfidf_200_train_accuracy <- mean(history_DS_tfidf_200$metrics$accuracy) # training accuracy of NN on downsampled tfidf model
NN_DS_tfidf_200_val_accuracy <- mean(history_DS_tfidf_200$metrics$val_accuracy) # validation accuracy of NN on downsampled tfidf model

# model evaluation
NN_DS_tfidf_200_test_accuracy <- model_DS_tfidf_200 %>% evaluate(x_test_DS_tfidf_200, y_test_DS_tfidf_200) # test accuracy of NN on upsampled tfidf model

# confusion matrix
y_test_DS_tfidf_200_hat <- model_DS_tfidf_200 %>% predict(x_test_DS_tfidf_200) %>% k_argmax() %>% as.numeric()
table(y_test_DS_tfidf_200_original, y_test_DS_tfidf_200_hat)



