#installing packages & loading data set for experimentation
install.packages("palmerpenguins")
install.packages('class')
install.packages('caret')
install.packages('e1071')
library(caTools)
library(palmerpenguins)
library(class)
library(caret)
library(e1071)
library(dplyr)
library(ggplot2)
data("penguins")
summary(penguins)

#Checking for missing data
any(is.na(penguins$species))
any(is.na(penguins$bill_length_mm))
any(is.na(penguins$bill_depth_mm))
any(is.na(penguins$flipper_length_mm))
any(is.na(penguins$body_mass_g))
any(is.na(penguins$sex))
any(is.na(penguins$year))

#delete missing data
penguins <- na.omit(penguins)

#Impute Numerical Columns with Mean
#https://www.geeksforgeeks.org/how-to-impute-missing-values-in-r/
#penguins$bill_length_mm[is.na(penguins$bill_length_mm)] <- mean(penguins$bill_length_mm, na.rm = TRUE)
#penguins$bill_depth_mm[is.na(penguins$bill_depth_mm)] <- mean(penguins$bill_depth_mm, na.rm = TRUE)
#penguins$flipper_length_mm[is.na(penguins$flipper_length_mm)] <- mean(penguins$flipper_length_mm, na.rm = TRUE)
#penguins$body_mass_g[is.na(penguins$body_mass_g)] <- mean(penguins$body_mass_g, na.rm = TRUE)

#Split dataset
set.seed(12345)
trainrows <-sample(1:nrow(penguins), replace = F, size = nrow(penguins)*0.6)

#creating training sets
train_penguins <- penguins[trainrows, 3:6] #60% training data
train_label <- penguins$species[trainrows] #dataframe for the penguin class
table(train_label)

#test dataset
test_penguins <- penguins[-trainrows, 3:6] #remaining 40%
test_label <- penguins$species[-trainrows] #dataframe for the penguins class
table(test_label)

#Perform PCA
pca_model <- prcomp(train_penguins)
num_pcs <- 2 #choose number of principle components
rotation_matrix <- pca_model$rotation[, 1:num_pcs]
train_penguins_pca <- as.matrix(train_penguins) %*% rotation_matrix
test_penguins_pca <- as.matrix(test_penguins)
print(head(train_penguins_pca))

#build knn using pca
knn_pca_results <- knn(train = train_penguins_pca, test = test_penguins_pca %*% rotation_matrix, cl = train_label, k = 14)

#testing PCA
knn_pca_acc <- mean(knn_pca_results == test_label)
confusionMatrix_knn_pca <- table(knn_pca_results, test_label)

#print results
print(paste("kNN Accuracy on PCA-transformed data: ", knn_pca_acc))
print("Confusion Matrix for kNN on PCA-transformed data: ")
print(confusionMatrix_knn_pca)

#building model for kNN using the square root of the number of rows in training dataset
NROW(train_label)
knn.14 <- knn(train=train_penguins, test=test_penguins, cl=train_label, k=14)
knn.15 <- knn(train=train_penguins, test=test_penguins, cl=train_label, k=15)

#building model for Naïve Bayes
nb_model <- naiveBayes(train_label ~ ., data = train_penguins)

#10-FOLD CROSS VALIDATION
#Defining control parameters for 10-fold cross-validation
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

#Perform cross validation for kNN
knn_results_tfcv <- train(
  x = train_penguins,
  y = train_label,
  method = "knn",
  trControl = ctrl,
  tuneGrid = expand.grid(k = c(14, 15))
)

#Perform cross validation for nb
nb_results_tfcv <- train(
  x = train_penguins,
  y = train_label,
  method = "naive_bayes",
  trControl = ctrl
)

#print results for both
print(knn_results_tfcv)
print(nb_results_tfcv)

#TESTING EACH MODEL USING TEST DATA
#testing kNN model
#https://www.digitalocean.com/community/tutorials/predict-function-in-r
knn_predictions <- predict(knn_results_tfcv, newdata = test_penguins)
knn_acc <- mean(knn_predictions == test_label)
confusionMatrix_knn <- table(knn_predictions, test_label)

print(paste("kNN Accuracy: ", knn_acc))
print("Confusion Matrix for kNN: ")
print(confusionMatrix_knn)

#testing nb model
nb_predictions <- predict(nb_results_tfcv, newdata = test_penguins)
nb_acc <- mean(nb_predictions == test_label)
confusionMatrix_nb <- table(nb_predictions, test_label)

print(paste("Naïve Bayes Accuracy: ", nb_acc))
print("Confusion Matrix for Naïve Bayes: ")
print(confusionMatrix_nb)

#VISUALISATION
# Create a dataframe with actual data
actual_data <- data.frame(
  bill_length_mm = test_penguins$bill_length_mm,
  bill_depth_mm = test_penguins$bill_depth_mm,
  Actual = test_label
)

# Scatterplot using ggplot2
ggplot(data = actual_data, aes(x = bill_length_mm, y = bill_depth_mm, color = Actual)) +
  geom_point() +
  labs(color = "Actual Class") +
  ggtitle("Scatterplot of Actual Data")

#Visualise kNN
#Create a dataframe with test data and predicted classes
test_data_with_predictions_knn <- data.frame(
  bill_length_mm = test_penguins$bill_length_mm,
  bill_depth_mm = test_penguins$bill_depth_mm,
  Predicted_knn = knn_predictions,
  Actual = test_label
)

#Scatterplot using ggplot2 for kNN
ggplot(data = test_data_with_predictions_knn, aes(x = bill_length_mm, y = bill_depth_mm, color = Predicted_knn)) +
  geom_point() +
  labs(color = "Predicted Class") +
  ggtitle("Scatterplot of Data Points with kNN Predictions")

#Visualise nb
#Create a dataframe with test data and predicted classes for Naïve Bayes
test_data_with_predictions_nb <- data.frame(
  bill_length_mm = test_penguins$bill_length_mm,
  bill_depth_mm = test_penguins$bill_depth_mm,
  Predicted_nb = nb_predictions,
  Actual = test_label
)

#Scatterplot using ggplot2
ggplot(data = test_data_with_predictions_nb, aes(x = bill_length_mm, y = bill_depth_mm, color = Predicted_nb)) +
  geom_point() +
  labs(color = "Predicted Class") +
  ggtitle("Scatterplot of Data Points with Naïve Bayes Predictions")



