#Tuğberk Dikmen
#21802480
#HW1 Question 3

import numpy as np
import pandas as pd

# Train Multinomial nb model
def multino_train_nb(train_x, train_y, smoothing_param=0):
    class_prior = np.array([np.sum(train_y == class_val) for class_val in range(np.unique(train_y).size)]) / train_y.size

    #calculating conditional probability
    conditional_prob = np.array([(train_x[train_y == class_val].sum(axis=0) + smoothing_param).astype(float) 
                                 for class_val in range(np.unique(train_y).size)])

    conditional_prob /= (conditional_prob.sum(axis=1, keepdims=True) + smoothing_param * train_x.shape[1])
    log_class_prior = np.log(class_prior)
    log_conditional_prob = np.log(conditional_prob)
    
    return log_class_prior, log_conditional_prob

# Train bernoulli nb
def bernu_train_nb(train_x, train_y, smoothing_param=1):
    class_prior = np.array([np.sum(train_y == class_val) for class_val in range(np.unique(train_y).size)]) / train_y.size
    
    # Applying additive smoothing
    total_docs_per_class = np.array([train_x[train_y == class_val].shape[0] + 2 * smoothing_param 
                                     for class_val in range(np.unique(train_y).size)])
    conditional_prob = np.array([(train_x[train_y == class_val].sum(axis=0) + smoothing_param).astype(float) 
                                 for class_val in range(np.unique(train_y).size)])
    
    conditional_prob /= total_docs_per_class[:, None]
    
    log_class_prior = np.log(class_prior)
    log_conditional_prob = np.log(conditional_prob)
    log_complement_conditional_prob = np.log(1 - conditional_prob)
    
    return log_class_prior, log_conditional_prob, log_complement_conditional_prob

# Make predictions using multinomial nb
def multino_predict_nb(test_x, log_class_prior, log_conditional_prob):
    prediction_scores = log_class_prior + test_x.dot(log_conditional_prob.T)
    return np.argmax(prediction_scores, axis=1)

# Predict with bernoulli nb
def bernu_predict_nb(test_x, log_class_prior, log_conditional_prob, log_complement_conditional_prob):
    prediction_scores = log_class_prior + np.dot(test_x, log_conditional_prob.T) + np.dot(1 - test_x, log_complement_conditional_prob.T)
    return np.argmax(prediction_scores, axis=1)

# to find accuracy
def getAccuracy(actual, predicted):
    model_accuracy = np.mean(predicted == actual)
    return model_accuracy

# to find confusion matrix
def getMatrix(actual, predicted):
    confusion_matrix = pd.crosstab(pd.Series(actual, name='Actual (ROW)'), pd.Series(predicted, name='Predicted (COLUMN)'))
    return confusion_matrix

# Loading and processing data
# Load dataset for Multinomial Naive Bayes classifier
#upload the training data
multinomial_train_x = pd.read_csv('x_train.csv', delimiter=' ').values
multinomial_train_y = pd.read_csv('y_train.csv', delimiter=' ', header=None).values.flatten()
#upload the testing data
multinomial_test_x = pd.read_csv('x_test.csv', delimiter=' ').values
multinomial_test_y = pd.read_csv('y_test.csv', delimiter=' ', header=None).values.flatten()

# Multinomial Naive Bayes without smoothing
log_prior_mnb, log_likelihoods_mnb = multino_train_nb(multinomial_train_x, multinomial_train_y)
predictions_mnb = multino_predict_nb(multinomial_test_x, log_prior_mnb, log_likelihoods_mnb)

# Multinomial Naive Bayes with smoothing
log_prior_mnb_smoothed, log_likelihoods_mnb_smoothed = multino_train_nb(multinomial_train_x, multinomial_train_y, smoothing_param=1)
predictions_mnb_smoothed = multino_predict_nb(multinomial_test_x, log_prior_mnb_smoothed, log_likelihoods_mnb_smoothed)

# Preparing data for Bernoulli Naive Bayes
# Upload data for Bernoulli Naive Bayes classifier
bernu_train_x = pd.read_csv('x_train.csv', delimiter=' ').values
bernu_train_y = pd.read_csv('y_train.csv', delimiter=' ', header=None).values.flatten()
bernu_test_x = pd.read_csv('x_test.csv', delimiter=' ').values
bernu_test_y = pd.read_csv('y_test.csv', delimiter=' ', header=None).values.flatten()

# Binarize data: 1 for word presence, 0 for absence
bernu_train_x = np.where(bernu_train_x > 0, 1, 0)
bernu_test_x = np.where(bernu_test_x > 0, 1, 0)

# Training Bernoulli Naive Bayes
prior_bernu, ll_bernu, comp_ll_bernu = bernu_train_nb(bernu_train_x, bernu_train_y)
predictions_bnb = bernu_predict_nb(bernu_test_x, prior_bernu, ll_bernu, comp_ll_bernu)

# The Results Taken
#unsmoothed multinomial naive bias results
print("----------------------------------------")
print("Unsmoothed Multinomial Naive Bayes ")
print(f"Accuracy: {getAccuracy(multinomial_test_y, predictions_mnb):.3f}\n")
print("Confusion Matrix")
print(getMatrix(multinomial_test_y, predictions_mnb))
print("----------------------------------------\n")

#smoothed multinomial naive bias results 
print("\nSmoothed Multinomial Naive Bayes ")
print(f"Accuracy: {getAccuracy(multinomial_test_y, predictions_mnb_smoothed):.3f}\n")
print("Confusion Matrix")
print(getMatrix(multinomial_test_y, predictions_mnb_smoothed))
print("----------------------------------------\n")

#bernoulli naive bieas results
print("\nBernoulli Naive Bayes")
print(f"Accuracy: {getAccuracy(bernu_test_y, predictions_bnb):.3f}\n")
print("Confusion Matrix")
print(getMatrix(bernu_test_y, predictions_bnb))
print("----------------------------------------\n")