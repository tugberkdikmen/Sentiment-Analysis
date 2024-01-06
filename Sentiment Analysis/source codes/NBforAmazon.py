import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from sklearn.utils import resample  # Import resample from scikit-learn
import matplotlib.pyplot as plt

# Download NLTK stopwords (if not already downloaded)
import nltk
nltk.download('stopwords')

# Reading and processing the JSON file
file_path_new = 'amazon/Musical_Instruments_5.json'

# Load JSON data line by line
data_json = []
with open(file_path_new, 'r', encoding='utf-8') as file:
    for line in file:
        data_json.append(json.loads(line))

# Convert list of JSON objects to DataFrame
df = pd.DataFrame(data_json)

# Extract relevant columns
data = df[['reviewText', 'overall']]
data = data.dropna(subset=['reviewText'])

# Define the sentiment rating function
def sentiment_rating(score):
    # Mapping scores to sentiment categories
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    elif score in [4, 5]:
        return 'positive'

# Apply the sentiment rating function to the 'overall' column
data['Sentiment'] = data['overall'].apply(sentiment_rating)


# Print the updated value counts
print("Number of data points in each sentiment category before upsampling:")
print(data['Sentiment'].value_counts())

# Text Vectorization
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['reviewText'])

# Label Encoding
y = data['Sentiment']

# Splitting Data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation on Validation Set
val_predictions = model.predict(X_val)
print("Validation Set Accuracy:", accuracy_score(y_val, val_predictions))
print("Validation Set Classification Report:")
print(classification_report(y_val, val_predictions))

# Model Evaluation on Test Set
test_predictions = model.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, test_predictions))
print("Test Set Classification Report:")
print(classification_report(y_test, test_predictions))

# Print overall sizes for initial case
total_samples_initial = len(data)
train_percentage_initial = X_train.shape[0] / total_samples_initial * 100
val_percentage_initial = X_val.shape[0] / total_samples_initial * 100
test_percentage_initial = X_test.shape[0] / total_samples_initial * 100

print("\nInitial Sizes:")
print("Train Set Percentage:", train_percentage_initial)
print("Validation Set Percentage:", val_percentage_initial)
print("Test Set Percentage:", test_percentage_initial)

# Oversample the minority classes (Sentiment 0 and Sentiment 2)
data_majority = data[data['Sentiment'] == 'positive']
data_minority_negative = data[data['Sentiment'] == 'negative']
data_minority_neutral = data[data['Sentiment'] == 'neutral']

# Upsample minority classes
data_minority_negative_upsampled = resample(data_minority_negative, replace=True, n_samples=len(data_majority), random_state=42)
data_minority_neutral_upsampled = resample(data_minority_neutral, replace=True, n_samples=len(data_majority), random_state=42)

# Combine the upsampled minority classes with the majority class
data_upsampled = pd.concat([data_majority, data_minority_negative_upsampled, data_minority_neutral_upsampled])

# Print the number of data points for each sentiment category in the upsampled dataset
print("\nNumber of data points in each sentiment category after upsampling:")
print(data_upsampled['Sentiment'].value_counts())

# Text Vectorization
stop = set(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words=list(stop))
X = vectorizer.fit_transform(data_upsampled['reviewText'])

# Label Encoding
y = data_upsampled['Sentiment']

# Splitting Data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation on Validation Set
val_predictions = model.predict(X_val)
print("\nValidation Set Accuracy:", accuracy_score(y_val, val_predictions))
print("Validation Set Classification Report:")
print(classification_report(y_val, val_predictions))

# Model Evaluation on Test Set
test_predictions = model.predict(X_test)
print("\nTest Set Accuracy:", accuracy_score(y_test, test_predictions))
print("Test Set Classification Report:")
print(classification_report(y_test, test_predictions))

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

# Binarize the labels for ROC calculation
y_test_bin = label_binarize(y_test, classes=['negative', 'neutral', 'positive'])
n_classes = y_test_bin.shape[1]

# Predict probabilities
y_score = model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(['negative', 'neutral', 'positive'][i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
pred_classes = model.predict(X_test)
cm = confusion_matrix(y_test, pred_classes, labels=['negative', 'neutral', 'positive'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
