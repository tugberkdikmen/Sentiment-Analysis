import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


file_path = r'archive/olist_order_reviews_dataset.csv'  # Use the correct file path


# Load the CSV file with column names
df = pd.read_csv(file_path)


# Extract relevant columns
df = df[['review_score', 'review_comment_message']]

# Handle missing values if any
df = df.dropna()

# Define the function for label encoding
def label_encoding(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    elif score in [4, 5]:
        return 'positive'

# Apply label encoding to 'review_score' column
df['label'] = df['review_score'].apply(label_encoding)

label_distribution = df['label'].value_counts()
print("Label Distribution:\n", label_distribution)

# Split the data into train, validation, and test sets
train_df, test_val_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

# Feature extraction
X_train = train_df['review_comment_message'].values
X_val = val_df['review_comment_message'].values
X_test = test_df['review_comment_message'].values

total_samples = len(df)
train_percentage = len(train_df) / total_samples * 100
val_percentage = len(val_df) / total_samples * 100
test_percentage = len(test_df) / total_samples * 100

print(f"Train Set Percentage: {train_percentage}")
print(f"Validation Set Percentage: {val_percentage}")
print(f"Test Set Percentage: {test_percentage}")

# Train the SVM model
model = make_pipeline(TfidfVectorizer(), SVC(C=1))
model.fit(X_train, train_df['label'])

# Evaluate the model on the validation set
val_predictions = model.predict(X_val)
print("Validation Set Classification Report:\n", classification_report(val_df['label'], val_predictions))

# Test the model on the test set
test_predictions = model.predict(X_test)
print("Test Set Classification Report:\n", classification_report(test_df['label'], test_predictions))

neutral_samples = df[df['label'] == 'neutral']
df = pd.concat([df, neutral_samples, neutral_samples], axis=0, ignore_index=True)

# Display the distribution of labels after duplication
label_distribution_after = df['label'].value_counts()
print("\nLabel Distribution After Duplication:\n", label_distribution_after)

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Feature extraction
X_train = train_df['review_comment_message'].values
X_val = val_df['review_comment_message'].values
X_test = test_df['review_comment_message'].values

# Train the SVM model
model = make_pipeline(TfidfVectorizer(), SVC())
model.fit(X_train, train_df['label'])

# Evaluate the model on the validation set
val_predictions = model.predict(X_val)
print("Validation Set Classification Report:\n", classification_report(val_df['label'], val_predictions))

# Test the model on the test set
test_predictions = model.predict(X_test)
print("Test Set Classification Report:\n", classification_report(test_df['label'], test_predictions))
