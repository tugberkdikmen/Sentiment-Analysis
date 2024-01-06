import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.utils import resample

# Download NLTK stopwords (if not already downloaded)
import nltk
nltk.download('stopwords')

# Reading and processing the JSON file
file_path_new = r'C:\Users\efser\Desktop\CS_464_ML\Sentiment_Analysis_on_Customer_Feedback_Project\Dataset1\Musical_Instruments_5.json'

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

# Define label_mapping dictionary
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

# Print the updated value counts
print("Number of data points in each sentiment category before upsampling:")
print(data['Sentiment'].value_counts())

# Upsample minority classes for the training set only
data_majority = data[data['Sentiment'] == 'positive']
data_minority_negative = data[data['Sentiment'] == 'negative']
data_minority_neutral = data[data['Sentiment'] == 'neutral']

# Upsample minority classes for the training set only
data_minority_negative_upsampled = resample(data_minority_negative, replace=True, n_samples=len(data_majority), random_state=42)
data_minority_neutral_upsampled = resample(data_minority_neutral, replace=True, n_samples=len(data_majority), random_state=42)

# Combine the upsampled minority classes with the majority class for the training set only
data_upsampled = pd.concat([data_majority, data_minority_negative_upsampled, data_minority_neutral_upsampled])

# Print the number of data points for each sentiment category after upsampling
print("\nNumber of data points in each sentiment category after upsampling:")
print(data_upsampled['Sentiment'].value_counts())

# Convert string labels to numerical values for the training set only
y_train_upsampled = np.array([label_mapping[label] for label in data_upsampled['Sentiment']])

# Split the dataset after upsampling
X_train, X_temp, y_train, y_temp = train_test_split(data_upsampled['reviewText'], y_train_upsampled, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Create the AdamW optimizer using transformers library
optimizer, _ = create_optimizer(2e-5, num_train_steps=100000, num_warmup_steps=10000)

# Compile the model with the AdamW optimizer
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Tokenize the training data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256, return_tensors='tf')

# Convert labels to one-hot encoded format
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=3)

# Adjust batch size and use gradient accumulation
effective_batch_size = 32
accumulation_steps = 2  # Increase this based on GPU/TPU memory
batch_size = effective_batch_size // accumulation_steps

# Check the shapes of the input data and labels
print("\nShapes before training:")
print(train_encodings['input_ids'].shape, train_encodings['token_type_ids'].shape, train_encodings['attention_mask'].shape, y_train_one_hot.shape)

# Train the model with the re-tokenized data after upsampling and gradient accumulation
history_upsampled = model.fit(
    x={
        'input_ids': train_encodings['input_ids'],
        'token_type_ids': train_encodings['token_type_ids'],
        'attention_mask': train_encodings['attention_mask'],
    },
    y=y_train_one_hot,
    validation_data=(
        {
            'input_ids': tokenizer(list(X_val), truncation=True, padding=True, max_length=256, return_tensors='tf')['input_ids'],
            'token_type_ids': tokenizer(list(X_val), truncation=True, padding=True, max_length=256, return_tensors='tf')['token_type_ids'],
            'attention_mask': tokenizer(list(X_val), truncation=True, padding=True, max_length=256, return_tensors='tf')['attention_mask'],
        },
        tf.keras.utils.to_categorical(y_val, num_classes=3)
    ),
    epochs=3,
    batch_size=batch_size,
    steps_per_epoch=len(X_train) // batch_size // accumulation_steps,
)
