import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data from the uploaded CSV file
file_path = 'amazon/Musical_Instruments_reviews.csv'
df = pd.read_csv(file_path)
data = df[['reviewText', 'overall']]
df = df.dropna()

def label_encoding(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    elif score in [4, 5]:
        return 'positive'

df['label'] = df['overall'].apply(label_encoding)

# Balance the dataset
neutral_samples = df[df['label'] == 'neutral']
negative_samples = df[df['label'] == 'negative']
df = pd.concat([df, neutral_samples, neutral_samples, neutral_samples, negative_samples, negative_samples], axis=0, ignore_index=True)
label_distribution_after = df['label'].value_counts()
print("\nLabel Distribution After Duplication:\n", label_distribution_after)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_df['reviewText'])
max_length = max([len(x.split()) for x in train_df['reviewText']])
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(train_df['reviewText']), maxlen=max_length)
X_val_pad = pad_sequences(tokenizer.texts_to_sequences(val_df['reviewText']), maxlen=max_length)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(test_df['reviewText']), maxlen=max_length)

# Prepare labels
label_encoder = LabelEncoder()
y_train = to_categorical(label_encoder.fit_transform(train_df['label']))
y_val = to_categorical(label_encoder.transform(val_df['label']))
y_test = to_categorical(label_encoder.transform(test_df['label']))

# [Rest of your code remains the same for model building, training, and evaluation]
# Build model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=300, input_length=max_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))  # 3 units for 3 classes

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train_pad, y_train, batch_size=64, epochs=2, validation_data=(X_val_pad, y_val))

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate confusion matrix
test_predictions = model.predict(X_test_pad)
test_predictions = np.argmax(test_predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_test_labels, test_predictions)

# Print classification report
print("\nTest Set Classification Report:")
print(classification_report(y_test_labels, test_predictions, target_names=label_encoder.classes_))
# Generate confusion matrix
test_predictions = model.predict(X_test_pad)
test_predictions = np.argmax(test_predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_test_labels, test_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
