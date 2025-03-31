import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model("text_classifier.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load dataset
df = pd.read_csv("Data_cleaned.csv")
texts = df["judul_clean"].astype(str).tolist()
labels = df["label"].values

# Tokenize & pad sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

# Split test set (same split as training)
split = int(0.8 * len(padded_sequences))
X_test, y_test = padded_sequences[split:], labels[split:]

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")  # Example: Test Accuracy: 0.8765
