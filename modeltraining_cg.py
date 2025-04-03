import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Load the cleaned dataset
data_file = "Data_cleaned.csv"
print("Loading cleaned data...")
df = pd.read_csv(data_file)

# Combine 'judul_clean' and 'narasi_clean' for a richer text input (optional)
# You can choose to use just one if preferred.
df["text"] = df["judul_clean"].fillna('') + " " + df["narasi_clean"].fillna('')

# Labels: assuming 0 = real, 1 = fake
labels = df["label"].values

# Set up parameters
max_features = 5000    # Use top 5000 words
max_length = 150       # Maximum length of text sequences (adjust as needed)
embedding_dim = 64
batch_size = 32
epochs = 20            # Increase if you want longer training

# Tokenize the text data
print("Tokenizing text...")
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=max_length)

# Split into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Build the LSTM model
print("Building model...")
model = Sequential([
    Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.6),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Set up callbacks for early stopping and saving the best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
]

# Start training
print("Starting training... This might take a while, so you can let it run overnight.")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

# Save the final model and tokenizer for later use
model.save("final_model.h5")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training complete. Model and tokenizer saved.")