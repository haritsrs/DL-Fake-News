import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load and prepare data
df = pd.read_csv("Data_cleaned.csv")

def clean_text(text):
    """Improved text preprocessing"""
    indonesian_stopwords = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "ini", "itu", 
        "atau", "dalam", "oleh", "seperti", "juga", "karena", "ada", "saja", "sudah"
    }
    text = str(text).lower()  # Ensure text format and lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()  # Tokenization
    words = [word for word in words if word not in indonesian_stopwords]  # Stopword removal
    return ' '.join(words)

# Apply preprocessing
df["text"] = (df["judul_clean"].fillna('') + " [SEP] " + df["narasi_clean"].fillna('')).apply(clean_text)
labels = df["label"].values

# Tokenization
max_features = 10000
max_length = 200
embedding_dim = 128
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>", filters='')
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# Hybrid Conv1D + LSTM Model
model = Sequential([
    Embedding(max_features, embedding_dim, mask_zero=True),
    Dropout(0.5),
    Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling1D(3),
    BatchNormalization(),
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
    LSTM(32, kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Custom learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'AUC'])

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_AUC', patience=7, mode='max', restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_AUC', save_best_only=True, mode='max')
]

# Training
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluation
results = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {results[1]*100:.2f}% | Test AUC: {results[2]*100:.2f}%")
print("Class distribution:", np.unique(labels, return_counts=True))
print("Sample texts:", df["text"].head(3).tolist())
