import pandas as pd
import re
import string
import nltk
import swifter  # Fast parallel processing
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from multiprocessing import cpu_count

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
file_path = "Data_latih.csv"
print("Loading dataset...")
df = pd.read_csv(file_path)

# Drop unnecessary columns
print("Dropping unnecessary columns...")
df = df.drop(columns=["ID", "tanggal", "nama file gambar"])

# Ensure 'label' is an integer
df['label'] = df['label'].astype(int)

# Initialize Sastrawi Stemmer
print("Initializing stemmer...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load stopwords
print("Loading stopwords...")
indonesian_stopwords = set(stopwords.words('indonesian'))

def clean_text(text):
    """Optimized text cleaning function."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)
    words = [word for word in words if word not in indonesian_stopwords]  # Remove stopwords
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    return " ".join(words)

# Apply preprocessing in parallel
tqdm.pandas()  # Show progress bar
print("Preprocessing 'judul' column...")
df["judul_clean"] = df["judul"].astype(str).swifter.apply(clean_text)

print("Preprocessing 'narasi' column...")
df["narasi_clean"] = df["narasi"].astype(str).swifter.apply(clean_text)

# Save the cleaned dataset
output_file = "Data_cleaned.csv"
df.to_csv(output_file, index=False)
print(f"Preprocessing done! Cleaned data saved as '{output_file}'.")
