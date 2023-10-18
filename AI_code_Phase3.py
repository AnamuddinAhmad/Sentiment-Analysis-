import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Load the dataset (replace 'your_dataset.csv' with the actual dataset file)
dataset = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset for a quick overview
print("Sample data:")
print(dataset.head())

# Data Preprocessing: Text Cleaning and Preprocessing
def clean_text(text):
    # Remove special characters, links, and extra spaces
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text cleaning to the 'text' column in the dataset
dataset['text'] = dataset['text'].apply(clean_text)

# Tokenization
def tokenize_text(text):
    return text.split()

# Apply tokenization to the 'text' column
dataset['tokenized_text'] = dataset['text'].apply(tokenize_text)

# Stop Word Removal
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

# Apply stop word removal
dataset['tokenized_text'] = dataset['tokenized_text'].apply(remove_stopwords)

# Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

# Apply lemmatization
dataset['tokenized_text'] = dataset['tokenized_text'].apply(lemmatize_tokens)

# Splitting the dataset into training and testing sets (you can adjust the test_size)
X = dataset['tokenized_text']
y = dataset['sentiment']  # Assuming 'sentiment' is the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the preprocessed data
print("\nPreprocessed data:")
print(X_train.head())

# You can save this preprocessed data to use it for further analysis or modeling.

# Next steps (Development Part 2) would involve feature extraction, model building, and more.
