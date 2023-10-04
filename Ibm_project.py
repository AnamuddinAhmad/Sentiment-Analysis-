import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data (if not already downloaded)
nltk.download("punkt")
nltk.download("stopwords")

# Load the dataset (you can replace this with your own dataset)
data = pd.read_csv("C:\Users\durge\OneDrive\Desktop\IBM_Project\tweets.csv")

# Data preprocessing
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic characters
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    # Join the words back into a sentence
    return " ".join(filtered_words)

data["clean_text"] = data["text"].apply(preprocess_text)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data["clean_text"])
y = data["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a sentiment classifier (Naive Bayes is just an example)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
