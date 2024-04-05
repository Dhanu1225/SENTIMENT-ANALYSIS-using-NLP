# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:36:00 2024

@author: DELL
"""

import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Assuming "Downloads//reviews1.csv" is the correct path to your CSV file
csv_file_path = "C:\Pandas//reviews1.csv"
df = load_data(csv_file_path)

# Now you can use the df variable
# For example, you can print the first few rows of the DataFrame
print(df.head())
    
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens into a string
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

def main():
    st.title('Sentiment Analysis App')

    
       # Load CSV data
       
    df = load_data(csv_file_path)
        
        # Preprocess text
    df['MainReview'] = df['MainReview'].apply(preprocess_text)
         # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['MainReview'], df['sentiment_final'], test_size=0.2, random_state=42)
        
        # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

        # Train Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_vect, y_train)
        
        # Evaluate model
    train_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train_vect))
    test_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test_vect))
    st.write(f"Training Accuracy: {train_accuracy:.2f}")
    st.write(f"Test Accuracy: {test_accuracy:.2f}")

        # Save model and vectorizer
    joblib.dump(rf_classifier, "random_forest_sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        
        # Sentiment analysis on user input
    user_input = st.text_input("Enter text to analyze sentiment:")
    if st.button("Analyze Sentiment"):
        if user_input:
                preprocessed_input = preprocess_text(user_input)
                vectorized_input = vectorizer.transform([preprocessed_input])
                sentiment = rf_classifier.predict(vectorized_input)[0]
                st.write(f"Sentiment: {sentiment}")
        else:
                st.warning("Please enter some text.")

if __name__ == "__main__":
    main()

