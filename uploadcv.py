from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import streamlit as st
import sys

# Import additional necessary modules for text preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/?authSource=admin")
db = client["Job-Recomendation"]
collection = db["Resume_from_CANDIDATE"]

# Function to concatenate various fields into the "All" column
def calculate_all_column(doc):
    fields = [
        doc.get('id', ''),
        doc.get('name', ''),
        doc.get('job_title', ''),
        doc.get('responsibilities', ''),
        doc.get('basic_qualifications', ''),
        doc.get('knowledge_skills', ''),
        doc.get('competencies', ''),
        doc.get('location', ''),
        doc.get('rating', ''),
        doc.get('salary', ''),
        doc.get('posted', ''),
        doc.get('apply_link', ''),
        doc.get('company_link', '')
    ]
    all_column = ', '.join(fields)
    return all_column

# Fetch all documents from the collection
documents = list(collection.find())

# Update each document with the "All" column
for doc in documents:
    all_column_value = calculate_all_column(doc)
    collection.update_one({"_id": doc["_id"]}, {"$set": {"All": all_column_value}})

print("All documents updated with the 'All' column.")

# Load the updated documents
documents = list(collection.find())
df = pd.DataFrame(documents)

# Preprocess text data: remove stop words and empty strings
def preprocess_text(text):
    if not text or text.isspace():
        return ''
    # Tokenize and remove stop words
    vectorizer = CountVectorizer(stop_words='english')
    analyzer = vectorizer.build_analyzer()
    tokens = analyzer(text)
    return ' '.join(tokens)

df['clean_all'] = df['All'].apply(preprocess_text)

# Remove empty strings after preprocessing
df = df[df['clean_all'].str.strip().astype(bool)]

# Function for TFIDF calculation
def TFIDF(corpus, query):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform(query)
        return tfidf_matrix, query_vector
    except Exception as e:
        raise jobException(e, sys)

# Assuming `jd_df` is already defined and contains the job description
jd_df = pd.DataFrame({'jd': ['Your job description here']})  # Replace with actual job description

# Perform TFIDF calculation
tf, query = TFIDF(df['clean_all'], jd_df['jd'])
