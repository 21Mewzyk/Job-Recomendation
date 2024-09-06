import streamlit as st
import pandas as pd
import numpy as np
import base64
import os, sys
import pymongo
from JobRecommendation.exception import jobException
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64

# MongoDB configuration
dataBase = "Job_Hunter_DB"
collection = "Resume_Data"

# Set page configuration with a title and an icon
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RECRUITER")

# Add the logo and sidebar to the Streamlit app
add_logo()
sidebar()

# Encryption configuration
ENCRYPTION_PASSWORD = b"your-strong-password"  # Password for encryption/decryption
SALT = b'\x00\x01\x02\x03\x04\x05\x06\x07'  # The same salt used during encryption

# Derive an encryption key from the password using PBKDF2HMAC
def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password)

# AES decryption
def decrypt_data(encrypted_data, key):
    encrypted_data = base64.b64decode(encrypted_data)
    iv = encrypted_data[:16]  # Extract the IV from the start
    encrypted_data = encrypted_data[16:]
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()
    
    return decrypted_data

# Function to count the number of documents (CVs) in a specified MongoDB collection
def count_documents_in_collection(db_name, collection_name):
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB connection string if different
        db = client[db_name]
        collection = db[collection_name]
        
        # Return the count of documents in the collection
        document_count = collection.count_documents({})
        return document_count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Main function to run the app
def app():
    st.title('Candidate Recommendation')  # Set the page title
    
    # Create two columns for layout
    c1, c2 = st.columns((3, 2))
    
    # Slider to select the number of CV recommendations
    no_of_cv = c2.slider('Number of CV Recommendations:', min_value=0, max_value=6, step=1)
    
    # Text area to paste the job description
    jd = c1.text_area("PASTE YOUR JOB DESCRIPTION HERE")

    # Proceed if job description is provided
    if len(jd) >= 1:
        # Process the job description using NLP
        NLP_Processed_JD = text_preprocessing.nlp(jd)
        jd_text = ' '.join(NLP_Processed_JD)
        
        # Fetch the CV data from MongoDB and load into a DataFrame
        df = MongoDB_function.get_collection_as_dataframe(dataBase, collection)

        # Preprocess the CVs for text analysis
        df["clean_all"] = df["All"].apply(lambda x: ' '.join(text_preprocessing.nlp(x)))

        # Combine the job description and all CVs for TF-IDF vectorization
        documents = [jd_text] + df["clean_all"].tolist()
        
        # Perform TF-IDF vectorization on the job description and CVs
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        # Calculate cosine similarity between job description and CVs
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Add the cosine similarity scores to the DataFrame
        df["cosine_similarity"] = cosine_similarities

        # Set a threshold for cosine similarity (filtering out irrelevant CVs)
        similarity_threshold = 0.1  # Adjust threshold based on your preference

        # Filter CVs that meet the similarity threshold
        df_filtered = df[df["cosine_similarity"] >= similarity_threshold]

        # Sort CVs by their similarity scores in descending order
        df_sorted = df_filtered.sort_values(by="cosine_similarity", ascending=False).head(no_of_cv)

        # Display the top CVs (recommendations) if enough CVs meet the criteria
        if len(df_sorted) < no_of_cv:
            st.error(f"Not enough CVs to recommend. Only {len(df_sorted)} CVs available.")
            return
        
        # Display CV recommendations in an expander for better visibility
        db_expander = st.expander(label='CV recommendations:')
        with db_expander:
            no_of_cols = 3  # Define number of columns to display CVs
            cols = st.columns(no_of_cols)  # Create columns
            key = derive_key(ENCRYPTION_PASSWORD, SALT)  # Derive the key for decryption
            for i in range(len(df_sorted)):
                # Display details for each recommended CV in the columns
                cols[i % no_of_cols].text(f"CV ID: {df_sorted['Unnamed: 0'].iloc[i]}")
                cols[i % no_of_cols].text(f"Name: {df_sorted['name'].iloc[i]}")
                cols[i % no_of_cols].text(f"Phone no.: {df_sorted['mobile_number'].iloc[i]}")
                cols[i % no_of_cols].text(f"Skills: {df_sorted['skills'].iloc[i]}")
                cols[i % no_of_cols].text(f"Degree: {df_sorted['degree'].iloc[i]}")
                cols[i % no_of_cols].text(f"College/University: {df_sorted['college_name'].iloc[i]}")
                cols[i % no_of_cols].text(f"No. of Pages Resume: {df_sorted['no_of_pages'].iloc[i]}")
                cols[i % no_of_cols].text(f"Email: {df_sorted['email'].iloc[i]}")
                
                # Decrypt the base64-encoded and encrypted PDF resume
                encrypted_pdf = df_sorted['pdf_to_base64'].iloc[i]
                decrypted_pdf = decrypt_data(encrypted_pdf, key).decode('utf-8')
                
                # Generate a download link for the decrypted resume (encoded in base64)
                cols[i % no_of_cols].markdown(f'<a href="data:application/octet-stream;base64,{decrypted_pdf}" download="resume.pdf"><button style="background-color:GreenYellow;">Download Resume</button></a>', unsafe_allow_html=True)
                
                # Generate and display an embedded preview of the resume (PDF)
                embed_code = utils.show_pdf(decrypted_pdf)
                cvID = df_sorted['Unnamed: 0'].iloc[i]
                show_pdf = cols[i % no_of_cols].button(f"{cvID}.pdf")  # Button to show PDF
                if show_pdf:
                    st.markdown(embed_code, unsafe_allow_html=True)  # Display embedded PDF
                
                # Separate each CV with a line
                cols[i % no_of_cols].text('___________________________________________________')
    else:
        # Prompt the user to provide a job description
        st.write("<p style='font-size:15px;'>Please Provide The Job Description </p>", unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    app()
