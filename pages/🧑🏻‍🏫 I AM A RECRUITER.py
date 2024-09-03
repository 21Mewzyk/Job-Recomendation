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

dataBase = "Job_Hunter_DB"
collection = "Resume_Data"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RECRUITER")

add_logo()
sidebar()

def count_documents_in_collection(db_name, collection_name):
    try:
        # Establish connection to MongoDB
        client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB connection string if different
        db = client[db_name]
        collection = db[collection_name]
        
        # Count documents in the collection
        document_count = collection.count_documents({})
        return document_count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def app():
    st.title('Candidate Recommendation')
    c1, c2 = st.columns((3, 2))
    no_of_cv = c2.slider('Number of CV Recommendations:', min_value=0, max_value=6, step=1)
    jd = c1.text_area("PASTE YOUR JOB DESCRIPTION HERE")

    if len(jd) >= 1:
        NLP_Processed_JD = text_preprocessing.nlp(jd)
        jd_text = ' '.join(NLP_Processed_JD)
        
        # Count documents in the collection before proceeding
        document_count = count_documents_in_collection(dataBase, collection)
        if document_count is not None:
            st.write(f"Total number of CVs in the database: {document_count}")
        else:
            st.write("Failed to retrieve the number of documents.")

        df = MongoDB_function.get_collection_as_dataframe(dataBase, collection)

        # Process CVs
        df["clean_all"] = df["All"].apply(lambda x: ' '.join(text_preprocessing.nlp(x)))

        # Combine Job Description and CVs for TF-IDF Vectorization
        documents = [jd_text] + df["clean_all"].tolist()
        
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        # Calculate Cosine Similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Add Cosine Similarity Scores to DataFrame
        df["cosine_similarity"] = cosine_similarities

        # Define a threshold for cosine similarity
        similarity_threshold = 0.1  # Adjust this threshold based on your needs

        # Filter CVs that meet the similarity threshold
        df_filtered = df[df["cosine_similarity"] >= similarity_threshold]

        # Rank CVs by Cosine Similarity
        df_sorted = df_filtered.sort_values(by="cosine_similarity", ascending=False).head(no_of_cv)

        # Display the top-ranked CVs
        if len(df_sorted) < no_of_cv:
            st.error(f"Not enough CVs to recommend. Only {len(df_sorted)} CVs available.")
            return
        
        db_expander = st.expander(label='CV recommendations:')
        with db_expander:
            no_of_cols = 3
            cols = st.columns(no_of_cols)
            for i in range(len(df_sorted)):
                cols[i % no_of_cols].text(f"CV ID: {df_sorted['Unnamed: 0'].iloc[i]}")
                cols[i % no_of_cols].text(f"Name: {df_sorted['name'].iloc[i]}")
                cols[i % no_of_cols].text(f"Phone no.: {df_sorted['mobile_number'].iloc[i]}")
                cols[i % no_of_cols].text(f"Skills: {df_sorted['skills'].iloc[i]}")
                cols[i % no_of_cols].text(f"Degree: {df_sorted['degree'].iloc[i]}")
                cols[i % no_of_cols].text(f"College/University: {df_sorted['college_name'].iloc[i]}")
                cols[i % no_of_cols].text(f"No. of Pages Resume: {df_sorted['no_of_pages'].iloc[i]}")
                cols[i % no_of_cols].text(f"Email: {df_sorted['email'].iloc[i]}")
                encoded_pdf = df_sorted['pdf_to_base64'].iloc[i]
                cols[i % no_of_cols].markdown(f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume.pdf"><button style="background-color:GreenYellow;">Download Resume</button></a>', unsafe_allow_html=True)
                embed_code = utils.show_pdf(encoded_pdf)
                cvID = df_sorted['Unnamed: 0'].iloc[i]
                show_pdf = cols[i % no_of_cols].button(f"{cvID}.pdf")
                if show_pdf:
                    st.markdown(embed_code, unsafe_allow_html=True)
                cols[i % no_of_cols].text('___________________________________________________')
    else:
        st.write("<p style='font-size:15px;'>Please Provide The Job Description </p>", unsafe_allow_html=True)

if __name__ == '__main__':
    app()