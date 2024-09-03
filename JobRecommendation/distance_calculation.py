from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from JobRecommendation.exception import jobException
import streamlit as st
import sys
import pandas as pd

@st.cache_data
def TFIDF(scraped_data, cv):
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        # TF-IDF Scraped data
        tfidf_jobid = tfidf_vectorizer.fit_transform(scraped_data)
        # TF-IDF CV
        user_tfidf = tfidf_vectorizer.transform(cv)
        # Using cosine_similarity on (Scraped data) & (CV)
        cos_similarity_tfidf = list(map(lambda x: cosine_similarity(user_tfidf, x), tfidf_jobid))
        
        # Flatten the similarity score list and sort by the highest score
        relevance_scores = [score[0][0] for score in cos_similarity_tfidf]
        ranked_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
        
        # Reorder the output based on the ranked indices
        sorted_output = [cos_similarity_tfidf[i] for i in ranked_indices]
        
        return sorted_output  # Return the sorted list of CVs
    except Exception as e:
        raise jobException(e, sys)

@st.cache_data
def count_vectorize(scraped_data, cv):
    try:
        # CountV the scraped data
        count_vectorizer = CountVectorizer()
        count_jobid = count_vectorizer.fit_transform(scraped_data)  # fitting and transforming the vector
        # CountV the cv
        user_count = count_vectorizer.transform(cv)
        cos_similarity_countv = list(map(lambda x: cosine_similarity(user_count, x), count_jobid))
        
        # Flatten the similarity score list and sort by the highest score
        relevance_scores = [score[0][0] for score in cos_similarity_countv]
        ranked_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
        
        # Reorder the output based on the ranked indices
        sorted_output = [cos_similarity_countv[i] for i in ranked_indices]
        
        return sorted_output
    except Exception as e:
        raise jobException(e, sys)

@st.cache_data
def KNN(scraped_data, cv, number_of_neighbors):
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        # n_neighbors = 100
        KNN = NearestNeighbors(n_neighbors=number_of_neighbors, p=2)
        KNN.fit(tfidf_vectorizer.fit_transform(scraped_data))
        NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv), return_distance=True)
        top = NNs[1][0][1:]
        index_score = NNs[0][0][1:]
        
        # Combine indices and scores for sorting
        combined = sorted(zip(top, index_score), key=lambda x: x[1], reverse=True)
        
        # Unzip the sorted tuples
        sorted_top, sorted_index_score = zip(*combined)
        
        return list(sorted_top), list(sorted_index_score)
    except Exception as e:
        raise jobException(e, sys)

def calculate_relevance(skills_list, job_skills):
    relevance_score = sum(skill in job_skills for skill in skills_list if pd.notna(skill))
    return relevance_score

@st.cache_data
def rank_cvs_by_relevance(df, job_skills):
    df['relevance_score'] = df.apply(lambda row: calculate_relevance(
        [row[f"skills[{i}]"] for i in range(5)], job_skills), axis=1)
    
    df_sorted = df.sort_values(by='relevance_score', ascending=False)
    return df_sorted
