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
from JobRecommendation import text_preprocessing, distance_calculation
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
        jd_df = pd.DataFrame()
        jd_df['jd'] = [' '.join(NLP_Processed_JD)]
        
        # Count documents in the collection before proceeding
        document_count = count_documents_in_collection(dataBase, collection)
        if document_count is not None:
            st.write(f"Total number of CVs in the database: {document_count}")
        else:
            st.write("Failed to retrieve the number of documents.")

        @st.cache_data
        def get_recommendation(top, df, scores):
            try:
                recommendation = pd.DataFrame(columns=['name', 'degree', "email", 'Unnamed: 0', 'mobile_number', 'skills', 'no_of_pages', 'score'])
                count = 0
                for i in top:
                    recommendation.at[count, 'name'] = df['name'][i]
                    recommendation.at[count, 'degree'] = df['degree'][i]
                    recommendation.at[count, 'email'] = df['email'][i]
                    recommendation.at[count, 'Unnamed: 0'] = df.index[i]
                    recommendation.at[count, 'mobile_number'] = df['mobile_number'][i]
                    recommendation.at[count, 'skills'] = df['skills'][i]
                    recommendation.at[count, 'no_of_pages'] = df['no_of_pages'][i]
                    recommendation.at[count, 'score'] = scores[count]
                    count += 1
                return recommendation
            except Exception as e:
                raise jobException(e, sys)

        df = MongoDB_function.get_collection_as_dataframe(dataBase, collection)

        cv_data = []
        for i in range(len(df["All"])):
            NLP_Processed_cv = text_preprocessing.nlp(df["All"].values[i])
            cv_data.append(NLP_Processed_cv)

        cv_ = []
        for i in cv_data:
            cv_.append([' '.join(i)])

        df["clean_all"] = pd.DataFrame(cv_)

        # TF-IDF Calculation
        tf = distance_calculation.TFIDF(df['clean_all'], jd_df['jd'])
        top_tf = sorted(range(len(tf)), key=lambda i: tf[i], reverse=True)[:100]
        list_scores_tf = [tf[i][0][0] for i in top_tf]
        TF = get_recommendation(top_tf, df, list_scores_tf)

        # Count Vectorizer Calculation
        countv = distance_calculation.count_vectorize(df['clean_all'], jd_df['jd'])
        top_cv = sorted(range(len(countv)), key=lambda i: countv[i], reverse=True)[:100]
        list_scores_cv = [countv[i][0][0] for i in top_cv]
        cv = get_recommendation(top_cv, df, list_scores_cv)

        # Dynamic KNN Calculation
        if document_count is not None and document_count > 0:
            neighbors = min(document_count, 10)  # Use up to 10 neighbors, adjust as needed
            top_knn, index_score_knn = distance_calculation.KNN(df['clean_all'], jd_df['jd'], number_of_neighbors=neighbors)
            knn = get_recommendation(top_knn, df, index_score_knn)
        else:
            st.error("No CVs available in the database for KNN calculation.")
            return

        # Merge and Calculate Final Score
        merge1 = knn[['Unnamed: 0', 'name', 'score']].merge(TF[['Unnamed: 0', 'score']], on="Unnamed: 0")
        final = merge1.merge(cv[['Unnamed: 0', 'score']], on='Unnamed: 0')
        final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})

        # Normalize Scores
        from sklearn.preprocessing import MinMaxScaler
        slr = MinMaxScaler()
        final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

        # Adjust Weights
        final['KNN'] = (1 - final['KNN']) * 0.4
        final['TF-IDF'] = final['TF-IDF'] * 0.3
        final['CV'] = final['CV'] * 0.3
        final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

        # Sort by Final Score
        final = final.sort_values(by="Final", ascending=False)
        final1 = final.sort_values(by="Final", ascending=False).copy()
        final_df = df.merge(final1, on='Unnamed: 0')
        final_df = final_df.sort_values(by="Final", ascending=False)
        final_df = final_df.reset_index(drop=True)
        final_df = final_df.head(no_of_cv)

        if len(final_df) < no_of_cv:
            st.error(f"Not enough CVs to recommend. Only {len(final_df)} CVs available.")
            return
        
        db_expander = st.expander(label='CV recommendations:')
        with db_expander:
            no_of_cols = 3
            cols = st.columns(no_of_cols)
            for i in range(len(final_df)):
                cols[i % no_of_cols].text(f"CV ID: {final_df['Unnamed: 0'][i]}")
                cols[i % no_of_cols].text(f"Name: {final_df['name_x'][i]}")
                cols[i % no_of_cols].text(f"Phone no.: {final_df['mobile_number'][i]}")
                cols[i % no_of_cols].text(f"Skills: {final_df['skills'][i]}")
                cols[i % no_of_cols].text(f"Degree: {final_df['degree'][i]}")
                cols[i % no_of_cols].text(f"No. of Pages Resume: {final_df['no_of_pages'][i]}")
                cols[i % no_of_cols].text(f"Email: {final_df['email'][i]}")
                encoded_pdf = final_df['pdf_to_base64'][i]
                cols[i % no_of_cols].markdown(f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume.pdf"><button style="background-color:GreenYellow;">Download Resume</button></a>', unsafe_allow_html=True)
                embed_code = utils.show_pdf(encoded_pdf)
                cvID = final1['Unnamed: 0'][i]
                show_pdf = cols[i % no_of_cols].button(f"{cvID}.pdf")
                if show_pdf:
                    st.markdown(embed_code, unsafe_allow_html=True)
                cols[i % no_of_cols].text('___________________________________________________')
    else:
        st.write("<p style='font-size:15px;'>Please Provide The Job Description </p>", unsafe_allow_html=True)

if __name__ == '__main__':
    app()
