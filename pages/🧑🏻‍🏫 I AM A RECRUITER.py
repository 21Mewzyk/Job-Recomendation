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
from sklearn.preprocessing import MinMaxScaler

dataBase = "Job-Recomendation"
collection = "Resume_Data"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RECRUITER")

add_logo()
sidebar()

def app():
    st.title('Candidate Recommendation')
    c1, c2 = st.columns((3, 2))
    no_of_cv = c2.slider('Number of CV Recommendations:', min_value=0, max_value=6, step=1)
    jd = c1.text_area("PASTE YOUR JOB DESCRIPTION HERE")

    if len(jd) >= 1:
        NLP_Processed_JD = text_preprocessing.nlp(jd)

        jd_df = pd.DataFrame()
        jd_df['jd'] = [' '.join(NLP_Processed_JD)]

        @st.cache_data
        def get_recommendation(top, df_all, scores):
            try:
                # Debug: Print the columns of df_all
                st.write("Columns in df_all before creating recommendation:", df_all.columns)
                recommendation = pd.DataFrame(columns=['name', 'degree', "email", 'index', 'mobile_number', 'skills', 'no_of_pages', 'score', 'pdf_to_base64'])
                count = 0
                for i in top:
                    recommendation.at[count, 'name'] = df_all['name'][i]
                    recommendation.at[count, 'degree'] = df_all['degree'][i]
                    recommendation.at[count, 'email'] = df_all['email'][i]
                    recommendation.at[count, 'index'] = df_all.index[i]
                    recommendation.at[count, 'mobile_number'] = df_all['mobile_number'][i]
                    recommendation.at[count, 'skills'] = df_all['skills'][i]
                    recommendation.at[count, 'no_of_pages'] = df_all['no_of_pages'][i]
                    recommendation.at[count, 'score'] = scores[count]
                    recommendation.at[count, 'pdf_to_base64'] = df_all['pdf_to_base64'][i]
                    count += 1
                return recommendation
            except Exception as e:
                raise jobException(e, sys)

        df = MongoDB_function.get_collection_as_dataframe(dataBase, collection)

        # Debug: Show available columns
        st.write("Available columns in the data before preprocessing:", df.columns)

        # Ensure 'pdf_to_base64' column is present
        if 'pdf_to_base64' not in df.columns:
            st.error("Column 'pdf_to_base64' is not found in the DataFrame.")
            return

        cv_data = []
        for i in range(len(df["All"])):
            NLP_Processed_cv = text_preprocessing.nlp(df["All"].values[i])
            cv_data.append(NLP_Processed_cv)

        cv_ = []
        for i in cv_data:
            cv_.append([' '.join(i)])

        df["clean_all"] = pd.DataFrame(cv_)

        # Debug: Show available columns after preprocessing
        st.write("Available columns in the data after preprocessing:", df.columns)

        # Define the number of neighbors for KNN
        n_neighbors = 3  # Adjust this value as needed

        # Ensure n_neighbors does not exceed the number of samples
        if len(df['clean_all']) < n_neighbors:
            n_neighbors = len(df['clean_all'])

        # TfidfVectorizer function
        tf = distance_calculation.TFIDF(df['clean_all'], jd_df['jd'])
        top = sorted(range(len(tf)), key=lambda i: tf[i], reverse=True)[:100]
        list_scores = [tf[i][0][0] for i in top]
        TF = get_recommendation(top, df, list_scores)

        # Count Vectorizer function
        countv = distance_calculation.count_vectorize(df['clean_all'], jd_df['jd'])
        top = sorted(range(len(countv)), key=lambda i: countv[i], reverse=True)[:100]
        list_scores = [countv[i][0][0] for i in top]
        cv = get_recommendation(top, df, list_scores)

        # KNN function
        top, index_score = distance_calculation.KNN(df['clean_all'], jd_df['jd'], n_neighbors)
        knn = get_recommendation(top, df, index_score)

        # Ensure 'index' column exists
        if 'index' not in knn.columns:
            knn['index'] = knn.index
        if 'index' not in TF.columns:
            TF['index'] = TF.index
        if 'index' not in cv.columns:
            cv['index'] = cv.index

        merge1 = knn[['index', 'name', 'score']].merge(TF[['index', 'score']], on="index")
        final = merge1.merge(cv[['index', 'score']], on='index')
        final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})

        # Debug: Show available columns before merging pdf_to_base64
        st.write("Available columns before merging pdf_to_base64:", final.columns)

        # Include 'pdf_to_base64' in the final DataFrame
        final = final.merge(df[['index', 'pdf_to_base64']], left_on='index', right_index=True)

        # Debug: Show available columns in the final DataFrame
        st.write("Available columns in the final DataFrame:", final.columns)

        # Scale it
        slr = MinMaxScaler()
        final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

        # Multiply by weights
        final['KNN'] = (1 - final['KNN']) / 3
        final['TF-IDF'] = final['TF-IDF'] / 3
        final['CV'] = final['CV'] / 3
        final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

        final = final.sort_values(by="Final", ascending=False)
        final1 = final.sort_values(by="Final", ascending=False).copy()
        final_df = df.merge(final1, left_index=True, right_on='index')
        final_df = final_df.sort_values(by="Final", ascending=False)
        final_df = final_df.reset_index(drop=True)  # reset the index
        final_df = final_df.head(no_of_cv)

        db_expander = st.expander(label='CV recommendations:')
        with db_expander:
            no_of_cols = 3
            cols = st.columns(no_of_cols)
            for i in range(0, no_of_cv):
                cols[i % no_of_cols].text(f"CV ID: {final_df['index'][i]}")
                cols[i % no_of_cols].text(f"Name: {final_df['name_x'][i]}")
                cols[i % no_of_cols].text(f"Phone no.: {final_df['mobile_number'][i]}")
                cols[i % no_of_cols].text(f"Skills: {final_df['skills'][i]}")
                cols[i % no_of_cols].text(f"Degree: {final_df['degree'][i]}")
                cols[i % no_of_cols].text(f"No. of Pages Resume: {final_df['no_of_pages'][i]}")
                cols[i % no_of_cols].text(f"Email: {final_df['email'][i]}")
                encoded_pdf = final_df['pdf_to_base64'][i]
                cols[i % no_of_cols].markdown(f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume.pdf"><button style="background-color:GreenYellow;">Download Resume</button></a>', unsafe_allow_html=True)
                embed_code = utils.show_pdf(encoded_pdf)
                cvID = final1['index'][i]
                show_pdf = cols[i % no_of_cols].button(f"{cvID}.pdf")
                if show_pdf:
                    st.markdown(embed_code, unsafe_allow_html=True)
                cols[i % no_of_cols].text('___________________________________________________')
    else:
        st.write("<p style='font-size:15px;'>Please Provide The Job Description </p>", unsafe_allow_html=True)

if __name__ == '__main__':
    app()
