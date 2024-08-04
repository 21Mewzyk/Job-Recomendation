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
                recommendation = pd.DataFrame(columns=['name', 'degree', "email", 'index', 'mobile_number', 'skills', 'no_of_pages', 'score'])
                count = 0
                for i in top:
                    recommendation.at[count, 'name'] = df['name'][i]
                    recommendation.at[count, 'degree'] = df['degree'][i]
                    recommendation.at[count, 'email'] = df['email'][i]
                    recommendation.at[count, 'index'] = df.index[i]
                    recommendation.at[count, 'mobile_number'] = df['mobile_number'][i]
                    recommendation.at[count, 'skills'] = df['skills'][i]
                    recommendation.at[count, 'no_of_pages'] = df['no_of_pages'][i]
                    recommendation.at[count, 'score'] = scores[count]
                    count += 1
                return recommendation
            except Exception as e:
                raise jobException(e, sys)

        df = MongoDB_function.get_collection_as_dataframe(dataBase, collection)

        # Ensure required columns are present
        required_columns = ['clean_all', 'pdf_to_base64']  # Add all required columns here
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Example of adding 'clean_all' column with default values if it doesn't exist or is empty
        if 'clean_all' not in df.columns or df['clean_all'].isnull().all():
            df['clean_all'] = 'default_value'  # Replace with your logic for computing values

        cv_data = []
        for i in range(len(df["clean_all"])):
            NLP_Processed_cv = text_preprocessing.nlp(df["clean_all"].values[i])
            cv_data.append(NLP_Processed_cv)

        cv_ = []
        for i in cv_data:
            cv_.append([' '.join(i)])

        df["clean_all"] = pd.DataFrame(cv_)

        tf = distance_calculation.TFIDF(df['clean_all'], jd_df['jd'])
        top = sorted(range(len(tf)), key=lambda i: tf[i], reverse=True)[:100]
        list_scores = [tf[i][0][0] for i in top]
        TF = get_recommendation(top, df, list_scores)

        countv = distance_calculation.count_vectorize(df['clean_all'], jd_df['jd'])
        top = sorted(range(len(countv)), key=lambda i: countv[i], reverse=True)[:100]
        list_scores = [countv[i][0][0] for i in top]
        cv = get_recommendation(top, df, list_scores)

        top, index_score = distance_calculation.KNN(df['clean_all'], jd_df['jd'], number_of_neighbors=min(len(df['clean_all']), 19))
        knn = get_recommendation(top, df, index_score)

        merge1 = knn[['index', 'name', 'score']].merge(TF[['index', 'score']], on="index")
        final = merge1.merge(cv[['index', 'score']], on='index')
        final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})

        from sklearn.preprocessing import MinMaxScaler
        slr = MinMaxScaler()
        final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

        final['KNN'] = (1 - final['KNN']) / 3
        final['TF-IDF'] = final['TF-IDF'] / 3
        final['CV'] = final['CV'] / 3
        final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

        final = final.sort_values(by="Final", ascending=False)
        final1 = final.sort_values(by="Final", ascending=False).copy()
        final_df = df.merge(final1, left_index=True, right_on='index')
        final_df = final_df.sort_values(by="Final", ascending=False)
        final_df = final_df.reset_index(drop=True)
        final_df = final_df.head(no_of_cv)

        db_expander = st.expander(label='CV recommendations:')
        with db_expander:
            no_of_cols = 3
            cols = st.columns(no_of_cols)
            for i in range(0, min(no_of_cv, len(final_df))):  # Ensure we don't go out of range
                cols[i % no_of_cols].text(f"CV ID: {final_df['index'][i]}")
                cols[i % no_of_cols].text(f"Name: {final_df['name_x'][i]}")
                cols[i % no_of_cols].text(f"Phone no.: {final_df['mobile_number'][i]}")
                cols[i % no_of_cols].text(f"Skills: {final_df['skills'][i]}")
                cols[i % no_of_cols].text(f"Degree: {final_df['degree'][i]}")
                cols[i % no_of_cols].text(f"No. of Pages Resume: {final_df['no_of_pages'][i]}")
                cols[i % no_of_cols].text(f"Email: {final_df['email'][i]}")

                # Check if 'pdf_to_base64' column exists and handle it accordingly
                if 'pdf_to_base64' in final_df.columns:
                    encoded_pdf = final_df['pdf_to_base64'][i]
                    cols[i % no_of_cols].markdown(f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume.pdf"><button style="background-color:GreenYellow;">Download Resume</button></a>', unsafe_allow_html=True)
                    embed_code = utils.show_pdf(encoded_pdf)
                    cvID = final1['index'][i]
                    show_pdf = cols[i % no_of_cols].button(f"{cvID}.pdf")
                    if show_pdf:
                        st.markdown(embed_code, unsafe_allow_html=True)
                else:
                    cols[i % no_of_cols].text('PDF not available')

                cols[i % no_of_cols].text('___________________________________________________')

if __name__ == '__main__':
    app()
