import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import time, datetime
import base64, random
from pyresparser import ResumeParser
import os, sys
import pymongo
import json
from streamlit_lottie import st_lottie, st_lottie_spinner
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation
from JobRecommendation.exception import jobException

dataBase = "Job_Hunter_DB"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_Data"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

animation_file = "D:/Vscode_projects/Job-Recommendation/Animations/Loading 2.json"
animation_data = load_lottiefile(animation_file)

add_logo()
sidebar()

st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    st.title('Job Recommendation')
    c1, c2 = st.columns((3, 2))
    cv = c1.file_uploader('Upload your CV', type='pdf')
    no_of_jobs = st.slider('Max Number of Job Recommendations:', min_value=1, max_value=100, step=1)

    if cv is not None:
        if st.button('Proceed'):
            placeholder = st.empty()  
            with placeholder.container():
                # Center the animation
                st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                st_lottie(animation_data, height=700, width=700, key="download", reverse=True, speed=1, loop=True, quality='high')
                st.markdown("</div>", unsafe_allow_html=True)

            try:
                count_ = 0
                cv_text = utils.extract_data(cv)
                encoded_pdf = utils.pdf_to_base64(cv)
                resume_data = ResumeParser(cv).get_extracted_data()
                resume_data["pdf_to_base64"] = encoded_pdf

                timestamp = utils.generateUniqueFileName()
                save = {timestamp: resume_data}
                if count_ == 0:
                    count_ = 1
                # Fetch the last CV ID and increment it
                db = MongoDB_function.get_database(dataBase)
                last_cv = db[collection2].find_one(sort=[("Unnamed: 0", pymongo.DESCENDING)])
                if last_cv:
                    new_cv_id = last_cv["Unnamed: 0"] + 1
                else:
                    new_cv_id = 1
    
                # Prepare the document to insert
                resume_data["Unnamed: 0"] = new_cv_id
                resume_data["Unnamed: 0"] = int(resume_data["Unnamed: 0"])
    
                # Reorder the resume_data to have "Unnamed: 0" before "name"
                ordered_resume_data = {k: resume_data[k] for k in ["Unnamed: 0"] + [key for key in resume_data if key != "Unnamed: 0"]}

                #Insert the ordered document into the database
                db[collection2].insert_one(ordered_resume_data)


                try:
                    NLP_Processed_CV = text_preprocessing.nlp(cv_text)
                except NameError:
                    st.error('Please enter a valid input')

                df2 = pd.DataFrame()
                df2['title'] = ["I"]
                df2['job highlights'] = ["I"]
                df2['job description'] = ["I"]
                df2['company overview'] = ["I"]
                df2['industry'] = ["I"]
                df2['All'] = " ".join(NLP_Processed_CV)

                df = MongoDB_function.get_collection_as_dataframe(dataBase, collection1)

                @st.cache_data
                def get_recommendation(top, df_all, scores):
                    try:
                        recommendation = pd.DataFrame(columns=['positionName', 'company', "location", 'JobID', 'description', 'score'])
                        count = 0
                        for i in top:
                            recommendation.at[count, 'positionName'] = df['positionName'][i]
                            recommendation.at[count, 'company'] = df['company'][i]
                            recommendation.at[count, 'location'] = df['location'][i]
                            recommendation.at[count, 'JobID'] = df.index[i]
                            recommendation.at[count, 'description'] = df['description'][i]
                            recommendation.at[count, 'score'] = scores[count]
                            count += 1
                        return recommendation
                    except Exception as e:
                        raise jobException(e, sys)

                output2 = distance_calculation.TFIDF(df['All'], df2['All'])
                top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:1000]
                list_scores = [output2[i][0][0] for i in top]
                TF = get_recommendation(top, df, list_scores)

                output3 = distance_calculation.count_vectorize(df['All'], df2['All'])
                top = sorted(range(len(output3)), key=lambda i: output3[i], reverse=True)[:1000]
                list_scores = [output3[i][0][0] for i in top]
                cv = get_recommendation(top, df, list_scores)

                top, index_score = distance_calculation.KNN(df['All'], df2['All'], number_of_neighbors=100)
                knn = get_recommendation(top, df, index_score)

                merge1 = knn[['JobID', 'positionName', 'score']].merge(TF[['JobID', 'score']], on="JobID")
                final = merge1.merge(cv[['JobID', 'score']], on="JobID")
                final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})

                from sklearn.preprocessing import MinMaxScaler
                slr = MinMaxScaler()
                final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

                final['KNN'] = (1 - final['KNN']) / 3
                final['TF-IDF'] = final['TF-IDF'] / 3
                final['CV'] = final['CV'] / 3
                final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']
                final.sort_values(by="Final", ascending=False)

                final2 = final.sort_values(by="Final", ascending=False).copy()
                final_df = df.merge(final2, on="JobID")
                final_df = final_df.sort_values(by="Final", ascending=False)
                final_df.fillna('Not Available', inplace=True)

                final_jobrecomm = final_df.head(no_of_jobs)

                final_jobrecomm = final_jobrecomm.replace(np.nan, "Not Provided")

                @st.cache_data
                def make_clickable(link):
                    return link

                final_jobrecomm['url'] = final_jobrecomm['url'].apply(make_clickable)
                final_df = final_jobrecomm[['company', 'positionName_x', 'description', 'location', 'salary', 'url']]
                final_df.rename({'company': 'Company', 'positionName_x': 'Position Name', 'description': 'Job Description', 'location': 'Location', 'salary': 'Salary', 'url': 'Indeed Apply Link'}, axis=1, inplace=True)

                st.write("### Job Recommendations")
                st.dataframe(final_df)

                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')

                placeholder.empty()  

            except Exception as e:
                raise jobException(e, sys)

if __name__ == '__main__':
    app()
