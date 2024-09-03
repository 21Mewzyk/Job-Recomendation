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
import hashlib

dataBase = "Job_Hunter_DB"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_Data"

cv_save_folder = "D:/Vscode_projects/Job-Recommendation/CVs"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def generate_cv_hash(cv_content):
    """Generate a hash for the CV content to check for duplicates."""
    return hashlib.md5(cv_content.encode('utf-8')).hexdigest()

def check_if_cv_exists(db, collection, cv_hash):
    """Check if a CV with the same hash already exists in the database."""
    return db[collection].find_one({'cv_hash': cv_hash})

def save_cv_to_folder(cv, folder_path):
    """Save the uploaded CV to the specified folder and return the file path."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = f"{cv.name}"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "wb") as f:
        f.write(cv.getbuffer())
    return file_path

animation_file = "D:/Vscode_projects/Job-Recommendation/Animations/Loading 2.json"  # Update with your lottie file location.
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
                st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                st_lottie(animation_data, height=700, width=700, key="download", reverse=True, speed=1, loop=True, quality='high')
                st.markdown("</div>", unsafe_allow_html=True)

            try:
                count_ = 0
                cv_text = utils.extract_data(cv)
                encoded_pdf = utils.pdf_to_base64(cv)
                resume_data = ResumeParser(cv).get_extracted_data()
                resume_data["pdf_to_base64"] = encoded_pdf

                # Save the CV to the specified folder and get the file path
                file_path = save_cv_to_folder(cv, cv_save_folder)
                resume_data["file_path"] = file_path

                # Generate a hash for the CV content
                cv_hash = generate_cv_hash(cv_text)
                resume_data["cv_hash"] = cv_hash

                # Safely concatenate all extracted data into a single string for the 'All' column
                all_info = " ".join([
                    str(resume_data.get("name", "") or ""),
                    str(resume_data.get("email", "") or ""),
                    str(file_path or ""),  # Include the file path after the email
                    str(resume_data.get("phone", "") or ""),
                    str(resume_data.get("education", "") or ""),
                    str(resume_data.get("experience", "") or ""),
                    str(resume_data.get("skills", "") or ""),
                    str(resume_data.get("summary", "") or ""),
                    str(cv_text or "")
                ]).strip()

                # Connect to MongoDB
                db = MongoDB_function.get_database(dataBase)

                # Check if the CV already exists in the database
                if not check_if_cv_exists(db, collection2, cv_hash):
                    timestamp = utils.generateUniqueFileName()
                    save = {timestamp: resume_data}
                    if count_ == 0:
                        count_ = 1
                    last_cv = db[collection2].find_one(sort=[("Unnamed: 0", pymongo.DESCENDING)])
                    if last_cv:
                        new_cv_id = last_cv["Unnamed: 0"] + 1
                    else:
                        new_cv_id = 1

                    resume_data["Unnamed: 0"] = new_cv_id
                    resume_data["Unnamed: 0"] = int(resume_data["Unnamed: 0"])

                    # Reorder the fields without manually setting _id, MongoDB will handle it
                    ordered_resume_data = {}
                    ordered_resume_data["Unnamed: 0"] = resume_data["Unnamed: 0"]
                    for key in resume_data:
                        if key not in ["_id", "Unnamed: 0"]:
                            if key == "email":
                                ordered_resume_data["email"] = resume_data["email"]
                                ordered_resume_data["file_path"] = resume_data["file_path"]
                            else:
                                ordered_resume_data[key] = resume_data[key]
                    ordered_resume_data["All"] = all_info

                    # Insert the ordered document into the database
                    db[collection2].insert_one(ordered_resume_data)
                    st.success("CV successfully uploaded to MongoDB.")

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
