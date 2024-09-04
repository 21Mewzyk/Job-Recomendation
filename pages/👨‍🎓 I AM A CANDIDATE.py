import streamlit as st
import pandas as pd
import numpy as np
import re
import base64, hashlib
import os, sys
from pyresparser import ResumeParser
from JobRecommendation import utils, MongoDB_function, text_preprocessing, distance_calculation
from JobRecommendation.exception import jobException
from streamlit_lottie import st_lottie
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
import pymongo
from sklearn.preprocessing import MinMaxScaler
import json

dataBase = "Job_Hunter_DB"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_Data"
cv_save_folder = "D:/Vscode_projects/Job-Recommendation/CVs"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def generate_cv_hash(cv_content):
    return hashlib.md5(cv_content.encode('utf-8')).hexdigest()

def check_if_cv_exists(db, collection, cv_hash):
    return db[collection].find_one({'cv_hash': cv_hash})

def save_cv_to_folder(cv, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, cv.name)
    with open(file_path, "wb") as f:
        f.write(cv.getbuffer())
    return file_path

def extract_degree(cv_text):
    patterns = [
        r"\b(Bachelor(?:'s)? of [A-Za-z]+)\b", r"\b(Master(?:'s)? of [A-Za-z]+)\b", 
        r"\b(Doctor(?:ate)? of [A-Za-z]+)\b", r"\b(B\.?Sc\.?|M\.?Sc\.?)\b"
    ]
    matches = [re.findall(pat, cv_text, re.IGNORECASE) for pat in patterns]
    return ", ".join([item for sublist in matches for item in sublist])

def extract_college_name(cv_text):
    patterns = [r"\b(?:[A-Za-z\s]+University)\b", r"\b(?:[A-Za-z\s]+College)\b"]
    matches = [re.findall(pat, cv_text, re.IGNORECASE) for pat in patterns]
    return ", ".join([item for sublist in matches for item in sublist])

animation_data = load_lottiefile("D:/Vscode_projects/Job-Recommendation/Animations/Loading 2.json")
add_logo()
sidebar()

st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    st.title('Job Recommendation')
    c1, c2, c3 = st.columns((3, 2, 2))
    location_filter = c2.text_input('Enter preferred job location:', '')
    min_salary, max_salary = c3.slider('Select salary range (in PHP):', 25000, 500000, (25000, 500000), 1000)
    cv = c1.file_uploader('Upload your CV', type='pdf')
    no_of_jobs = st.slider('Max Number of Job Recommendations:', 1, 100, 1)

    if cv is not None:
        if st.button('Proceed'):
            placeholder = st.empty()
            with placeholder.container():
                st_lottie(animation_data, height=700, width=700)

            try:
                cv_text = utils.extract_data(cv)
                resume_data = ResumeParser(cv).get_extracted_data()
                resume_data["degree"] = resume_data.get("degree") or extract_degree(cv_text)
                resume_data["college_name"] = resume_data.get("college_name") or extract_college_name(cv_text)

                file_path = save_cv_to_folder(cv, cv_save_folder)
                cv_hash = generate_cv_hash(cv_text)
                db = MongoDB_function.get_database(dataBase)

                if not check_if_cv_exists(db, collection2, cv_hash):
                    new_cv_id = (db[collection2].find_one(sort=[("Unnamed: 0", pymongo.DESCENDING)]) or {}).get("Unnamed: 0", 0) + 1
                    resume_data["Unnamed: 0"] = new_cv_id
                    resume_data["All"] = " ".join(str(resume_data.get(key, "")) for key in resume_data)
                    db[collection2].insert_one(resume_data)
                    st.success("CV successfully uploaded to MongoDB.")

                NLP_Processed_CV = text_preprocessing.nlp(cv_text)
                df2 = pd.DataFrame({'All': " ".join(NLP_Processed_CV)}, index=[0])
                df = MongoDB_function.get_collection_as_dataframe(dataBase, collection1)

                output = distance_calculation.TFIDF(df['All'], df2['All'])
                top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:1000]
                final_recommendation = pd.DataFrame({
                    'JobID': df.index[top],
                    'positionName': df['positionName'][top],
                    'company': df['company'][top],
                    'location': df['location'][top],
                    'description': df['description'][top],
                    'salary': df['salary'][top],
                    'url': df['url'][top]  # Add the Indeed Apply Link column
                })

                final_recommendation['salary'] = pd.to_numeric(final_recommendation['salary'], errors='coerce')
                final_recommendation.rename(columns={'salary': 'Annual Salary in PHP'}, inplace=True)

                filtered_df = final_recommendation[
                    (final_recommendation['Annual Salary in PHP'] >= min_salary) & 
                    (final_recommendation['Annual Salary in PHP'] <= max_salary)
                ]
                if location_filter:
                    filtered_df = filtered_df[filtered_df['location'].str.contains(location_filter, case=False, na=False)]

                filtered_df = filtered_df.head(no_of_jobs)

                st.write("### Filtered Job Recommendations")
                if filtered_df.empty:
                    st.warning("No jobs found matching your criteria.")
                else:
                    st.dataframe(filtered_df)

                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button("Press to Download", csv, "file.csv", "text/csv")

                placeholder.empty()

            except Exception as e:
                raise jobException(e, sys)

if __name__ == '__main__':
    app()
