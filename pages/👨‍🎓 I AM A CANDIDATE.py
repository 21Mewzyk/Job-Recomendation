import streamlit as st
import pandas as pd
import numpy as np
import base64
import os, sys
import pymongo
from pyresparser import ResumeParser
from JobRecommendation.exception import jobException
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation

dataBase = "Job-Recomendation"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_from_CANDIDATE"
resume_data_collection = "Resume_Data"  # New collection for storing resume data
cvs_folder = r"D:\Vscode_projects\Job-Recommendation\CVs"  # Absolute path to the CVs folder

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

add_logo()
sidebar()

# MongoDB connection setup
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI if different
db = client[dataBase]

def load_cvs_to_base64(folder_path):
    cv_base64 = {}
    try:
        # Log all files in the folder
        st.write(f"Debug: Files in {folder_path}: {os.listdir(folder_path)}")
        for cv_file in os.listdir(folder_path):
            if cv_file.endswith(".pdf"):
                with open(os.path.join(folder_path, cv_file), "rb") as pdf_file:
                    encoded_string = base64.b64encode(pdf_file.read()).decode('utf-8')
                    cv_base64[cv_file.lower()] = encoded_string  # Store filenames in lowercase
        st.write(f"Debug: Contents of cv_base64: {list(cv_base64.keys())}")  # Debug info
    except FileNotFoundError:
        st.error(f"Folder not found: {folder_path}")
    except Exception as e:
        st.error(f"An error occurred while loading CVs: {e}")
    return cv_base64

def save_uploaded_file(uploaded_file, folder_path):
    try:
        with open(os.path.join(folder_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"An error occurred while saving the file: {e}")
        return False

def sanitize_filename(name):
    # Function to sanitize filenames to match the files in the directory
    return name.replace(" ", "").replace(".", "").lower()

def app():
    st.title('Job Recommendation')
    c1, c2 = st.columns((3, 2))
    cv = c1.file_uploader('Upload your CV', type='pdf')
    no_of_jobs = st.slider('Maximum Number of Job Recommendations:', min_value=1, max_value=100, step=10)

    if cv is not None:
        if st.button('Proceed'):
            # Save the uploaded CV
            if save_uploaded_file(cv, cvs_folder):
                try:
                    # Process the saved CV
                    cv_path = os.path.join(cvs_folder, cv.name)
                    with open(cv_path, "rb") as pdf_file:
                        encoded_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                    
                    cv_text = utils.extract_data(cv_path)
                    resume_data = ResumeParser(cv_path).get_extracted_data()
                    resume_data["pdf_to_base64"] = encoded_pdf

                    # Check if the resume already exists in the database
                    existing_resume = db[resume_data_collection].find_one({"email": resume_data.get("email")})
                    if existing_resume:
                        st.info("Resume already exists in the database. Skipping save.")
                    else:
                        # Save resume data to MongoDB
                        db[resume_data_collection].insert_one(resume_data)
                        st.success(f"File {cv.name} saved successfully!")

                        timestamp = utils.generateUniqueFileName()
                        save = {timestamp: resume_data}
                        MongoDB_function.resume_store(save, dataBase, collection2)

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
                        text = 'more details'
                        return f'<a target="_blank" href="{link}">{text}</a>'

                    final_jobrecomm['externalApplyLink'] = final_jobrecomm['externalApplyLink'].apply(make_clickable)
                    final_jobrecomm['url'] = final_jobrecomm['url'].apply(make_clickable)
                    final_df = final_jobrecomm[['company', 'positionName_x', 'description', 'location', 'salary', 'rating', 'reviewsCount', "externalApplyLink", 'url']]
                    final_df.rename({'company': 'Company', 'positionName_x': 'Position Name', 'description': 'Job Description', 'location': 'Location', 'salary': 'Salary', 'rating': 'Company Rating', 'reviewsCount': 'Company ReviewCount', 'externalApplyLink': 'Web Apply Link', 'url': 'Indeed Apply Link'}, axis=1, inplace=True)

                    st.write("### Job Recommendations")
                    st.dataframe(final_df)

                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')
                except Exception as e:
                    raise jobException(e, sys)

if __name__ == '__main__':
    app()
