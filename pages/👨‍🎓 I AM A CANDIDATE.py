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
from JobRecommendation.animation import load_lottieurl
from streamlit_lottie import st_lottie, st_lottie_spinner
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation
from JobRecommendation.exception import jobException

dataBase = "Job-Recomendation"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_from_CANDIDATE"
collection3 = "all_locations_Data"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

url = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_x62chJ.json")
add_logo()
sidebar()

st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    st.title('Job Recommendation')
    c1, c2 = st.columns((3, 2))
    cv = c1.file_uploader('Upload your CV', type='pdf')
    job_loc = MongoDB_function.get_collection_as_dataframe(dataBase, collection3)
    all_locations = list(job_loc["location"].dropna().unique())

    RL = c2.multiselect('Filter', all_locations)

    no_of_jobs = st.slider('Number of Job Recommendations:', min_value=1, max_value=100, step=10)

    if cv is not None:
        if st.button('Proceed Further !! '):
            with st_lottie_spinner(url, key="download", reverse=True, speed=1, loop=True, quality='high'):
                time.sleep(10)
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

                    result_jd = final_df
                    if len(RL) == 0:
                        result_jd = final_df
                    else:
                        result_jd = result_jd[result_jd["location"].isin(list(RL))]

                    final_jobrecomm = result_jd.head(no_of_jobs)

                    db_expander = st.expander(label='Job Recommendations:')

                    final_jobrecomm = final_jobrecomm.replace(np.nan, "Not Provided")

                    @st.cache_data
                    def make_clickable(link):
                        if link.startswith('<a'):
                            return link
                        return f'<a target="_blank" href="{link}">more details</a>'

                    with db_expander:
                        def convert_df(df):
                            try:
                                return df.to_csv(index=False).encode('utf-8')
                            except Exception as e:
                                raise jobException(e, sys)

                        # Ensure columns exist before applying make_clickable
                        if 'externalApplyLink' in final_jobrecomm.columns:
                            final_jobrecomm['externalApplyLink'] = final_jobrecomm['externalApplyLink'].apply(make_clickable)
                        else:
                            final_jobrecomm['externalApplyLink'] = 'Not Provided'

                        if 'url' in final_jobrecomm.columns:
                            final_jobrecomm['url'] = final_jobrecomm['url'].apply(make_clickable)
                        else:
                            final_jobrecomm['url'] = 'Not Provided'

                        final_df = final_jobrecomm[['company', 'positionName_x', 'description', 'location', 'salary', 'rating', 'reviewsCount', "externalApplyLink", 'url']]
                        final_df.rename({'company': 'Company', 'positionName_x': 'Position Name', 'description': 'Job Description', 'location': 'Location', 'salary': 'Salary', 'rating': 'Company Rating', 'reviewsCount': 'Company ReviewCount', 'externalApplyLink': 'Web Apply Link', 'url': 'Indeed Apply Link'}, axis=1, inplace=True)
                        
                        # Display job recommendations in a grid format with buttons
                        no_of_cols = 3  # Number of columns for the grid
                        cols = st.columns(no_of_cols)
                        for idx, row in final_df.iterrows():
                            col_idx = idx % no_of_cols
                            with cols[col_idx]:
                                st.markdown(f"**{row['Position Name']}**")
                                st.markdown(f"**Company:** {row['Company']}")
                                st.markdown(f"**Location:** {row['Location']}")
                                st.markdown(f"**Salary:** {row['Salary']}")
                                st.markdown(f"**Rating:** {row['Company Rating']}")
                                st.markdown(f"**Reviews Count:** {row['Company ReviewCount']}")
                                st.markdown(f"**Job Description:** {row['Job Description']}")
                                st.markdown(f'<a href="{row["Web Apply Link"]}" target="_blank"><button style="background-color:blue;color:white;border:none;padding:10px 20px">More Details</button></a>', unsafe_allow_html=True)
                                st.markdown(f'<a href="{row["Indeed Apply Link"]}" target="_blank"><button style="background-color:blue;color:white;border:none;padding:10px 20px">Indeed Link</button></a>', unsafe_allow_html=True)
                                st.markdown("---")

                    csv = convert_df(final_df)
                    st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')
                    st.balloons()
                except Exception as e:
                    raise jobException(e, sys)

if __name__ == '__main__':
    app()
