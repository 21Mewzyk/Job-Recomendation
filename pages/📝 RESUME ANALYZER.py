import streamlit as st
import pandas as pd
import base64, random
import time, datetime
from pyresparser import ResumeParser
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymongo
import plotly.express as px
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation.courses import ds_course, web_course, android_course, ios_course, uiux_course, ds_keyword, web_keyword, android_keyword, ios_keyword, uiux_keyword
from JobRecommendation import utils, MongoDB_function

# MongoDB configuration
dataBase = "Job_Hunter_DB"
collection = "Resume_Data"

# Set page configuration
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RESUME ANALYZER")

# Add logos and sidebar to the app
add_logo()
sidebar()

# Function to recommend courses based on selected skills
def course_recommender(course_list):
    st.subheader("*Courses & Certificates🎓 Recommendations*")
    c = 0
    rec_course = []  # List to store recommended courses
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)  # Randomize course order
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")  # Display course name and link
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

# Function to parse a resume into different sections like projects, career objectives, etc.
def parse_resume_improved(text):
    sections = {
        'projects': [],
        'career_objective': [],
        'achievements': [],
        'declaration': [],
        'hobbies': []
    }
    
    # Define keywords for detecting sections in the resume
    keywords = {
        'projects': ['projects', 'project'],
        'career_objective': ['career objective', 'objective'],
        'achievements': ['achievements', 'accomplishments', 'awards', 'seminars', 'trainings', 'education', 'educational background'],
        'declaration': ['declaration'],
        'hobbies': ['hobbies', 'interests', 'extracurricular activities', 'activities', 'pastime', 'leisure']
    }
    
    lines = text.split('\n')  # Split resume text by lines
    current_section = None  # Track the current section while parsing

    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect new sections based on keywords
        new_section_detected = False
        for section, kw_list in keywords.items():
            if any(kw in line_lower for kw in kw_list) and len(line_lower.split()) <= 4:
                current_section = section
                new_section_detected = True
                break
        
        if new_section_detected:
            continue

        # Add content to the current section
        if current_section:
            sections[current_section].append(line.strip())

    # Clean up the sections, join lines into single text
    for section in sections:
        if sections[section]:
            filtered_lines = [line for line in sections[section] if line]
            sections[section] = ' '.join(filtered_lines).strip()
        else:
            sections[section] = None

    return sections

# Main function to run the app
def run():
    st.title("Resume Analyzer")  # Display app title

    # Upload resume file
    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
    if pdf_file is not None:
        count_ = 0

        # Convert the resume to base64 encoding
        encoded_pdf = utils.pdf_to_base64(pdf_file)

        # Extract data from the resume using ResumeParser
        resume_data = ResumeParser(pdf_file).get_extracted_data()
        resume_data["pdf_to_base64"] = encoded_pdf

        if resume_data:
            # Extract resume content as text and parse it into sections
            resume_text = utils.pdf_reader(pdf_file)
            parsed_resume = parse_resume_improved(resume_text)

            st.header("Resume Analysis")  # Section for resume analysis

            try:
                # Display basic info about the candidate
                st.success("Hello " + resume_data['name'])
                st.subheader("Your Basic info")
                st.text('Name: ' + resume_data['name'])
                st.text('Email: ' + resume_data['email'])
                st.text('Contact: ' + resume_data['mobile_number'])
                st.text('Resume pages: ' + str(resume_data['no_of_pages']))
            except:
                pass

            # Determine the candidate's experience level based on resume length
            cand_level = ''
            if resume_data['no_of_pages'] == 1:
                cand_level = "Fresher"
                st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''', unsafe_allow_html=True)
            elif resume_data['no_of_pages'] == 2:
                cand_level = "Intermediate"
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''', unsafe_allow_html=True)
            elif resume_data['no_of_pages'] >= 3:
                cand_level = "Experienced"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!</h4>''', unsafe_allow_html=True)

            # Skills recommendation section
            st.subheader("**Skills Recommendation💡**")
            keywords = st_tags(label='### Skills that you have', text='See our skills recommendation', value=resume_data['skills'], key='1')

            recommended_skills = []
            reco_field = ''
            rec_course = ''

            # Check the skills and recommend relevant fields and skills
            for i in resume_data['skills']:
                if i.lower() in ds_keyword:
                    reco_field = 'Data Science'
                    st.success("** Our analysis says you are looking for Data Science Jobs.**")
                    recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 'Clustering & Classification', 'Data Analytics', 'ML Algorithms']
                    recommended_keywords = st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='2')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to resume will boost🚀 the chances of getting a Job💼</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(ds_course)
                    break
                # Similar blocks for other fields (Web, Android, iOS, UI/UX)
                # Additional field checks for web_keyword, android_keyword, ios_keyword, uiux_keyword

            # Store resume data in MongoDB with a timestamp
            timestamp = utils.generateUniqueFileName()
            save = {timestamp: resume_data}
            if count_ == 0:
                count_ = 1
                MongoDB_function.resume_store(save, dataBase, collection)

            # Resume tips based on parsed sections
            st.subheader("**Resume Tips & Ideas💡**")
            resume_score = 0

            if parsed_resume['career_objective']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Career Objective🎯</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] Please add your career objective.</h4>''', unsafe_allow_html=True)

            if parsed_resume['declaration']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Declaration✍</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] Please add Declaration✍.</h4>''', unsafe_allow_html=True)

            if parsed_resume['hobbies']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies⚽</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] Please add Hobbies⚽.</h4>''', unsafe_allow_html=True)

            if parsed_resume['achievements']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements🏅</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] Please add Achievements🏅.</h4>''', unsafe_allow_html=True)

            if parsed_resume['projects']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] Please add Projects👨‍💻.</h4>''', unsafe_allow_html=True)

            # Display resume score based on content
            st.subheader("**Resume Score📝**")
            my_bar = st.progress(0)
            score = 0
            for percent_complete in range(resume_score):
                score += 1
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            st.success('** Your Resume Writing Score: ' + str(score) + '**')
            st.warning("** Note: This score is based on the content in your Resume. **")
        else:
            st.error("Error reading the resume data")

# Run the app
run()
