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

dataBase = "Job_Hunter_DB"
collection = "Resume_Data"
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RESUME ANALYZER")

add_logo()
sidebar()

def course_recommender(course_list):
    st.subheader("*Courses & Certificates🎓 Recommendations*")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

def parse_resume_improved(text):
    sections = {
        'projects': [],
        'career_objective': [],
        'achievements': [],
        'declaration': [],
        'hobbies': []
    }
    
    # Define keywords for each section
    keywords = {
        'projects': ['projects', 'project'],
        'career_objective': ['career objective', 'objective'],
        'achievements': ['achievements', 'accomplishments', 'awards', 'seminars', 'trainings', 'education', 'educational background'],
        'declaration': ['declaration'],
        'hobbies': ['hobbies', 'interests', 'extracurricular activities', 'activities', 'pastime', 'leisure']
    }
    
    # Split the text into lines
    lines = text.split('\n')
    
    # Initialize current section
    current_section = None
    section_order = list(sections.keys())
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if the line indicates a new section
        new_section_detected = False
        for section, kw_list in keywords.items():
            if any(kw in line_lower for kw in kw_list) and len(line_lower.split()) <= 4:
                current_section = section
                new_section_detected = True
                break
        
        # If a new section is detected, move to the next iteration
        if new_section_detected:
            continue
        
        # Add line to the current section if it is set
        if current_section:
            sections[current_section].append(line.strip())
    
    # Clean and join lines to form the content for each section
    for section in sections:
        if sections[section]:
            # Remove any lines that are not relevant (e.g., contact info, addresses)
            filtered_lines = [line for line in sections[section] if line]
            sections[section] = ' '.join(filtered_lines).strip()
        else:
            sections[section] = None
    
    return sections

def run():
    st.title("Resume Analyzer")

    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
    if pdf_file is not None:
        count_ = 0

        encoded_pdf = utils.pdf_to_base64(pdf_file)
        resume_data = ResumeParser(pdf_file).get_extracted_data()

        resume_data["pdf_to_base64"] = encoded_pdf

        if resume_data:
            resume_text = utils.pdf_reader(pdf_file)
            parsed_resume = parse_resume_improved(resume_text)

            st.header("Resume Analysis")

            try:
                st.success("Hello " + resume_data['name'])
                st.subheader("Your Basic info")
                st.text('Name: ' + resume_data['name'])
                st.text('Email: ' + resume_data['email'])
                st.text('Contact: ' + resume_data['mobile_number'])
                st.text('Resume pages: ' + str(resume_data['no_of_pages']))
            except:
                pass

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

            st.subheader("**Skills Recommendation💡**")
            keywords = st_tags(label='### Skills that you have', text='See our skills recommendation', value=resume_data['skills'], key='1')

            recommended_skills = []
            reco_field = ''
            rec_course = ''
            for i in resume_data['skills']:
                if i.lower() in ds_keyword:
                    print(i.lower())
                    reco_field = 'Data Science'
                    st.success("** Our analysis says you are looking for Data Science Jobs.**")
                    recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask", 'Streamlit']
                    recommended_keywords = st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='2')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to resume will boost🚀 the chances of getting a Job💼</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(ds_course)
                    break
                elif i.lower() in web_keyword:
                    print(i.lower())
                    reco_field = 'Web Development'
                    st.success("** Our analysis says you are looking for Web Development Jobs **")
                    recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                    recommended_keywords = st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='3')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to resume will boost🚀 the chances of getting a Job💼</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(web_course)
                    break
                elif i.lower() in android_keyword:
                    print(i.lower())
                    reco_field = 'Android Development'
                    st.success("** Our analysis says you are looking for Android App Development Jobs **")
                    recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java', 'Kivy', 'GIT', 'SDK', 'SQLite']
                    recommended_keywords = st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='4')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to resume will boost🚀 the chances of getting a Job💼</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(android_course)
                    break
                elif i.lower() in ios_keyword:
                    print(i.lower())
                    reco_field = 'IOS Development'
                    st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                    recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation', 'Auto-Layout']
                    recommended_keywords = st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='5')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to resume will boost🚀 the chances of getting a Job💼</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(ios_course)
                    break
                elif i.lower() in uiux_keyword:
                    print(i.lower())
                    reco_field = 'UI-UX Development'
                    st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                    recommended_skills = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes', 'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator', 'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro', 'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp', 'user research', 'user experience']
                    recommended_keywords = st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='6')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to resume will boost🚀 the chances of getting a Job💼</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(uiux_course)
                    break

            # inserting data into mongodb  
            timestamp = utils.generateUniqueFileName()
            save = {timestamp: resume_data}
            if count_ == 0:
                count_ = 1
                MongoDB_function.resume_store(save, dataBase, collection)

            ### Resume writing recommendation
            st.subheader("**Resume Tips & Ideas💡**")
            resume_score = 0

            if parsed_resume['career_objective']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Career Objective🎯</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation, please add your career objective. It will give your career intention to the Recruiters.</h4>''', unsafe_allow_html=True)

            if parsed_resume['declaration']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Declaration✍</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation, please add Declaration✍. It will give the assurance that everything written on your resume is true and fully acknowledged by you.</h4>''', unsafe_allow_html=True)

            if parsed_resume['hobbies']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies⚽</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation, please add Hobbies⚽. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''', unsafe_allow_html=True)

            if parsed_resume['achievements']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements🏅</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation, please add Achievements🏅. It will show that you are capable of the required position.</h4>''', unsafe_allow_html=True)

            if parsed_resume['projects']:
                resume_score += 20
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻</h4>''', unsafe_allow_html=True)
            else:
                st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation, please add Projects👨‍💻. It will show that you have done work related to the required position or not.</h4>''', unsafe_allow_html=True)

            st.subheader("**Resume Score📝**")
            st.markdown(
                """
                <style>
                    .stProgress > div > div > div > div {
                        background-color: #d73b5c;
                    }
                </style>""",
                unsafe_allow_html=True,
            )
            my_bar = st.progress(0)
            score = 0
            for percent_complete in range(resume_score):
                score += 1
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            st.success('** Your Resume Writing Score: ' + str(score) + '**')
            st.warning("** Note: This score is calculated based on the content that you have added in your Resume. **")
        else:
            st.error("Wrong ID & Password Provided")

run()
