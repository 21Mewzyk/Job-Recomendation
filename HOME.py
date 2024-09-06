import streamlit as st 
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
import altair as alt
import plotly.express as px 
from streamlit_extras.switch_page_button import switch_page
import pandas as pd 
import numpy as np 
from datetime import datetime
from streamlit_lottie import st_lottie
import json

# Function to load Lottie animation file for visual display
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the Lottie animation (AI Robot) for the page
lottie_animation = load_lottiefile("D:\Vscode_projects\Job-Recommendation\Animations\AI Robot.json")

# Add logo and sidebar to the app
add_logo()
sidebar()

# Add a main welcome heading styled with custom HTML and CSS
st.markdown("<h1 style='text-align: center; font-family: monospace, sans-serif; padding: 20px; border: 2px solid #758283; border-radius: 5px;'>Welcome to Job Hunter !</h1>", unsafe_allow_html=True)

# Section for "What We Offer" with a heading styled using custom HTML
st.markdown("<div style='background-color: rgba(255, 0, 0, 0); padding: 10px;'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-family: monospace, sans-serif; padding: 20px;'>WHAT WE OFFER : </h2>", unsafe_allow_html=True)

# Three columns for navigation buttons
s1,s2, s3 = st.columns(3)

# First column for Job Recommendation button
with s1:
    candidate = st.button("Job Recommendation")
    if candidate:
        switch_page("i am a candidate")  # Switches to the candidate page when the button is clicked
    
# Second column for Resume Analyzer button
with s2:
    analyzer = st.button("Resume Analyzer")
    if analyzer:
        switch_page("resume analyzer")  # Switches to the resume analyzer page when clicked

# Third column for Candidate Recommendation button
with s3:
    recruiter = st.button("Candidate Recommendation")
    if recruiter:
        switch_page("i am a recruiter")  # Switches to the recruiter page when clicked

# Closing the previous styled div tag
st.markdown("</div>", unsafe_allow_html=True)

# Display Lottie animation in the center of the page
st_lottie(lottie_animation)

# Section explaining why users should choose "Job Hunter"
st.markdown("<h2 style='text-align: center; font-family: monospace, sans-serif; padding: 20px;'>Why Job Hunter ?</h2>", unsafe_allow_html=True)
st.write("<p style='font-size:20px;'>Job seekers and recruiters struggle to find the right match for open job positions, leading to a time-consuming and inefficient recruitment process. Job Hunter offers a solution to this problem with its advanced technologies that provide personalized job and candidate recommendations based on qualifications and experience.</p>", unsafe_allow_html=True)

# Section describing the aim of the platform
st.markdown("<h2 style='text-align: center; font-family: monospace, sans-serif; padding: 20px;'>AIM</h2>", unsafe_allow_html=True)
st.write("<p style='font-size:20px;'>The job search process can be daunting and time-consuming for both job seekers and recruiters. That's where this system comes in!</p>", unsafe_allow_html=True)
st.write("<p style='font-size:20px;'>Job Hunter is designed to assist applicants in searching for potential jobs and to help recruiters find talented candidates. Job Hunter offers a user-friendly interface that allows applicants to easily browse and search for job opportunities based on their preferences and qualifications. Jobseekers can upload their CVs to get tailored and precise job recommendations that match their skillset. Job Hunter also provides helpful tips and resources for applicants, such as Resume Analyzer and tips to make your Resume even better.</p>", unsafe_allow_html=True)
