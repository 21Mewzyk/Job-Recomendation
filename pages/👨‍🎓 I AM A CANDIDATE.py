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
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

# MongoDB configuration
dataBase = "Job_Hunter_DB"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_Data"
cv_save_folder = "D:/Vscode_projects/Job-Recommendation/CVs"

# Encryption configuration
ENCRYPTION_PASSWORD = b"your-strong-password"  # Password for encryption
SALT = b'\x00\x01\x02\x03\x04\x05\x06\x07'  # A salt for deriving the key (use a more secure method in production)

# Configure the Streamlit page layout and title
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

# Load Lottie animation JSON file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Derive an encryption key from the password using PBKDF2HMAC
def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password)

# AES encryption
def encrypt_data(data, key):
    iv = os.urandom(16)  # Initialization vector
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    
    padded_data = padder.update(data) + padder.finalize()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    # Return the encrypted data with the IV prepended (needed for decryption)
    return base64.b64encode(iv + encrypted_data).decode('utf-8')

# AES decryption
def decrypt_data(encrypted_data, key):
    encrypted_data = base64.b64decode(encrypted_data)
    iv = encrypted_data[:16]  # Extract the IV from the start
    encrypted_data = encrypted_data[16:]
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()
    
    return decrypted_data

# Generate a unique hash for the CV content to avoid duplicates
def generate_cv_hash(cv_content):
    return hashlib.md5(cv_content.encode('utf-8')).hexdigest()

# Check if a CV with the same hash already exists in the MongoDB collection
def check_if_cv_exists(db, collection, cv_hash):
    return db[collection].find_one({'cv_hash': cv_hash})

# Check if the CV file already exists locally by comparing the file hash
def check_if_cv_exists_in_folder(folder_path, cv_hash):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            file_content = file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            if file_hash == cv_hash:
                return True  # Duplicate found in folder
    return False

# Save the uploaded CV to the specified folder
def save_cv_to_folder(cv, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist
    file_path = os.path.join(folder_path, cv.name)
    with open(file_path, "wb") as f:
        f.write(cv.getbuffer())
    return file_path

# Convert the CV file to base64 encoding, then encrypt it to store in MongoDB
def cv_to_base64_and_encrypt(cv):
    # Convert CV to base64
    cv_base64 = base64.b64encode(cv.getvalue()).decode('utf-8')
    
    # Derive the encryption key from the password and salt
    key = derive_key(ENCRYPTION_PASSWORD, SALT)
    
    # Encrypt the base64 CV data
    encrypted_cv = encrypt_data(cv_base64.encode('utf-8'), key)
    
    return encrypted_cv

# Extract the degree information from the CV text using regex patterns
def extract_degree(cv_text):
    patterns = [
        r"\b(Bachelor(?:'s)? of [A-Za-z]+)\b", r"\b(Master(?:'s)? of [A-Za-z]+)\b", 
        r"\b(Doctor(?:ate)? of [A-Za-z]+)\b", r"\b(B\.?Sc\.?|M\.?Sc\.?)\b"
    ]
    matches = [re.findall(pat, cv_text, re.IGNORECASE) for pat in patterns]
    return ", ".join([item for sublist in matches for item in sublist])

# Extract the college or university name from the CV text using regex patterns
def extract_college_name(cv_text):
    patterns = [r"\b(?:[A-Za-z\s]+University)\b", r"\b(?:[A-Za-z\s]+College)\b"]
    matches = [re.findall(pat, cv_text, re.IGNORECASE) for pat in patterns]
    return ", ".join([item for sublist in matches for item in sublist])

# Load and set up the Lottie animation for the UI
animation_data = load_lottiefile("D:/Vscode_projects/Job-Recommendation/Animations/Loading 2.json")

# Add logos and sidebar to the Streamlit app
add_logo()
sidebar()

# Set an option to ignore Streamlit deprecation warnings for Pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    # Display the page title
    st.title('Job Recommendation')
    
    # Switches to show or hide location and salary filters
    show_location_filter = True
    show_salary_filter = True

    # Create columns for layout
    c1, c2, c3 = st.columns((3, 2, 2))

    # Conditionally display the location filter (text input)
    if show_location_filter:
        location_filter = c2.text_input('Enter preferred job location:', '')
    else:
        location_filter = ''  # No filter if the input is hidden

    # Conditionally display the salary range slider
    if show_salary_filter:
        min_salary, max_salary = c3.slider('Select salary range (in PHP):', 25000, 500000, (25000, 500000), 1000)
    else:
        min_salary, max_salary = None, None  # No filter if the slider is hidden

    # File uploader for uploading the CV in PDF format
    cv = c1.file_uploader('Upload your CV', type='pdf')

    # Slider for setting the number of job recommendations to retrieve
    no_of_jobs = st.slider('Max Number of Job Recommendations:', 1, 100, 1)

    if cv is not None:  # If a CV is uploaded
        if st.button('Proceed'):  # Proceed button triggers the process
            placeholder = st.empty()  # Placeholder for the loading animation
            with placeholder.container():
                st_lottie(animation_data, height=700, width=700)  # Show animation

            try:
                # Extract text from the uploaded CV
                cv_text = utils.extract_data(cv)

                # Extract structured resume data from the CV
                resume_data = ResumeParser(cv).get_extracted_data()
                resume_data["degree"] = resume_data.get("degree") or extract_degree(cv_text)
                resume_data["college_name"] = resume_data.get("college_name") or extract_college_name(cv_text)

                # Generate a unique hash for the CV to avoid duplicates
                cv_hash = generate_cv_hash(cv_text)

                # Check if the CV already exists in the local folder
                if check_if_cv_exists_in_folder(cv_save_folder, cv_hash):
                    st.warning("This CV already exists in the folder.")
                    return

                # Save the CV to the folder
                file_path = save_cv_to_folder(cv, cv_save_folder)

                # Convert and encrypt CV to base64 for storing in MongoDB
                cv_base64 = cv_to_base64_and_encrypt(cv)

                # Connect to MongoDB
                db = MongoDB_function.get_database(dataBase)

                # Check if the CV already exists in MongoDB
                if not check_if_cv_exists(db, collection2, cv_hash):
                    # Insert new CV into MongoDB if it's not a duplicate
                    new_cv_id = (db[collection2].find_one(sort=[("Unnamed: 0", pymongo.DESCENDING)]) or {}).get("Unnamed: 0", 0) + 1
                    resume_data["Unnamed: 0"] = new_cv_id
                    resume_data["All"] = " ".join(str(resume_data.get(key, "")) for key in resume_data)
                    resume_data["cv_hash"] = cv_hash
                    resume_data["pdf_to_base64"] = cv_base64

                    db[collection2].insert_one(resume_data)
                    st.success("")
                else:
                    # If the CV exists but doesn't have base64 data, add it
                    existing_record = db[collection2].find_one({"cv_hash": cv_hash})
                    if 'pdf_to_base64' not in existing_record:
                        db[collection2].update_one(
                            {'cv_hash': cv_hash},
                            {'$set': {'pdf_to_base64': cv_base64}}
                        )
                        st.success("")
                    else:
                        st.warning("")

                # Process the CV text for job recommendations
                NLP_Processed_CV = text_preprocessing.nlp(cv_text)
                df2 = pd.DataFrame({'All': " ".join(NLP_Processed_CV)}, index=[0])

                # Fetch the job data from MongoDB
                df = MongoDB_function.get_collection_as_dataframe(dataBase, collection1)

                # Perform TF-IDF calculation to find matching jobs
                output = distance_calculation.TFIDF(df['All'], df2['All'])
                top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:1000]

                # Create a DataFrame of the top job recommendations
                final_recommendation = pd.DataFrame({
                    'JobID': df.index[top],
                    'positionName': df['positionName'][top],
                    'company': df['company'][top],
                    'location': df['location'][top],
                    'description': df['description'][top],
                    'salary': df['salary'][top],
                    'url': df['url'][top] 
                })

                # Convert salary column to numeric and rename
                final_recommendation['salary'] = pd.to_numeric(final_recommendation['salary'], errors='coerce')
                final_recommendation.rename(columns={'salary': 'Annual Salary in PHP'}, inplace=True)

                # Apply filters based on user inputs (if available)
                apply_salary_filter = show_salary_filter and (min_salary is not None and max_salary is not None)
                apply_location_filter = show_location_filter and location_filter != ''

                # Filter jobs by salary if salary filter is applied
                if apply_salary_filter:
                    filtered_df = final_recommendation[
                        (final_recommendation['Annual Salary in PHP'] >= min_salary) & 
                        (final_recommendation['Annual Salary in PHP'] <= max_salary)
                    ]
                else:
                    filtered_df = final_recommendation

                # Filter jobs by location if location filter is applied
                if apply_location_filter:
                    filtered_df = filtered_df[filtered_df['location'].str.contains(location_filter, case=False, na=False)]

                # Limit the number of job recommendations based on user input
                filtered_df = filtered_df.head(no_of_jobs)

                # Display the filtered job recommendations
                st.write("### Filtered Job Recommendations")
                if filtered_df.empty:
                    st.warning("No jobs found matching your criteria.")
                else:
                    st.dataframe(filtered_df)

                # Allow the user to download the filtered job list as a CSV file
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button("Press to Download", csv, "file.csv", "text/csv")

                # Clear the placeholder to stop the loading animation
                placeholder.empty()

            except Exception as e:
                raise jobException(e, sys)

# Run the app function when the script is executed
if __name__ == '__main__':
    app()
