import os
from pyresparser import ResumeParser
import pandas as pd
from pymongo import MongoClient

def insert_resume_to_mongodb(file_path, database_name, collection_name):
    resume_data = ResumeParser(file_path).get_extracted_data()
    
    # Combine relevant fields into 'All' column, handle lists and None values appropriately
    def join_list_field(field):
        return ' '.join(field) if isinstance(field, list) else str(field)

    def handle_none(field, default=''):
        return str(field) if field is not None else default

    all_text = ' '.join([
        handle_none(resume_data.get('name', 'No Name')),
        handle_none(resume_data.get('email', 'No Email')),
        handle_none(resume_data.get('mobile_number', 'No Mobile Number')),
        handle_none(join_list_field(resume_data.get('skills', [])), 'No Skills'),
        handle_none(resume_data.get('college_name', 'No College Name')),
        handle_none(resume_data.get('degree', 'No Degree')),
        handle_none(join_list_field(resume_data.get('experience', [])), 'No Experience'),
        handle_none(join_list_field(resume_data.get('company_names', [])), 'No Company Names'),
        handle_none(resume_data.get('total_experience', '0'))
    ])
    
    resume_data['All'] = all_text
    
    # Connect to MongoDB and insert data
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]
    collection.insert_one(resume_data)

def process_resumes_in_directory(directory_path, database_name, collection_name):
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory_path, file_name)
            insert_resume_to_mongodb(file_path, database_name, collection_name)
            print(f"Inserted {file_name} into MongoDB.")

# Example usage
process_resumes_in_directory('D:\\CVs', 'Job-Recomendation', 'Resume_Data')
