from JobRecommendation.config import client
import pandas as pd
from JobRecommendation.exception import jobException
import sys
from pymongo import MongoClient

def get_database(database_name: str):
    """
    Description: This function returns the MongoDB database connection.
    =========================================================
    Params:
    database_name: database name
    =========================================================
    return MongoDB database object
    """
    try:
        return client[database_name]
    except Exception as e:
        raise jobException(e, sys)

def get_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Description: This function returns a collection as a dataframe.
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return: Pandas dataframe of a collection
    """
    try:
        df = pd.DataFrame(list(client[database_name][collection_name].find()))
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df
    except Exception as e:
        raise jobException(e, sys)

def resume_store(data, database_name: str, collection_name: str):
    """
    Description: This function stores the resume data into the MongoDB collection.
    =========================================================
    Params:
    data: The resume data to store.
    database_name: The name of the database.
    collection_name: The name of the collection.
    =========================================================
    return: None
    """
    try:
        client[database_name][collection_name].insert_one(data)
    except Exception as e:
        raise jobException(e, sys)

# New functions added for fetching data and ranking CVs

def fetch_data_from_mongo(collection_name, db_name="Job_Hunter_DB"):
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string if different
    db = client[db_name]
    collection = db[collection_name]
    
    # Fetch the data and convert it into a DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    # Drop any unnecessary columns (like '_id', 'Unnamed: 0')
    df.drop(columns=['_id', 'Unnamed: 0'], errors='ignore', inplace=True)
    
    return df

def calculate_skill_matches(skills_list, job_skills):
    matches = sum(1 for skill in skills_list if skill in job_skills and pd.notna(skill))
    return matches

def rank_cvs_by_skill_matches(df, job_skills):
    df['match_count'] = df.apply(lambda row: calculate_skill_matches(
        [row[f"skills[{i}]"] for i in range(5)], job_skills), axis=1)
    
    df_sorted = df.sort_values(by='match_count', ascending=False)
    return df_sorted

def main():
    # Example job description skills (adjust according to your actual job description)
    job_description_skills = ["Marketing", "SEO", "Social Media", "Content Creation", "Advertising"]
    
    # Fetch data from MongoDB
    df = fetch_data_from_mongo("Resume_Data")
    
    # Rank CVs by the number of matching skills
    ranked_cvs = rank_cvs_by_skill_matches(df, job_description_skills)
    
    # Print or return the ranked CVs
    print(ranked_cvs[['name', 'match_count']])
    return ranked_cvs

if __name__ == "__main__":
    main()
