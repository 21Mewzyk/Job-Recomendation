import pymongo
import pandas as pd
import json
from JobRecommendation.config import client

# Provide the mongodb localhost url to connect python to mongodb.
db = client.test

# Update the file path to the correct location
DATA_FILE_PATH = "data/concatenated_data/all_locations.csv"
# Database Name
dataBase = "Job Hunter DB"
# Collection Name
collection = "all_locations_Data"

if __name__ == "__main__":
    # Test MongoDB connection
    try:
        print("Testing MongoDB connection...")
        client.server_info()  # This will throw an exception if not connected
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")

    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")
    # Resetting the index
    df.reset_index(drop=True, inplace=True)
    # Convert dataframe to json so that we can dump these records into MongoDB
    json_record = list(json.loads(df.T.to_json()).values())
    # Insert converted json record into MongoDB
    client[dataBase][collection].insert_many(json_record)
