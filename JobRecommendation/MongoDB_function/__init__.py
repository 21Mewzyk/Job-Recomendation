from JobRecommendation.config import client
import pandas as pd
from JobRecommendation.exception import jobException
import sys

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
