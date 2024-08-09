from pymongo import MongoClient

def count_documents_in_collection(db_name, collection_name):
    try:
        # Establish connection to MongoDB
        client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB connection string if different
        db = client[db_name]
        collection = db[collection_name]
        
        # Count documents in the collection
        document_count = collection.count_documents({})
        return document_count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    database_name = "Job_Hunter_DB"
    collection_name = "Resume_Data"
    
    count = count_documents_in_collection(database_name, collection_name)
    
    if count is not None:
        print(f"Number of documents in the '{collection_name}' collection: {count}")
    else:
        print("Failed to retrieve document count.")
