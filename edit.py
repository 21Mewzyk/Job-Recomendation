from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/?authSource=admin")
db = client["Job-Recomendation"]
collection = db["Resume_from_CANDIDATE"]

# Function to concatenate various fields into the "All" column
def calculate_all_column(doc):
    # Adjust the fields as per your document structure
    fields = [
        doc.get('id', ''),
        doc.get('name', ''),
        doc.get('job_title', ''),
        doc.get('responsibilities', ''),
        doc.get('basic_qualifications', ''),
        doc.get('knowledge_skills', ''),
        doc.get('competencies', ''),
        doc.get('location', ''),
        doc.get('rating', ''),
        doc.get('salary', ''),
        doc.get('posted', ''),
        doc.get('apply_link', ''),
        doc.get('company_link', '')
    ]
    # Concatenate the fields into a single string
    all_column = ', '.join(fields)
    return all_column

# Fetch all documents from the collection
documents = list(collection.find())

# Update each document with the "All" column
for doc in documents:
    all_column_value = calculate_all_column(doc)
    collection.update_one({"_id": doc["_id"]}, {"$set": {"All": all_column_value}})

print("All documents updated with the 'All' column.")
