from pymongo import MongoClient

# Replace with your MongoDB connection string
client = MongoClient('mongodb://localhost:27017/')
db = client['Job-Recomendation']  # Replace with your database name
collection = db['Resume_Data']  # Replace with your collection name

# Fields to update and their default values
fields_to_update = {
    'college_name': 'University of the Cordilleras',
    'degree': 'Bachelor of Science in Data Analytics',
    'designation': 'Data Science',
    'experience': 'Not Provided',
    'company_names': 'Not Provided'
}

for field, default_value in fields_to_update.items():
    collection.update_many(
        { field: None },
        { '$set': { field: default_value } }
    )

print("Null values updated successfully.")
