# Use the official Python image from the Docker Hub
FROM python:3.8

# Set the working directory in the container
WORKDIR /Job-Recomendation

# Copy the requirements file to the working directory
COPY requirements.txt requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Install numpy first to avoid version issues
RUN pip install --no-cache-dir numpy==1.24.2

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader words
RUN python -m nltk.downloader punkt

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "HOME.py"]
