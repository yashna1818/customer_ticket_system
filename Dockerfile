# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (specifically ffmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a directory for persistent data
RUN mkdir -p /data

# Set environment variable for the persistent SQLite path
ENV TICKETS_DB_PATH=/data/tickets.db

# Expose the Streamlit port
EXPOSE 8501

# Run the streamlit application on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
