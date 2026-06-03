# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies required for OpenCV, MediaPipe, and InsightFace
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    libxext6 \
    libsm6 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the logs directory exists
RUN mkdir -p api/_logs

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
# Render will set the PORT environment variable at runtime
EXPOSE 8080

# Run the application
CMD uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8080}
