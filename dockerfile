FROM python:3.9-slim

'''This file sets up a reproducible environment with all 
the necessary dependencies for training and serving the MNIST model.'''

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    software-properties-common

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point
ENTRYPOINT ["python", "app.py"]
