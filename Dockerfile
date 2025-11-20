# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package.json and install frontend dependencies
COPY package*.json ./
RUN npm install

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Build the frontend
RUN npm run build

# Expose port
EXPOSE 8000

# Start the application
CMD ["python3", "backend/server.py"]
