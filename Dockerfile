# Use the official Python slim image.
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file first to leverage Docker cache.
COPY requirements.txt ./

# Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port your application listens on; adjust if needed.
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]