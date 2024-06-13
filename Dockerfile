# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port on which the app runs
EXPOSE 8080

# Command to run the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
