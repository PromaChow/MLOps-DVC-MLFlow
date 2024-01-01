# Use a base image with Python
FROM python:3.8-slim-buster

# Copy requirements and install dependencies
COPY flask-app/requirements.txt .
RUN pip install -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy the entire content from the source image
COPY --from=abwqr/test-model:v1 . .
COPY flask-app/. .
# Expose the necessary port
EXPOSE 9000

# Specify the command to run when the container starts
CMD ["python3", "app.py"]



# FROM abwqr/test_model:v1 as model

# WORKDIR /app


# FROM python:3.8-slim-buster


# COPY requirements.txt .
# RUN pip install -r requirements.txt

# COPY . .
# COPY --from=model /kmeans_model .

# EXPOSE 9090

# CMD ["python3", "app.py"]
