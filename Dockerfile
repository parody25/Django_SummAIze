# Use an official Python runtime as a parent image
FROM python:3.11

COPY backend/Personalized_Wealth_Management_Advisor /app

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt && playwright install --with-deps

# Copy the rest of your Django app code into the container
# COPY backend/Personalized_Wealth_Management_Advisor /app/

# Expose the port that the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "manage.py", "runserver"]
