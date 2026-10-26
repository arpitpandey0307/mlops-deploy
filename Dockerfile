#Start with a light-weight base python image
#This gives our container a python runtime
FROM python:3.14-slim

#Set the working directory inside the container to /app
WORKDIR /app

#Copy the requirements.txt file into the container at /app
COPY requirements.txt .

#Install the dependencies specified in requirements.txt and --no-cache-dir flag is used to avoid caching the packages, which reduces the size of the image
RUN pip install --no-cache-dir -r requirements.txt

#copy the application code and saved model into the container at /app
COPY . .

#Document the port used by the application inside the container. This is the port that the FastAPI application will listen on.
EXPOSE 8000

#Start the API sever when the container lauches
# 0.0.0.0 is the IP address that the container will listen on and allows coonection through the network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]