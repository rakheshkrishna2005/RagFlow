FROM python:3.9-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies from requirements.txt in the current directory
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the API runs on
EXPOSE 8000

# Command to run the uvicorn server
# Render will set the PORT environment variable
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
