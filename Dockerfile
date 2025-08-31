FROM python:3.9-slim

WORKDIR /app

# Copy only the API directory and its requirements
COPY API/ /app/API/

# Install dependencies
RUN pip install --no-cache-dir -r API/requirements.txt

# Expose the port the API runs on
EXPOSE 8000

# Command to run the uvicorn server
# Render will set the PORT environment variable
CMD ["uvicorn", "API.api_server:app", "--host", "0.0.0.0", "--port", "8000"]