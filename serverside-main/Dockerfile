FROM python:3.9.16-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file (if you have one)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your server runs on (adjust as needed)
EXPOSE 8000

# Command to run your server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]