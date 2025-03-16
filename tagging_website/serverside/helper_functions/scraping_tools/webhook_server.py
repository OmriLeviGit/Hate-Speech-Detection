from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import csv
import json
import os
import time
from datetime import datetime
import uvicorn
import base64
from dotenv import load_dotenv
from pathlib import Path
import requests
import os
from fastapi import FastAPI, Request

load_dotenv(os.path.join(Path(__file__).parent.absolute(), ".." "..", '.env.local'))
USERNAME = os.environ.get('FETCH_USERNAME')
PASSWORD = os.environ.get('FETCH_PASSWORD')

app = FastAPI()

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_filepath(data_type: str, original_filename):
    save_dir = os.path.join(script_dir, "..", "..", "data", "ready_to_load", data_type)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, original_filename)

def trigger_brightdata_job(data_type: str, file_path: str):
    """Trigger a scraping job on Brightdata"""
    username = os.environ.get('FETCH_USERNAME')
    password = os.environ.get('FETCH_PASSWORD')
    api_token = os.environ.get('API_TOKEN')

    # Extract just the filename from the path
    file_name = os.path.basename(file_path)
    batch_name = os.path.splitext(file_name)[0]  # Remove extension
    
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    
    # Set up the webhook URL where Brightdata will send the results
    webhook_url = "http://18.198.62.179:5000/brightdata-webhook"
    
    # API endpoint
    url = "https://api.brightdata.com/datasets/v3/trigger"
    
    # Parameters for the API call
    params = {
        "dataset_id": "gd_lwxkxvnf1cynvib9co",
        "endpoint": f"{webhook_url}?data_type={data_type}",  # Pass data_type to webhook
        "batch_name": batch_name,
        "auth_header": f"{username}:{password}",
        "format": "csv",
        "uncompressed_webhook": "true",
        "include_errors": "true",
        "type": "discover_new",
        "discover_by": "profile_url"
    }
    
    # Open and send the file
    with open(file_path, 'rb') as file:
        files = {'data': (file_name, file)}
        response = requests.post(url, headers=headers, params=params, files=files)
    
    return response.json()

# Update webhook to handle the response
@app.post("/brightdata-webhook")
async def brightdata_webhook(request: Request):
    """Webhook to receive data from Brightdata when the job is complete"""
    
    # Extract data_type from query parameters
    data_type = request.query_params.get("data_type", "unknown")
    
    # Get content type and filename
    content_type = request.headers.get("Content-Type", "")
    
    # For filename, try to get it from Brightdata response
    filename = request.headers.get("X-Filename")
    if not filename:
        # Try to get batch name from query parameters or headers
        batch_name = request.query_params.get("batch_name") or request.headers.get("X-Batch-Name")
        if batch_name:
            filename = f"{batch_name}.csv"
        else:
            # Use timestamp as filename if nothing else is available
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bd_{timestamp}_0.csv"
    
    if "text/csv" in content_type or "application/octet-stream" in content_type:
        # Get the actual content
        data = await request.body()
        
        # Generate the path and save the file
        filepath = generate_filepath(data_type, filename)
        
        with open(filepath, "wb") as f:
            f.write(data)
        
        print(f"Data received and saved to: {filepath}")
        return {"status": "success", "saved_to": filepath}
    
    return {"status": "error", "message": "Unsupported content type"}


async def start_scrape(data_type: str, file_name: str):
    """Endpoint to start a scraping job"""
    
    # Construct the complete file path
    source_path = os.path.join(script_dir, "..", "..", "data", "users_of_interest", data_type, file_name)
    
    # Make sure the file exists
    if not os.path.exists(source_path):
        return {"status": "error", "message": f"File not found: {source_path}"}
    
    # Trigger the job
    try:
        result = trigger_brightdata_job(data_type, source_path)
        return {"status": "success", "brightdata_response": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # If you uncomment this, it will run a job on startup

    # import asyncio
    # data_type = "your_data_type"
    # file_name = "your_file.csv"
    # asyncio.run(start_scrape(data_type, file_name))
    
    uvicorn.run(app, host="0.0.0.0", port=5000)