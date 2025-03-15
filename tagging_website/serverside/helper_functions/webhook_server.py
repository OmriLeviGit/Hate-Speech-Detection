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


app = FastAPI()

# Filename for storing all users
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
os.makedirs(data_dir, exist_ok=True)

load_dotenv(os.path.join(Path(__file__).parent.absolute(), "..", '.env.local'))
USERNAME = os.environ.get('FETCH_USERNAME')
PASSWORD = os.environ.get('FETCH_PASSWORD')

@app.post("/brightdata-webhook")
async def brightdata_webhook(request: Request):
    try:
        # Check authorization header

        auth_header = request.headers.get("Authorization")
        expected_auth = f"{USERNAME}:{PASSWORD}"
        
        if not auth_header or auth_header != expected_auth:
            print(f"Auth failed. Got: {auth_header}")
            return JSONResponse(content={"status": "error", "message": "Unauthorized"}, status_code=401)
        
        print("Webhook endpoint called with valid authentication!")
        
        # Look for potential filename indicators in the request
        params = dict(request.query_params)
        batch_type = params.get('batch_type', 'batch_type')
        batch_number = params.get('batch_number', 'batch_number')
        snapshot_id = params.get('snapshot_id', 'snapshot_id')
        
        # Construct filename based on pattern
        filename = f"{batch_type}_batch{batch_number}_{snapshot_id}.csv"
        print(f"Using filename: {filename}")
        
        # Determine the save location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ready_dir = os.path.join(os.path.dirname(script_dir), "ready_to_load")
        
        # Make sure the directory exists
        os.makedirs(ready_dir, exist_ok=True)
        
        file_path = os.path.join(ready_dir, filename)
        
        # Process the incoming data
        try:
            # Try to get as JSON first
            data = await request.json()
            print(f"Received data of type: {type(data)}")
            print(f"Data sample: {str(data)[:100]}...")
            
            # Save to CSV
            if isinstance(data, list) and len(data) > 0:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
                print(f"Saved data to {file_path}")
            else:
                print("Data wasn't in expected format")
        except json.JSONDecodeError:
            # If not JSON, try to get raw data
            body = await request.body()
            with open(file_path, 'wb') as f:
                f.write(body)
            print(f"Saved raw data to {file_path}")
        
        return JSONResponse(content={"status": "success", "message": "Data received and saved"}, status_code=200)
    
    except Exception as e:
        error_msg = f"Error in webhook: {str(e)}"
        print(error_msg)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

def get_file_path(file_type, num):
    file_name = f"{file_type}_batch{num}.csv"
    return os.path.join(data_dir, file_name)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)