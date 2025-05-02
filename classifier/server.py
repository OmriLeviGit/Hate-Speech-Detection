import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from classifier.BaseTextClassifier import BaseTextClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

classifier = BaseTextClassifier.load_best_model("saved_models")

@app.post("/classifier/predict")
async def classify(data: dict):
    text = data.get("text")
    prediction = classifier.predict(text, return_decoded=True)
    return {"prediction": prediction}


if __name__ == '__main__':
    host_address = "0.0.0.0" if os.path.exists('/.dockerenv') else "localhost"
    uvicorn.run("server:app", host=host_address, port=12345, reload=True)