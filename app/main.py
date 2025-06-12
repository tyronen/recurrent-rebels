from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import httpx
import torch
from .model import BigModel # possibly change later

app = FastAPI()
HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"

#Initialize the model -- change as desired
MODEL_PATH = "app/model.pth"

model = BigModel(vector_size=200, scale=3)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

#Define the request and response pydantic models


class HNPostData(BaseModel):
    by: str
    title: str
    url: str
    time: int
    score: int | None = None 
    #add more if helpful

class PredictionResponse(BaseModel):
    prediction: float


def preprocess_input(data: dict) -> list[float]:
    #TODO: Implement the preprocessing logic 
    #return format must match model requirements
    return torch.randn(200)

async def get_hn_item(item_id: int) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HN_API_BASE_URL}/item/{item_id}.json")
            response.raise_for_status() # Raises an exception for 4XX or 5XX status codes
            item_data = response.json()
            if not item_data:
                raise HTTPException(status_code=404, detail=f"Hacker News item {item_id} not found or has no content.")

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error requesting Hacker News API: {e}")
    return item_data

#Define the prediction endpoint
@app.post("/predict/direct", response_model=PredictionResponse)
async def predict_direct(request: HNPostData):
    #Preprocess the input data (match to model requirements)
    input_data = preprocess_input(request.model_dump()) 

    #Make a prediction
    with torch.no_grad():
        prediction = model(input_data)

    #Postprocess the prediction (take exp if needed)
    prediction = (10 **prediction)

    return PredictionResponse(prediction=prediction.item())

@app.get("/predict_hn/{item_id}", response_model=PredictionResponse)
async def predict_from_id(item_id: int):
    """
    Fetches a Hacker News item, preprocesses it, and returns a model prediction.
    """
    # 1. Fetch data from Hacker News API asynchronously
    item_data = await get_hn_item(item_id)

    # 2. Preprocess the fetched data using your custom function
    input_tensor = preprocess_input(item_data)

    # 3. Make a prediction
    with torch.no_grad():
        output = model(input_tensor)

    # 4. Post-process the prediction
    prediction = 10 ** output.item()

    return PredictionResponse(prediction=prediction)
