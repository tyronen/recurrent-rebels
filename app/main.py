from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import httpx
import torch

from utils.model_handler import get_predictor

app = FastAPI()
HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"

#Initialize the model -- change as desired
MODEL_NAME = "full_model"
predictor = get_predictor(MODEL_NAME)

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
    #Preprocess the input data (get dict)
    input_data = request.model_dump() 
    #Make a prediction
    prediction = predictor.predict(input_data)
    return PredictionResponse(prediction=prediction)

# TODO: Implement HN endpoint once we have the preprocessing working
# @app.get("/predict_hn/{item_id}", response_model=PredictionResponse)
# async def predict_from_id(item_id: int):
#     """
#     Fetches a Hacker News item, preprocesses it, and returns a model prediction.
#     """
#     # 1. Fetch data from Hacker News API asynchronously
#     item_data = await get_hn_item(item_id)
#     
#     # 2. Make a prediction using our predictor
#     prediction = predictor.predict(item_data)
#     
#     return PredictionResponse(prediction=prediction)
