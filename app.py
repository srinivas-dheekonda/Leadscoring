import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize FastAPI application
app = FastAPI(
    title="Lead Scoring Inference API",
    description="API for predicting lead conversion probability using a trained machine learning model.",
    version="1.0"
)

# 2. Load the trained model from the pickle file
MODEL_PATH = "res1_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully into memory!")
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

# 3. Define the incoming data schema using Pydantic
# ⚠️ IMPORTANT: Update these field names and types to match the exact features your model was trained on!
class LeadInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# 4. Define a simple health-check route
@app.get("/")
def home():
    return {"status": "Online", "message": "Lead Scoring API is up and running."}

# 5. Define the prediction endpoint
@app.post("/predict")
def predict_lead(data: LeadInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded on the server.")
    
    try:
        # Convert incoming JSON payload into a Pandas DataFrame (or array) matching your training format
        input_df = pd.DataFrame([{
            "feature1": data.feature1,
            "feature2": data.feature2,
            "feature3": data.feature3
        }])
        
        # Run model prediction
        prediction = model.predict(input_df)
        
        # Optional: If your model supports probability scoring, extract it:
        # prob = float(model.predict_proba(input_df)[0][1])

        return {
            "prediction": int(prediction[0]),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    ## http://127.0.0.1:8000/docs