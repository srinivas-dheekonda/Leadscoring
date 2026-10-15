import io
import os
from catboost import CatBoostClassifier
import pandas as pd
import joblib


def model_fn(model_dir):
    """Load the trained CatBoost model from the model directory using joblib."""
    model_path = os.path.join(model_dir, "best_xg_model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
  """Parses incoming batch CSV data from S3."""
  if request_content_type == "text/csv":
    # Read the incoming batch rows into a DataFrame
    df = pd.read_csv(io.StringIO(request_body))
    return df
  else:
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    
    # 1. Get the exact feature names the model was trained on
    model_features = model.get_booster().feature_names
    
    # 2. Add any missing columns that the model expects (fill them with 0)
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
            
    # 3. Drop any extra columns that the model wasn't trained on
    # 4. Reorder the columns to match the training order precisely
    input_data = input_data[model_features]
    
    # Run predictions
    probabilities = model.predict_proba(input_data)[:, 1]
    return probabilities