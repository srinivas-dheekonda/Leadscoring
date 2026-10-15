import os
import joblib
import pandas as pd

def init():
    """Loads the model into memory once when the batch worker starts."""
    global model
    
    # Check if Azure ML environment variable exists
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    
    if model_dir:
        # Path when running live in an Azure container
        model_path = os.path.join(model_dir, "best_xg_model.joblib")
    else:
        # Fallback path if testing locally
        model_path = "best_xg_model.joblib"
        
    model = joblib.load(model_path)

def run(mini_batch):
    """Processes incoming mini-batches of data files in parallel."""
    results = []
    for file_path in mini_batch:
        # Read each input data file
        data = pd.read_csv(file_path)
        
        # Make predictions using your trained model
        predictions = model.predict(data)
        
        # Append predictions back to data frame
        df_out = data.copy()
        df_out["prediction"] = predictions
        results.append(df_out)
        
    return pd.concat(results, ignore_index=True)