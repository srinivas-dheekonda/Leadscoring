import pickle
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd 

app = FastAPI()

# Load the trained model
with open('res1_model.pkl', 'rb') as file:
    model = pickle.load(file)

# API prediction endpoint
@app.post("/predict")
def predict_lead(data: dict):
    df = pd.DataFrame([data])
    score = model.predict(df)[0]
    return {"lead_score": (score)}

# Web page frontend endpoint
@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lead Scoring Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f7f6; margin: 0; padding: 50px; }
            .container { max-width: 500px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin: auto; }
            h2 { color: #333; text-align: center; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; color: #555; font-weight: bold; }
            input { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
            button { width: 100%; background: #007BFF; color: white; border: none; padding: 10px; border-radius: 4px; font-size: 16px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #result { margin-top: 20px; font-size: 18px; text-align: center; font-weight: bold; color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Lead Scoring Predictor</h2>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="feature1">Feature 1 (e.g., Time on Site):</label>
                    <input type="number" id="feature1" step="any" required>
                </div>
                <div class="form-group">
                    <label for="feature2">Feature 2 (e.g., Page Views):</label>
                    <input type="number" id="feature2" step="any" required>
                </div>
                <button type="submit">Calculate Lead Score</button>
            </form>
            <div id="result"></div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // Collect input values (adjust keys based on your model's expected column names)
                const data = {
                    "Feature_1": parseFloat(document.getElementById('feature1').value),
                    "Feature_2": parseFloat(document.getElementById('feature2').value)
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                if (response.ok) {
                    document.getElementById('result').innerText = "Lead Score: " + result.lead_score.toFixed(4);
                } else {
                    document.getElementById('result').innerText = "Error calculating score.";
                }
            });
        </script>
    </body>
    </html>
    """