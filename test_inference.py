import My_project.CODE.inference as inference
import pandas as pd

# 1. Test model loading locally (from your current folder '.')
model_dir = "."
model = inference.model_fn(model_dir)
print("✅ Model loaded successfully!")

# 2. Test input parsing and prediction with your validation file
# Change 'test_v1.csv' to whatever your actual test file is named
with open("test_v1.csv", "r") as f:
  csv_data = f.read()

# Simulate SageMaker's input_fn
processed_data = inference.input_fn(csv_data, "text/csv")
print(f"✅ Data parsed successfully! Shape: {processed_data.shape}")

# Simulate SageMaker's predict_fn
predictions = inference.predict_fn(processed_data, model)
print("✅ Predictions generated successfully!")
print("Sample Predictions:", predictions[:5])