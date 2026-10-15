FROM python:3.12.13-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app and your pickle file 
COPY app.py .
COPY res1_model.pkl .

# Expose the correct port matching Uvicorn (8000)
EXPOSE 8000

# Command to run the application (fixed to app:app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]$