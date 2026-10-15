import os 
import pandas as pd
import joblib
import pyodbc
from azure.storage.blob import BlobServiceClient




# # Load model from azure blob storage
def load_model():
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container='ml-model-container-leadscoring', blob='models/lead_scoring/best_xg_model.joblib')
    model_path = "/tmp/model.joblib"
    with open(model_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    
    return joblib.load(model_path)

def fetch_unscored_leads():
    conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
    query = "SELECT lead_id, feature_1, feature_2, feature_3 FROM Leads WHERE lead_score IS NULL"
    
    with pyodbc.connect(conn_str) as conn:
        df = pd.read_sql(query, conn)
    return df
# 3. Write Scores Back to Azure SQL
def update_lead_scores(df_scored):
    conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        for _, row in df_scored.iterrows():
            cursor.execute(
                "UPDATE Leads SET lead_score = ?, scored_at = GETDATE() WHERE lead_id = ?",
                float(row['predicted_score']),
                row['lead_id']
            )
        conn.commit()

if __name__ == "__main__":
    model = load_model()
    leads_df = fetch_unscored_leads()
    
    if not leads_df.empty:
        features = leads_df[['feature_1', 'feature_2', 'feature_3']]
        # Assuming classification probability or regression score
        leads_df['predicted_score'] = model.predict_proba(features)[:, 1] 
        update_lead_scores(leads_df)
        print(f"Successfully scored {len(leads_df)} leads.")
    else:
        print("No new leads to score.")