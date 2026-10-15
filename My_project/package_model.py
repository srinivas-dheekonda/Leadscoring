import tarfile
import boto3


#### compress the model into model.tar.gz
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("best_xg_model.joblib", arcname="best_xg_model.joblib")
    tar.add("requirements.txt", arcname="requirements.txt")

print("✅ Model and requirements compressed into model.tar.gz successfully!")