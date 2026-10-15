import sagemaker
from sagemaker import Session
from sagemaker.sklearn import SKLearnModel


## initialize the sagemaker session
sagemaker_session = Session()
## give the role 
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()

## define the model artifacts and other S3 paths
perfix = "Lead-scoring"
input_data_uri = f"s3://{bucket}/{perfix}/input"
output_data_uri = f"s3://{bucket}/{perfix}/output"
model_artifact_uri = f"s3://{bucket}/{perfix}/model/model.tar.gz"


## create the model object 
lead_scoring_model = SKlearnModel(model_data=model_artifact_uri, role =role, instance_type="ml.m5.large", entry_point="app.py", source_dir=".", framework_version="0.23-1") 
