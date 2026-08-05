from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import sklearn
import joblib
import boto3
import pathlib
from io import  StringIO
import argparse
import joblib
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir,'model.jonlib'))
    return clf
if __name__== "__main__":
    print("[info] extracting aurgments")

