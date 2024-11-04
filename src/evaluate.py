import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

from urllib.parse import urlparse

import mlflow

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Susanta2102/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="Susanta2102"
os.environ["MLFLOW_TRACKING_PASSWORD"]="c98538c6100c84b5f1b711557cfc85de232b63b4"

#load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/Susanta2102/machinelearningpipeline.mlflow")
    
    ## load the model from the disk
    model = pickle.load(open(model_path,"rb"))
    
    ## predict the model
    predictions = model.predict(X)
    accuracy=accuracy_score(y,predictions)
    ##log metrics to mlflow
    
    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy: {accuracy}")
    
if __name__ == "__main__":
    evaluate(params["data"],params["model"])