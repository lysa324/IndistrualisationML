"""
This is a boilerplate pipeline 'GetPredictions'
generated using Kedro 0.18.11
"""

import numpy as np
import mlflow
import mlflow.sklearn
import json


def predict(X):
    model = mlflow.sklearn.load_model("runs:/92fabd2985b34059bfeb636da09bd9df/model")
    X = np.array(X)
    predictions = model.predict(X)
    predictions_json = json.dumps(predictions.tolist())
    print("Les prédictions avec le meilleur modéle  sont :", predictions_json)
    return predictions_json

