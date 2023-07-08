"""
This is a boilerplate pipeline 'GetPredictions'
generated using Kedro 0.18.11
"""

import numpy as np
import mlflow
import mlflow.sklearn
import json


"""
Cette fonction à pour but d'effectuer des prédiction sur des données entrantes a partir d'un user, en utilisant le meilleur modéle mlflow
"""


def predict(X):
    # Récupération de l'id du modéle choisit( meilleur modéle entraîné)
    model = mlflow.sklearn.load_model("runs:/92fabd2985b34059bfeb636da09bd9df/model")
    # Conversion des données entrante en un tableau numpy
    X = np.array(X)
    predictions = model.predict(X)
    # conversion des prédictions au format json pour les retrouner à l'utilisateur
    predictions_json = json.dumps(predictions.tolist())
    print("Les prédictions avec le meilleur modéle  sont :", predictions_json)
    return predictions_json
