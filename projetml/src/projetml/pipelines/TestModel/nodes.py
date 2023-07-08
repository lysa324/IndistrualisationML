"""
This is a boilerplate pipeline 'TestModel'
generated using Kedro 0.18.10
"""

import numpy as np
import mlflow
import mlflow.sklearn

"""
Cette fonction a pour but d'utiliser le modèle de prédiction enregistré au format pickle pour effectuer des prédictions sur un jeu de données en entrée
"""


def test_model(model, X):
    model = model
    X = np.array(X)
    predictions = model.predict(X)
    print("Les prédictions sont :", predictions)
    return predictions
