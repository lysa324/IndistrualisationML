"""
This is a boilerplate pipeline 'TestModel'
generated using Kedro 0.18.10
"""

import numpy as np
import mlflow
import mlflow.sklearn


def test_model(X):
    model = mlflow.sklearn.load_model("runs:/b0137741f94042c4bccd74f3c68714ba/model")
    X = np.array(X)
    predictions = model.predict(X)
    print("Les prédictions sont :", predictions)
    return predictions





