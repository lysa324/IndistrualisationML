"""
This is a boilerplate pipeline 'TestModel'
generated using Kedro 0.18.10
"""

import numpy as np
import mlflow
import mlflow.sklearn


def test_model(model,X):
    #model = mlflow.sklearn.load_model("runs:/92fabd2985b34059bfeb636da09bd9df/model")
    model = model
    X = np.array(X)
    predictions = model.predict(X)
    print("Les prédictions sont :", predictions)
    return predictions





