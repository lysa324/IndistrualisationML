"""
This is a boilerplate pipeline 'TestModel'
generated using Kedro 0.18.10
"""

import numpy as np
from kedro.extras.datasets.pandas import CSVDataSet

def test_model(model, X):
    model = model
    X = np.array(X)
    predictions = model.predict(X)
    print("Les prédictions sont :", predictions)
    return predictions














