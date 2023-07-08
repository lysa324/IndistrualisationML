from .nodes import create_model
from .nodes import train_model
import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
from mlflow.models import infer_signature


def test_create_model():
    input_shape = pd.DataFrame(
        {
            "column1": [0.54, 0.46, 0.65, 0.27, 0.57, 0.00, 0.91, 0.49, 0.47, 0.03],
            "column2": [0.65, 0.36, 0.63, 0.52, 0.23, 0.78, 0.52, 0.41, 0.62, 0.36],
            "column3": [0.68, 0.62, 0.86, 0.85, 0.24, 0.29, 0.59, 0.99, 0.47, 0.44],
            "column4": [0.32, 0.89, 0.06, 0.09, 0.46, 0.82, 0.92, 0.25, 0.59, 0.09],
            "column5": [0.71, 0.91, 0.22, 0.18, 0.25, 0.86, 0.17, 0.71, 0.87, 0.53],
            "column6": [0.65, 0.75, 0.23, 0.75, 0.74, 0.84, 0.12, 0.27, 0.39, 0.49],
            "column7": [0.16, 0.79, 0.05, 0.68, 0.21, 0.75, 0.58, 0.04, 0.18, 0.55],
        }
    )

    model = create_model(input_shape)

    assert model is not None


import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error


def test_train_model():
    input_shape = pd.DataFrame(
        {
            "before_exam_125_Hz": [0.54],
            "before_exam_250_Hz": [0.65],
            "before_exam_500_Hz": [0.68],
            "before_exam_1000_Hz": [0.32],
            "before_exam_2000_Hz": [0.71],
            "before_exam_4000_Hz": [0.65],
            "before_exam_8000_Hz": [0.16],
        }
    )
    x_train = pd.DataFrame(
        {
            "before_exam_125_Hz": [0.54],
            "before_exam_250_Hz": [0.65],
            "before_exam_500_Hz": [0.68],
            "before_exam_1000_Hz": [0.32],
            "before_exam_2000_Hz": [0.71],
            "before_exam_4000_Hz": [0.65],
            "before_exam_8000_Hz": [0.16],
        }
    )
    y_train = pd.DataFrame(
        {
            "after_exam_125_Hz": [0.26],
            "after_exam_250_Hz": [0.43],
            "after_exam_500_Hz": [0.20],
            "after_exam_1000_Hz": [0.08],
            "after_exam_2000_Hz": [0.34],
            "after_exam_4000_Hz": [0.56],
            "after_exam_8000_Hz": [0.17],
        }
    )
    x_val = pd.DataFrame(
        {
            "before_exam_125_Hz": [0.27],
            "before_exam_250_Hz": [0.50],
            "before_exam_500_Hz": [0.32],
            "before_exam_1000_Hz": [0.03],
            "before_exam_2000_Hz": [0.52],
            "before_exam_4000_Hz": [0.56],
            "before_exam_8000_Hz": [0.95],
        }
    )
    y_val = pd.DataFrame(
        {
            "after_exam_125_Hz": [0.26],
            "after_exam_250_Hz": [0.43],
            "after_exam_500_Hz": [0.35],
            "after_exam_1000_Hz": [0.05],
            "after_exam_2000_Hz": [0.18],
            "after_exam_4000_Hz": [0.29],
            "after_exam_8000_Hz": [0.20],
        }
    )

    model = train_model(input_shape, x_train, y_train, x_val, y_val)

    assert model is not None
