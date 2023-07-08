from .nodes import splitData
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest


import pandas as pd
from sklearn.model_selection import train_test_split
import pytest


def test_splitData():
    df = pd.DataFrame(
        {
            "before_exam_125_Hz": [1, 2, 3, 4, 5],
            "before_exam_250_Hz": [6, 7, 8, 9, 10],
            "before_exam_500_Hz": [11, 12, 13, 14, 15],
            "before_exam_1000_Hz": [16, 17, 18, 19, 20],
            "before_exam_2000_Hz": [21, 22, 23, 24, 25],
            "before_exam_4000_Hz": [26, 27, 28, 29, 30],
            "before_exam_8000_Hz": [31, 32, 33, 34, 35],
            "after_exam_125_Hz": [36, 37, 38, 39, 40],
            "after_exam_250_Hz": [41, 42, 43, 44, 45],
            "after_exam_500_Hz": [46, 47, 48, 49, 50],
            "after_exam_1000_Hz": [51, 52, 53, 54, 55],
            "after_exam_2000_Hz": [56, 57, 58, 59, 60],
            "after_exam_4000_Hz": [61, 62, 63, 64, 65],
            "after_exam_8000_Hz": [66, 67, 68, 69, 70],
        }
    )

    train_data, train_labels, test_data, test_labels = splitData(df)

    assert not train_data.empty
    assert not train_labels.empty
    assert not test_data.empty
    assert not test_labels.empty

    assert train_data.shape == (4, 7)
    assert train_labels.shape == (4, 7)
    assert test_data.shape == (1, 7)
    assert test_labels.shape == (1, 7)

    assert train_data.columns.tolist() == [
        "before_exam_125_Hz",
        "before_exam_250_Hz",
        "before_exam_500_Hz",
        "before_exam_1000_Hz",
        "before_exam_2000_Hz",
        "before_exam_4000_Hz",
        "before_exam_8000_Hz",
    ]
    assert train_labels.columns.tolist() == [
        "after_exam_125_Hz",
        "after_exam_250_Hz",
        "after_exam_500_Hz",
        "after_exam_1000_Hz",
        "after_exam_2000_Hz",
        "after_exam_4000_Hz",
        "after_exam_8000_Hz",
    ]
    assert test_data.columns.tolist() == [
        "before_exam_125_Hz",
        "before_exam_250_Hz",
        "before_exam_500_Hz",
        "before_exam_1000_Hz",
        "before_exam_2000_Hz",
        "before_exam_4000_Hz",
        "before_exam_8000_Hz",
    ]
    assert test_labels.columns.tolist() == [
        "after_exam_125_Hz",
        "after_exam_250_Hz",
        "after_exam_500_Hz",
        "after_exam_1000_Hz",
        "after_exam_2000_Hz",
        "after_exam_4000_Hz",
        "after_exam_8000_Hz",
    ]
