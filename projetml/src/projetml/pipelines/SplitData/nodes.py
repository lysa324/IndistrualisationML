import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

"""
Cette fonction a pour but de séparer le dataframe initale en 4 dataframe, les données d'entrainement et de test, les labels d'entrainement et de test
"""


def splitData(df: pd.DataFrame):
    # Séparation des données en 80% 20%
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Récupération des données test et de labels test a partir des 20 % de test_df
    test_data = test_df[
        [
            "before_exam_125_Hz",
            "before_exam_250_Hz",
            "before_exam_500_Hz",
            "before_exam_1000_Hz",
            "before_exam_2000_Hz",
            "before_exam_4000_Hz",
            "before_exam_8000_Hz",
        ]
    ]
    test_labels = test_df[
        [
            "after_exam_125_Hz",
            "after_exam_250_Hz",
            "after_exam_500_Hz",
            "after_exam_1000_Hz",
            "after_exam_2000_Hz",
            "after_exam_4000_Hz",
            "after_exam_8000_Hz",
        ]
    ]
    # Récupération des données train et de labels train à partir des 80 % de train_df
    train_data = train_df[
        [
            "before_exam_125_Hz",
            "before_exam_250_Hz",
            "before_exam_500_Hz",
            "before_exam_1000_Hz",
            "before_exam_2000_Hz",
            "before_exam_4000_Hz",
            "before_exam_8000_Hz",
        ]
    ]
    train_labels = train_df[
        [
            "after_exam_125_Hz",
            "after_exam_250_Hz",
            "after_exam_500_Hz",
            "after_exam_1000_Hz",
            "after_exam_2000_Hz",
            "after_exam_4000_Hz",
            "after_exam_8000_Hz",
        ]
    ]

    return train_data, train_labels, test_data, test_labels
