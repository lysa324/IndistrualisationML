"""
This is a boilerplate pipeline 'PreprocessData'
generated using Kedro 0.18.10
"""
import pandas as pd
import numpy as np
from kedro.config import ConfigLoader, TemplatedConfigLoader
import yaml

"""
Cette fonction à pour but de prétraiter un dataframe afin de le rendre exploitable dans un modéle ML
"""


def datapreprocess(df: pd.DataFrame):
    # Supprimer les lignes entièrement nulles
    df = df.dropna(axis=0, how="all")

    # Supprimer les données non exploitables
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    df = df.drop_duplicates()
    if df.empty:
        raise ValueError("Attention le DataFrame est vide.")

    # Normaliser les données en utilisant le min et le max
    min_val = df.min()
    max_val = df.max()
    df = (df - min_val) / (max_val - min_val)

    return df
