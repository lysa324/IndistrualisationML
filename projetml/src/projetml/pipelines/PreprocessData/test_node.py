import pandas as pd
import numpy as np
from .nodes import datapreprocess 
import pytest



def test_datapreprocess():

    data = {'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, np.nan]}
    df = pd.DataFrame(data)

    # Appeler la fonction datapreprocess sur le DataFrame de test
    preprocessed_df = datapreprocess(df)

    # Vérifier les transformations appliquées au DataFrame
    assert preprocessed_df.isnull().sum().sum() == 0  # Aucune valeur nulle
    assert preprocessed_df.duplicated().sum() == 0  # Aucun doublon
    assert preprocessed_df.empty is False  # DataFrame non vide

    # Vérifier la normalisation des données
    min_val = df.min()
    max_val = df.max()
    expected_df = (df - min_val) / (max_val - min_val)
    assert preprocessed_df.equals(expected_df)  

    print("La fonction de prétraitement des données a été testée avec succès.")

# Appeler la fonction de test
test_datapreprocess()


