import pandas as pd
import numpy as np
from .nodes import datapreprocess 
import pytest

def test_datapreprocess():
    # Créer un DataFrame de test
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [6, 7, 8, np.inf, 10],
        'C': [11, 12, 13, 14, 15]
    })


    processed_df = datapreprocess(df)

    assert not processed_df.empty

    assert pd.notna(processed_df).all(axis=1).all()

    assert not np.isinf(processed_df).any().any()

    assert pd.notna(processed_df).all().all()

    assert not processed_df.duplicated().any()

    assert (processed_df >= 0).all().all() and (processed_df <= 1).all().all()





