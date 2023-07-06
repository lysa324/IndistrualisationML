import pandas as pd
import numpy as np
from .node import datapreprocess 

def test_datapreprocess():

    data = {'col1': [np.nan, 20, 30, 40,np.nan],
            'col2': [5, np.nan, 25, 35, 45],
            'col3': [1, 2, np.nan, 4, 5]}
    df = pd.DataFrame(data)


    processed_df = datapreprocess(df)


    assert 'col1' not in processed_df.columns
    assert 'col2' not in processed_df.columns
    assert 'col3' in processed_df.columns


    assert len(processed_df) == 3


    assert not np.any(np.isnan(processed_df.values))


    assert len(processed_df.drop_duplicates()) == len(processed_df)


    assert not processed_df.empty

    print("Tous les tests ont réussi.")

# Appeler la fonction de test
test_datapreprocess()

