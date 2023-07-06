from .nodes import splitData 
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest



def test_splitData():
    # Create a sample DataFrame for testing
    columns = ["before_exam_125_Hz", "before_exam_250_Hz", "before_exam_500_Hz", "before_exam_1000_Hz", "before_exam_2000_Hz", "before_exam_4000_Hz", "before_exam_8000_Hz",
               "after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz", "after_exam_8000_Hz"]

    data = [[1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17],
            [2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27],
            [3, 4, 5, 6, 7, 8, 9, 31, 32, 33, 34, 35, 36, 37],
            [4, 5, 6, 7, 8, 9, 10, 41, 42, 43, 44, 45, 46, 47],
            [5, 6, 7, 8, 9, 10, 11, 51, 52, 53, 54, 55, 56, 57]]
    df = pd.DataFrame(data, columns=columns)

    # Call the splitData function
    train_data, train_labels, test_data, test_labels = splitData(df)

    # Verify the shapes of the returned data
    assert train_data.shape == (4, 7)
    assert train_labels.shape == (4,7)
    assert test_data.shape == (1, 7)
    assert test_labels.shape == (1, 7)

    print("tous les tests ont bien été effectué!")

    
    

 


