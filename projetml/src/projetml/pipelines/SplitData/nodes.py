import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split

def splitData(df: pd.DataFrame):


    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_data = test_df[["before_exam_125_Hz", "before_exam_250_Hz", "before_exam_500_Hz", "before_exam_1000_Hz", "before_exam_2000_Hz", "before_exam_4000_Hz", "before_exam_8000_Hz"]]
    test_labels = test_df[["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz", "after_exam_8000_Hz"]]

    train_data = train_df[["before_exam_125_Hz", "before_exam_250_Hz", "before_exam_500_Hz", "before_exam_1000_Hz", "before_exam_2000_Hz", "before_exam_4000_Hz", "before_exam_8000_Hz"]]
    train_labels = train_df[["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz", "after_exam_8000_Hz"]]


    return train_data, train_labels, test_data, test_labels

