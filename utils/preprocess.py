import pandas as pd

def preprocess_data(df):
    df = df.dropna()
    X = df.drop(columns=['treatment', 'outcome'])
    treatment = df['treatment']
    y = df['outcome']
    return X, treatment, y
