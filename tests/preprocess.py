import pandas as pd

def preprocess_data(df):

    """
    Preprocess the input DataFrame for treatment effect estimation.
    Args:
        df (pd.DataFrame): Input DataFrame containing features, treatment, and outcome.
    Returns:
        X (pd.DataFrame): Features DataFrame.
        treatment (pd.Series): Treatment assignment.
        y (pd.Series): Outcome variable.
    """

    # Ensure the DataFrame has the required columns
    df = df.dropna()
    X = df.drop(columns=['treatment', 'outcome'])
    treatment = df['treatment']
    y = df['outcome']
    return X, treatment, y

