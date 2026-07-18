import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def create_features(df):

    df["credit_history_ratio"] = (
        df["cb_person_cred_hist_length"]
        /
        df["person_age"]
    )

    df["income_per_age"] = (
        df["person_income"]
        /
        df["person_age"]
    )
    
    return df
