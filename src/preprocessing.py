import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def create_features(df):

    df = df.copy()

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


def encode_features(df):

    df = df.copy()

    df["person_gender"] = df["person_gender"].map({
        "male": 1,
        "female": 0
    })

    df["previous_loan_defaults_on_file"] = (
        df["previous_loan_defaults_on_file"]
        .map({
            "Yes": 1,
            "No": 0
        })
    )

    df = pd.get_dummies(
        df,
        columns=[
            "person_education",
            "person_home_ownership",
            "loan_intent"
        ],
        dtype=int
    )

    return df


def preprocess_input(
    df,
    scaler,
    training_columns
):

    df = create_features(df)

    df = encode_features(df)

    df = df.reindex(
        columns=training_columns,
        fill_value=0
    )

    num_cols = [
        "person_age",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
        "credit_history_ratio",
        "income_per_age"
    ]

    df[num_cols] = scaler.transform(
        df[num_cols]
    )

    return df
