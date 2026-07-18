from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    return model




