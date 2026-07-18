from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
    recall_score
)

def evaluate_model(model,X_test,y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    return {
        "accuracy": accuracy_score(y_test,y_pred),
        "precison": precision_score(y_test,y_pred),
        "f1_score":f1_score(y_test,y_pred),
        "recall" : recall_score(y_test,y_pred),
        "roc_auc" : roc_auc_score(y_test,y_prob)
    }



