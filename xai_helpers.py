"""Helpers for stroke_xai.ipynb: MLP preprocessing, patient selection, metrics, plain-language labels."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

PLAIN_LABELS = {
    "age": "Your age",
    "avg_glucose_level": "Blood sugar level",
    "bmi": "Body weight (BMI)",
    "hypertension": "High blood pressure",
    "heart_disease": "Heart disease",
}
PREFIX_LABELS = {
    "gender": "Sex",
    "ever_married": "Marital status",
    "work_type": "Type of work",
    "Residence_type": "Where you live",
    "smoking_status": "Smoking history",
}


def mlp_preprocess(raw, random_state=2021):
    """Same encoding and split as utils.DataLoader, but BMI is imputed with the train median."""
    encoded = pd.get_dummies(raw[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
    data = pd.concat([encoded, raw], axis=1).drop(CATEGORICAL_COLS + ["id"], axis=1)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    median = X_train["bmi"].median()
    X_train = X_train.assign(bmi=X_train["bmi"].fillna(median)).astype(float)
    X_test = X_test.assign(bmi=X_test["bmi"].fillna(median)).astype(float)
    return X_train, X_test, y_train, y_test


def select_patients(y_true, proba, threshold=0.5):
    """Return positional indices of a high-risk example (A) and a low-risk example (B).

    A is the true positive with the highest probability; if there is none, it is the
    actual stroke case with the highest probability. B is the true negative with the
    lowest probability.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    tp = np.flatnonzero((y_true == 1) & (proba >= threshold))
    if tp.size:
        a, a_is_tp = int(tp[np.argmax(proba[tp])]), True
    else:
        pos = np.flatnonzero(y_true == 1)
        if pos.size == 0:
            raise ValueError("y_true contains no stroke cases")
        a, a_is_tp = int(pos[np.argmax(proba[pos])]), False
    tn = np.flatnonzero((y_true == 0) & (proba < threshold))
    if tn.size == 0:
        raise ValueError("no true negatives")
    b = int(tn[np.argmin(proba[tn])])
    return {"A": a, "B": b, "a_is_tp": a_is_tp}


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "recall_stroke": float(recall_score(y_true, y_pred, pos_label=1)),
    }


def plain_label(feature):
    if feature in PLAIN_LABELS:
        return PLAIN_LABELS[feature]
    for prefix, label in PREFIX_LABELS.items():
        if feature.startswith(prefix + "_"):
            return label
    return feature.replace("_", " ")
