import numpy as np
import pandas as pd

from utils import DataLoader
from xai_helpers import classification_metrics, mlp_preprocess, plain_label, select_patients

DATA = "dataset/healthcare-dataset-stroke-data.csv"


def test_mlp_preprocess_matches_starter_columns_and_split():
    dl = DataLoader()
    dl.load_dataset(DATA)
    dl.preprocess_data()
    _, X_test_ref, _, _ = dl.get_data_split()

    X_train, X_test, y_train, y_test = mlp_preprocess(pd.read_csv(DATA))

    assert list(X_test.columns) == list(X_test_ref.columns)
    assert X_test.index.equals(X_test_ref.index)
    assert set(y_train.unique()) <= {0, 1}
    assert len(X_train) + len(X_test) == 5110


def test_mlp_preprocess_imputes_bmi_with_train_median():
    raw = pd.read_csv(DATA)
    X_train, X_test, _, _ = mlp_preprocess(raw)

    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    raw_train_bmi = raw.loc[X_train.index, "bmi"]
    filled = X_train.loc[raw_train_bmi.isna(), "bmi"]
    assert len(filled) > 0
    assert (filled == raw_train_bmi.median()).all()
    assert (X_train["bmi"] > 0).all()


def test_select_patients_picks_top_true_positive_and_lowest_true_negative():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.2, 0.9, 0.6, 0.05, 0.3])
    assert select_patients(y, p) == {"A": 1, "B": 3, "a_is_tp": True}


def test_select_patients_falls_back_when_no_true_positive():
    y = np.array([0, 1, 1])
    p = np.array([0.1, 0.3, 0.4])
    out = select_patients(y, p)
    assert out["A"] == 2
    assert out["a_is_tp"] is False


def test_plain_label():
    assert plain_label("age") == "Your age"
    assert plain_label("avg_glucose_level") == "Blood sugar level"
    assert plain_label("smoking_status_smokes") == "Smoking history"
    assert plain_label("ever_married_Yes") == "Marital status"
    assert plain_label("some_new_col") == "some new col"


def test_classification_metrics():
    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert m["accuracy"] == 0.75
    assert m["recall_stroke"] == 0.5
    assert set(m) == {"accuracy", "f1_macro", "recall_stroke"}
