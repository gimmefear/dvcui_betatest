import pandas as pd
import numpy as np
import os
import yaml
import json
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def evaluate(
    train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict
) -> Tuple[float, object]:
    model = LogisticRegression(**params)
    sc = StandardScaler()
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    fitted_sc = sc.fit(X_train)

    X_train_sc = fitted_sc.transform(X_train)
    X_test_sc = fitted_sc.transform(X_test)

    cv_score = cross_val_score(model, X_train_sc, y_train, cv=3, n_jobs=-1, verbose=1)

    train_model = model.fit(X_train_sc, y_train)
    predicts = train_model.predict_proba(X_test_sc)
    roc_auc = roc_auc_score(y_test, predicts, multi_class="ovr")
    return np.mean(cv_score), roc_auc


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.load(f)

    WORKDIR = params["WORK_DIR"]
    TRAIN_PATH = params["data_save_paths"]["train_data_path"]
    TEST_PATH = params["data_save_paths"]["test_data_path"]
    CV_SCORE_PATH = params["metrics_save_paths"]["cv_score_path"]
    TEST_SCORE_PATH = params["metrics_save_paths"]["test_score_path"]

    train = pd.read_feather(os.path.join(WORKDIR, TRAIN_PATH))
    test = pd.read_feather(os.path.join(WORKDIR, TEST_PATH))
    model_params = params["model_params"]

    cv_score, test_score = evaluate(train, test, model_params)

    with open(os.path.join(WORKDIR, CV_SCORE_PATH), "w") as f:
        json.dump(cv_score, f)
        f.close()
    with open(os.path.join(WORKDIR, TEST_SCORE_PATH), "w") as f:
        json.dump(test_score, f)
        f.close()
