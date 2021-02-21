from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import os
import yaml


def load_data():
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    return df


if __name__ == "__main__":

    with open("params.yaml", "r") as f:
        params = yaml.load(f)

    WORKDIR = params["WORK_DIR"]
    IRIS_PATH = params["data_save_paths"]["iris_data_path"]
    data = load_data()
    data.reset_index().to_feather(os.path.join(WORKDIR, IRIS_PATH))
