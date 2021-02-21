import pandas as pd
import yaml
import os
from typing import Tuple
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splitting data

    Args:
        df (pd.DataFrame): iris data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train and test
    """
    data = df.copy()
    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    return train, test


if __name__ == "__main__":

    with open("params.yaml", "r") as f:
        params = yaml.load(f)
    WORKDIR = params["WORK_DIR"]
    DATA_PATH = params["data_save_paths"]["iris_data_path"]
    TRAIN_PATH = params["data_save_paths"]["train_data_path"]
    TEST_PATH = params["data_save_paths"]["test_data_path"]

    data = pd.read_feather(os.path.join(WORKDIR, DATA_PATH))
    train, test = split_data(data)

    train.reset_index().to_feather(os.path.join(WORKDIR, TRAIN_PATH))
    test.reset_index().to_feather(os.path.join(WORKDIR, TEST_PATH))
