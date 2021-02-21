import os
from pathlib import Path
import yaml

if __name__ == "__main__":
    WORKING_DIR = Path().absolute()

    with open(os.path.join(WORKING_DIR, "params.yaml"), "r") as f:
        params = yaml.load(f) or {}

    params["WORK_DIR"] = str(WORKING_DIR)
    params["PIPELINES_DIR"] = str(WORKING_DIR) + "/src/pipelines/"

    with open(os.path.join(WORKING_DIR, "params.yaml"), "w") as f:
        yaml.dump(params, f)
