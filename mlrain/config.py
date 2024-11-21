import os
from datetime import datetime
from pathlib import Path


BASE_PATH=os.getcwd()

DATA_PATH=Path(BASE_PATH, "data", "weatherAUS.csv")
LOG_PATH=Path(BASE_PATH, "log")

LOG_FILE_NAME=f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_ml_rain_run.log"


