import subprocess

from fastapi import FastAPI

PATH_CLEANING = "scripts/cleaning.py"
PATH_SELECTION = "scripts/selection.py"

app = FastAPI()


@app.post("/feature_engineering")
def run_feature_engineering():
    p1 = subprocess.Popen(["python", PATH_CLEANING])
    p1.wait()
    p2 = subprocess.Popen(["python", PATH_SELECTION])
