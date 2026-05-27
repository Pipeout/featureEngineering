import os
import subprocess

PATH_CLEANING = "scripts/cleaning.py"
PATH_SELECTION = "scripts/selection.py"

if __name__ == "__main__":
    p1 = subprocess.Popen(["python", PATH_CLEANING])
    p1.wait()
    p2 = subprocess.Popen(["python", PATH_SELECTION])
