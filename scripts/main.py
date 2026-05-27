import subprocess
import sys

PATH_CLEANING = "scripts/cleaning.py"
PATH_SELECTION = "scripts/selection.py"

if __name__ == "__main__":
    p1 = subprocess.run([sys.executable, PATH_CLEANING], check=True)
    p2 = subprocess.run([sys.executable, PATH_SELECTION], check=True)
