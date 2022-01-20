from google.colab import drive
from pathlib import Path
import sys
import os

drive.mount('/content/drive')

py_file_location = "/content/drive/MyDrive/ICIP_2022/code"
sys.path.append(os.path.abspath(py_file_location))

import main

env = 'COLAB'

main.run(env)