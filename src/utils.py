import os, sys, pickle
from src.exception import CustomException
import numpy as np
import pandas as pd

def saveObject(filePath, obj):
    try:
        dirPath = os.path.dirname(filePath)

        os.makedirs(dirPath, exist_ok=True)

        with open(filePath, "wb") as fileObj:
            pickle.dump(obj, fileObj)

    except Exception as e:
        raise CustomException(e, sys)