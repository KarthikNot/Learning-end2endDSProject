import os, sys, pickle
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
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
    
def evaluateModels(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)