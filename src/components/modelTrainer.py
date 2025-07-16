import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import ( 
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import saveObject, evaluateModels

@dataclass 
class ModelTrainerConfig:
    trainedModelFilePath : str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()

    def initiateModelTraining(self, trainArr, testArr):
        try:
            logging.info('Model training started!')
            logging.info('Splitting train and test input')
            X_train, y_train, X_test, y_test = (
                trainArr[:,:-1],
                trainArr[:,-1],
                testArr[:, :-1],
                testArr[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            modelReport : dict = evaluateModels(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)

            bestModelScore = max(sorted(modelReport.values()))


            bestModelName = list(modelReport.keys())[
                list(modelReport.values()).index(bestModelScore)
            ]
            bestModel = models[bestModelName]
        
            if bestModelScore < 0.6:
                raise CustomException('No Best Model!')
            logging.info('Best model found on train and test sets')

            saveObject(
                filePath=self.modelTrainerConfig.trainedModelFilePath,
                obj = bestModel
            )

            predictedOutput = bestModel.predict(X_test)
            r2Square = r2_score(y_test, predictedOutput)

            return r2Square
        except Exception as e:
            raise CustomException(e, sys)