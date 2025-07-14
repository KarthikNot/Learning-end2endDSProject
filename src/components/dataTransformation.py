import sys, os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import saveObject

@dataclass 
class dataTransformationConfig:
    preprocessorObjFilePath = os.path.join('artifacts', 'preprocessor.pkl')

class dataTransformation:
    def __init__(self):
        self.dataTransformationConfig = dataTransformationConfig()
    
    def getDataTransformerObj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numPipeline = Pipeline([
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            catPipeline = Pipeline([
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('numerical pipeline', numPipeline, numerical_columns),
                ('categorical pipeline', catPipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiateDataTransformation(self, trainPath : str, testPath : str):
        try:
            trainDf = pd.read_csv(trainPath)
            testDf = pd.read_csv(testPath)
            logging.info('Reading train and test data completed!')

            logging.info("Obtaining preprocessing object")
            preprocessorObj = self.getDataTransformerObj()
            targetColumnName = 'math_score'
            numericalColumns = ["writing_score", "reading_score"]

            inputFeatureTrainDf = trainDf.drop(columns=[targetColumnName],axis=1)
            targetFeatureTrainDf = trainDf[targetColumnName]

            inputFeatureTestDf = testDf.drop(columns=[targetColumnName],axis=1)
            targetFeatureTestDf = testDf[targetColumnName]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessorObj.fit_transform(inputFeatureTrainDf)
            input_feature_test_arr = preprocessorObj.transform(inputFeatureTestDf)

            train_arr = np.c_[input_feature_train_arr, np.array(targetFeatureTrainDf)]
            test_arr = np.c_[input_feature_test_arr, np.array(targetFeatureTestDf)]

            logging.info(f"Saved preprocessing object.")

            saveObject(
                filePath=self.dataTransformationConfig.preprocessorObjFilePath,
                obj=preprocessorObj
            )

            return (
                train_arr,
                test_arr,
                self.dataTransformationConfig.preprocessorObjFilePath,
            )

        except Exception as e:
            raise CustomException(e, sys)
