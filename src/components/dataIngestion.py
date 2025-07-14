import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.dataTransformation import dataTransformation, dataTransformationConfig

@dataclass
class dataIngestionConfig:
    trainDataPath : str = os.path.join('artifacts', 'train.csv')
    testDataPath : str = os.path.join('artifacts', 'test.csv')
    rawDataPath : str = os.path.join('artifacts', 'raw.csv')

class dataIngestion:
    def __init__(self):
        self.ingestionConfig = dataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info('Entered the Data Ingestion')
        try:
            df = pd.read_csv('./notebooks/data/stud.csv')
            logging.info('Data read as DataFrame!')

            os.makedirs(os.path.dirname(self.ingestionConfig.trainDataPath), exist_ok=True)

            df.to_csv(self.ingestionConfig.rawDataPath, index = False, header=True)
            logging.info('Train and Test sets intiated!')

            trainSet, testSet = train_test_split(df, test_size = 0.2, random_state = 1, shuffle = True)

            trainSet.to_csv(self.ingestionConfig.trainDataPath, index = False, header = True)
            trainSet.to_csv(self.ingestionConfig.testDataPath, index = False, header = True)
            logging.info('Data Ingestion completed!')

            return [self.ingestionConfig.trainDataPath, self.ingestionConfig.testDataPath]
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = dataIngestion()
    train_data, test_data = obj.initiateDataIngestion()

    data_transformation = dataTransformation()
    data_transformation.initiateDataTransformation(train_data, test_data)