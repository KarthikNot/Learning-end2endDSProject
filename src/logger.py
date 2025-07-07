import logging, os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")

os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH, 
                    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)