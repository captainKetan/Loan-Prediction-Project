import os
from src.exception import CustomException
from src.logger import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

# Configuration
@dataclass
class DataIngestionConfig():
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    fetched_data_path=os.path.join('artifacts','fetched.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion starts now')

        try:
            os.makedirs(os.path.dirname(self.data_ingestion_config.fetched_data_path), exist_ok=True)
            df=pd.read_csv('notebook/data/data.csv')
            df.to_csv(self.data_ingestion_config.fetched_data_path, index=False)
            df.drop('Loan_ID', inplace=True, axis=1)
            # logging.info("Raw data fetched")
            encoder=LabelEncoder()
            df['Loan_Status']=encoder.fit_transform(df['Loan_Status'])  # Encoding the target feature
            # logging.info("Target variable encoding done")
            
            train, test=train_test_split(df, test_size=0.20, random_state=42)
            train.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion completed')
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info(f'Error occurred in Data Ingestion: {str(e)}')


# Usage
# obj = DataIngestion()
# p1,p2 = obj.initiate_data_ingestion()
# df1=pd.read_csv(p1)
# print(df1.head())