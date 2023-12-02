import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from dataclasses import dataclass
import src.utils as srut
# import data_ingestion as di

# Configuration
@dataclass
class DataTransformationConfig():
    preprocessor_obj_path=os.path.join('artifacts','preprocessor.pkl')


# Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def data_transformer_object(self, train_df):
        try:
            logging.info("Process of creating preprocessor started")
            train_df.drop('Loan_Status', inplace=True, axis=1)
            numerical_cols=[x for x in train_df if train_df[x].dtype != 'O']
            categorical_cols=[x for x in train_df if train_df[x].dtype == 'O']
            logging.info("Creating Pipeline")
            # logging.info("Num pipe")

            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            # logging.info("Cat pipe")
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(categories='auto')),
                    ('scaler', StandardScaler())
                ]
            )
            # logging.info("col transformler")
            preprocessor=ColumnTransformer([
                ('numerical_pipeline',num_pipeline, numerical_cols),
                ('categorical_pipeline',cat_pipeline,categorical_cols)
            ])
            logging.info("Pipeline Creation completed")
            return preprocessor
        
        except Exception as e:
            logging.info(f"Error in creation of preprocessor in data_transformer_object: {str(e)}")
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            X_train, X_test=train_df.iloc[:,:-1], test_df.iloc[:,:-1]
            y_train, y_test=train_df.iloc[:,-1], test_df.iloc[:,-1]
            
            logging.info("Obtaining preprocessor object")
            preprocessor=self.data_transformer_object(train_df)
            logging.info("Obtained preprocessor")
            
            logging.info("Applying preprocessor object")
            X_train_arr=preprocessor.fit_transform(X_train)
            X_test_arr=preprocessor.transform(X_test) # preprocessing will convert the data into array

            logging.info("Combining the tranformed model as train and test df")
            train_arr=np.c_[X_train_arr, np.array(y_train)]
            test_arr=np.c_[X_test_arr, np.array(y_test)]

            srut.save_object(preprocessor, self.data_transformation_config.preprocessor_obj_path)
            logging.info("Transformation Completed")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        
        except Exception as e:
            logging.info(f"Error occurred in initiate_data_transformation: {str(e)}")
            raise CustomException(e,sys)

#Usage
# obj1=di.DataIngestion()
# a,b=obj1.initiate_data_ingestion()
# # print(a)
# obj2=DataTransformation()
# p1,p2,_=obj2.initiate_data_transformation(a,b)
# print(p1)