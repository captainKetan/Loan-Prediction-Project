from src.exception import CustomException
from src.logger import logging
import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd
import numpy as np
from dataclasses import dataclass

if __name__=="__main__":
    obj=DataIngestion()
    train_df_path,test_df_path=obj.initiate_data_ingestion()

    data_trans=DataTransformation()
    train_arr,test_arr,_=data_trans.initiate_data_transformation(train_df_path, test_df_path)

    model=ModelTrainer()
    model.initiate_model_training(train_arr, test_arr)