import pandas as pd
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import src.utils as srut
import numpy as np
# import data_ingestion as di
# import data_transformation as dt


# Configuration
@dataclass
class ModelTrainerConfig():
    model_trainer_obj_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting data into i/p and o/p and as train and test")
            X_train, X_test, y_train, y_test = train_arr[:,:-1], test_arr[:,:-1], train_arr[:,-1], test_arr[:,-1]
            models={
                'LogisticRegression':LogisticRegression(),
                'SupportVectorClassifier':SVC(),
                'RandomForestClassifier':RandomForestClassifier(),
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'KNN':KNeighborsClassifier(),
                'GradientBoostClassifier':GradientBoostingClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier()
            }
            logging.info("Model Training starts")
            model_report, trained_models=srut.evaluate_model(X_train,X_test,y_train,y_test,models)
            logging.info("Training with multiple models completed")
            best_model_score=max(model_report.values())
            logging.info('Best model score founded')
            # best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_name = max(model_report, key=model_report.get)
            logging.info('Best Model Name founded')
            best_model=trained_models[best_model_name]
            logging.info('Best Model Taken out')

            print(f"Best model is {best_model_name} with accuray {best_model_score}")
            logging.info(f"Best model is {best_model_name} with accuray {best_model_score}")
            # print(model_report)

            srut.save_object(best_model,self.model_trainer_config.model_trainer_obj_path)


        except Exception as e:
            logging.info(f"Error in initiation of model training: {str(e)}")


# Usage
# obj1=di.DataIngestion()
# a,b=obj1.initiate_data_ingestion()
# # print(a)
# obj2=dt.DataTransformation()
# p1,p2,_=obj2.initiate_data_transformation(a,b)
# # print(p1)
# obj3=ModelTrainer()
# obj3.initiate_model_training(p1,p2)