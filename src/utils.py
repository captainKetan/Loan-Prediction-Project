import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, confusion_matrix

def save_object(object,file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(object,f)
    except Exception as e:
        logging.info("Failed to save object")
        raise CustomException(e,sys)
    
def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        report, trained_models={}, {}
        for i in models:
            model=models[i]
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            score=accuracy_score(y_pred,y_test)
            report.update({i:score})
            trained_models.update({i:model})
        return (report, trained_models)
    
    except Exception as e:
        logging.info("Failed to save object")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.info("Failed to save object")
        raise CustomException(e,sys)