import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class Predict:
    def __init__(self):
        pass

    def predict_result(self,df):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_data=preprocessor.transform(df)
            pred=model.predict(scaled_data)
            res='No'
            if pred[0]==1.0:
                res='Yes'
            return res
        
        except Exception as e:
            logging.info(f'Exception Occured in prediction pipeline: {str(e)}')
            raise CustomException(e,sys)

class ProcessData:
    def __init__(self,
                 gend:str,
                 marr:str,
                 dep:str,
                 edu:str,
                 self_emp:str,
                 appli_income:float,
                 coappli_income:float,
                 loan_amt:float,
                 loan_amt_term:float,
                 credit_hist:float,
                 property_area:str):
        self.gend=gend
        self.marr=marr
        self.dep=dep
        self.edu=edu
        self.self_emp=self_emp
        self.appli_income=appli_income
        self.coappli_income=coappli_income
        self.loan_amt=loan_amt
        self.loan_amt_term=loan_amt_term
        self.credit_hist=credit_hist
        self.property_area=property_area

    def data_to_dataframe(self):
        try:
            input_dict={
                'Gender':[self.gend],
                'Married':[self.marr],
                'Dependents':[self.dep],
                'Education':[self.edu],
                'Self_Employed':[self.self_emp],
                'ApplicantIncome':[self.appli_income],
                'CoapplicantIncome':[self.coappli_income],
                'LoanAmount':[self.loan_amt],
                'Loan_Amount_Term':[self.loan_amt_term],
                'Credit_History':[self.credit_hist],
                'Property_Area':[self.property_area]
            }
            df=pd.DataFrame(input_dict)
            logging.info("DataFrame created")
            return df
        
        except Exception as e:
            logging.info(f'Exception Occured in prediction pipeline: {str(e)}')
            raise CustomException(e,sys)