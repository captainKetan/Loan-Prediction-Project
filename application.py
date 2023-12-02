from flask import Flask, render_template, request, jsonify
import os
import sys
from src.pipelines.prediction_pipeline import ProcessData, Predict


application=Flask(__name__)
app=application

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_result():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=ProcessData(
            gend=str(request.form.get('Gender')),
            marr=str(request.form.get('Married')),
            dep=str(request.form.get('Dependents')),
            edu=str(request.form.get('Education')),
            self_emp=str(request.form.get('Self_Employed')),
            appli_income=float(request.form.get('ApplicantIncome')),
            coappli_income=float(request.form.get('CoapplicantIncome')),
            loan_amt=float(request.form.get('LoanAmount')),
            loan_amt_term=float(request.form.get('Loan_Amount_Term')),
            credit_hist=float(request.form.get('Credit_History')),
            property_area=str(request.form.get('Property_Area'))
        )
        data_df=data.data_to_dataframe()
        obj=Predict()
        output_pred='Eligible for Loan: ' + obj.predict_result(data_df)

        return render_template('form.html', final_result=output_pred)
    
if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)