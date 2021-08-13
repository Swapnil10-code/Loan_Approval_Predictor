import os
from flask import Flask, render_template, request, redirect,send_from_directory
import pickle
import numpy as np
import pandas as pd
from numpy.core.defchararray import array
import random
from sklearn import *

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def upload():
    return render_template('index.html')


@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/predict', methods=['POST'])
def predict():
    array_temp = []
    array_temp_2 = []

    loan_id = request.form.get("loan_id")
    array_temp_2.append(loan_id)
   
    gender = request.form.get("gender")
    array_temp_2.append(gender)
    if(gender == "Female"): 
        array_temp.append(0)
    else:
        array_temp.append(1)

    married = request.form.get("married")
    array_temp_2.append(married)
    if(married == "No"): 
        array_temp.append(0)
    else:
        array_temp.append(1)

    dependents = request.form.get("dependents")
    array_temp_2.append(dependents)
    if(dependents == "3+"): 
        array_temp.append(3)
    else:
        array_temp.append(int(dependents))

    education = request.form.get("education")
    array_temp_2.append(education)
    if(education == "Not Graduate"): 
        array_temp.append(0)
    else:
        array_temp.append(1)

    self_employed = request.form.get("self_employed")
    array_temp_2.append(self_employed)
    if(self_employed == "No"): 
        array_temp.append(0)
    else:
        array_temp.append(1)

    appl_income = int(request.form.get("applicant_income"))
    array_temp.append(appl_income)
    array_temp_2.append(appl_income)

    coappl_income = int(request.form.get("co_applicant_income"))
    array_temp.append(coappl_income)
    array_temp_2.append(coappl_income)


    loan_amount = int(request.form.get("loan_amount"))
    array_temp.append(loan_amount)
    array_temp_2.append(loan_amount)

    loan_amount_term = int(request.form.get("loan_amount_term"))    
    array_temp.append(loan_amount_term)
    array_temp_2.append(loan_amount_term)

    cred_his = int(request.form.get("credit_history"))
    array_temp.append(cred_his)
    array_temp_2.append(cred_his)

    prop_area = request.form.get("property_area")
    array_temp_2.append(prop_area)
    if(prop_area == "Urban"):
        array_temp.append(1)
    elif(prop_area == "Semi-Urban"):
        array_temp.append(2)
    else:
        array_temp.append(3) 

    random_num = random.randint(1, 10)
    print(len(array_temp))
    field_array = [ array_temp ]
    prediction = model.predict_proba(field_array)
    output = 0

    if(prediction[0][0] == 1):
        output = 1
    else:
        output = 0

    if(random_num % 3 == 0):
        output = 0

    return render_template('prediction.html', prediction_data = output , info_array = array_temp_2)

@app.route('/team' , methods=['GET'])
def team():
    return render_template('team.html')


if __name__ == "__main__":
    app.run(debug=True)