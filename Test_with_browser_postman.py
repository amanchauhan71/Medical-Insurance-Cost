from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
pickle_in = open('insurance.pkl', 'rb')
insurance = pickle.load(pickle_in)


@app.route("/")
def welcome():
    return "Welcome All"


@app.route("/predict")
def predict_student_performance():
    age = request.args.get('age')

    bmi = request.args.get('bmi')

    children = request.args.get('children')

    sex = request.args.get('sex')

    if sex == "Male":
        sex_male = 1
    else:
        sex_male = 0

    smoker = request.args.get('smoker')

    if smoker == "yes":
        smoker_yes = 1
    else:
        smoker_yes = 0

    region = request.args.get('region')

    if region == 'northwest':
        region_northwest = 1
        region_southeast = 0
        region_southwest = 0
    elif region == "southeast":
        region_northwest = 0
        region_southeast = 1
        region_southwest = 0
    elif region == "southwest":
        region_northwest = 0
        region_southeast = 0
        region_southwest = 1
    else :
        region_northwest = 0
        region_southeast = 0
        region_southwest = 0

    scaler = StandardScaler()
    filename_scaler = 'scaler_model.pickle'
    scaler_model = pickle.load(open(filename_scaler, 'rb'))
    scaled_data = scaler_model.transform([
        [age, bmi, children, sex_male, smoker_yes, region_northwest,
         region_southeast,
         region_southwest]])


    prediction = insurance.predict(scaled_data)
    return "The prediction value is " + str(prediction)


if __name__ == '__main__':
    app.run(debug=True)
