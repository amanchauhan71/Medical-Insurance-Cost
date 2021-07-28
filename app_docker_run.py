from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
pickle_in = open('insurance.pkl', 'rb')
insurance = pickle.load(pickle_in)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict_insurance_charge():
    if request.method == 'POST':

        age = request.form['age']
        # print(Age)
        bmi = request.form['bmi']
        children = request.form['children']
        sex = request.form['sex']
        if sex == "Male":
            sex_male = 1
        else:
            sex_male = 0

        smoker = request.form['smoker']

        if smoker == "yes":
            smoker_yes = 1
        else:
            smoker_yes = 0

        region = request.form['region']

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
        else:
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
        return render_template('index.html', prediction_text=str(prediction))

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
