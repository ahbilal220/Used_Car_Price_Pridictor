from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('Car_Price_Predictor.pkl', 'rb'))
car = pd.read_csv('OLX_cars_dataset00.csv')

@app.route('/')
def index():
    companies = sorted(car['Company'].unique())
    car_models = sorted(car['Name'].unique())
    year = sorted(car['Year'].unique(), reverse=True)
    fuel_type = car['Fuel'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Ensure data types are correct
    year = int(year)
    

    # Correct DataFrame with actual inputs
    input_data = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                              columns=['Name', 'Company', 'Year', 'Mileage', 'Fuel'])

    prediction = model.predict(input_data)
    print(prediction)

    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run()
