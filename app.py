import pickle
from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        # request all the input fields
        Latitude = float(request.form['Latitude'])
        Longitude = float(request.form['Longitude'])
        Year = int(request.form['Year'])
        pH = float(request.form['pH'])
        EC = float(request.form['EC in μS/cm'])
        CO3 = float(request.form['CO3'])
        HCO3 = float(request.form['HCO3'])
        Cl = float(request.form['Cl'])
        SO4 = float(request.form['SO4'])
        NO3 = float(request.form['NO3'])
        PO4 = float(request.form['PO4'])
        TH = float(request.form['TH'])
        Ca = float(request.form['Ca'])
        Mg = float(request.form['Mg'])
        Na = float(request.form['Na'])
        K = float(request.form['K'])
        F = float(request.form['F'])
        TDS = float(request.form['TDS'])


        # create numpy array for all the inputs
        # val = np.array([ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity])
        val = np.array([Latitude, Longitude, Year, pH, EC , CO3, HCO3, Cl,SO4, NO3, PO4, TH, Ca, Mg, Na, K,F ,TDS])
        # print (val)
        data=val.reshape(1,18)

        # define save model and scaler path
        model_path = os.path.join('models', 'clf.sav')
        scaler_path = os.path.join('models', 'scaler.sav')

        # load the model and scaler
        model = pickle.load(open(model_path, 'rb'))
        scc = pickle.load(open(scaler_path, 'rb'))

        # transform the input data using pre fitted standard scaler
        #data = scc.transform([val])

        # make a prediction for the given data
        res = model.predict(data)

        if res == 1:
            outcome = 'Potable'
        else:
            outcome = 'not potable'
        return render_template('index.html', result=outcome)
    return render_template('index.html')

# run application
if __name__ == "__main__":
    app.run(debug=True)
