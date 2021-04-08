import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    model = None
    poly = None

    medicine = data['medicine']
    if medicine == 'adorastatin':
        poly = PolynomialFeatures(degree = 3)
        model = pickle.load(open('adorastatin_model.pkl', 'rb'))
    elif medicine == 'aspirin':
        poly = PolynomialFeatures(degree = 4)
        model = pickle.load(open('aspirin_model.pkl', 'rb'))
    elif medicine == 'losatank':
        poly = PolynomialFeatures(degree = 2)
        model = pickle.load(open('losatank_model.pkl', 'rb'))
    elif medicine == 'metformin':
        poly = PolynomialFeatures(degree = 2)
        model = pickle.load(open('metformin_model.pkl', 'rb'))
    elif medicine == 'niffidipine':
        poly = PolynomialFeatures(degree = 4)
        model = pickle.load(open('niffidipine_model.pkl', 'rb'))     
    else:
        return jsonify({'success' : False, 'error' : 'Invalid Medicine'})


    x = int(data['month']) + ((int(data['year']) - 2013)*12) - 1
    prediction = model.predict(poly.fit_transform([[x]]))

    return jsonify({'success' : True, 'prediction' : prediction[0][0]})


if __name__ == "__main__":
    app.run(debug=True)