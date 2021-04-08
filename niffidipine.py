import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('niffidipine20er.csv')

X = pd.DataFrame({})
y = pd.DataFrame({})

tempX_set = { 'x' : [] }
tempY_set = { 'y' : [] }
count = 0
for index, row in dataset.iterrows():
    for i in range(7):
        tempX_set['x'].append(count)
        tempY_set['y'].append(row[i+1])
        count = count + 1


X = pd.DataFrame(tempX_set)
Y = pd.DataFrame(tempY_set)

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
  
poly.fit(X_poly, y)

regressor = LinearRegression()
regressor.fit(X_poly, Y)

pickle.dump(regressor, open('niffidipine_model.pkl','wb'))
