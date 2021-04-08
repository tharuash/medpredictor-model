import pickle
from sklearn.preprocessing import PolynomialFeatures

model = pickle.load(open('adorastatin_model.pkl','rb'))
poly = PolynomialFeatures(degree = 3)
print(model.predict(poly.fit_transform([[120]])))
#print(model.predict([[121]]))
