# Regression Polynomiale

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Importer le dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#X_test = np.array([[0], [3.5], [6.5], [7.5], [9.5]])
X_test = np.arange(0, 10.1, 0.05).reshape(-1, 1)

# Diviser le dataset entre le Training set et le Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Construction du modèle
poly = PolynomialFeatures(degree=5)
poly.fit(X, y)
X_poly = poly.transform(X)
X_test_poly = poly.transform(X_test)


linReg = LinearRegression()
linReg.fit(X_poly, y)


# Visualiser les Data
plt.scatter(X, y)
#plt.plot(X, linReg.predict(X_poly))

plt.plot(X_test, linReg.predict(X_test_poly))

#plt.plot(X, poly.predict(X))
#plt.xlabel('Position')
#plt.ylabel('Salaire')