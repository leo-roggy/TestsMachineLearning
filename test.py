
# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data Preprocessing

# Import du dataset
dataset = pd.read_csv("earnings.csv")
X = dataset.iloc[:, 5].values
y = dataset.iloc[:, 0].values
frequency = dataset.iloc[:, 2].values

# replace date by year
for index, date in enumerate(X):
    if(type(date) == str):
        date = date[0:4]
        date = 2020-int(date)
        date = np.clip(date, 10, 90)
        X[index]=date

# Gérer les données manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#X[:, 0] = imputer.fit_transform(X[:, 0].reshape(-1, 1)).reshape(1, -1)
X[:] = imputer.fit_transform(X[:].reshape(-1, 1)).reshape(1, -1)

# mettre toutes les valeurs en mensuelle
for index, frec in enumerate(frequency):
    if(frec == 'ANNU'):
        y[index] = y[index] / 12.0


# Diviser le dataset entre le Training set et le Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Construction du modèle
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Faire de nouvelles prédictions
y_pred = regressor.predict(X_test.reshape(-1, 1)).reshape(1, -1)


# Visualiser les résultats
plt.scatter(X_train, y_train, color = 'green')
#plt.scatter(X_test, y_test, color = 'red')
#plt.scatter(X_test, y_pred, color = 'yellow')
#plt.plot(X_train, regressor.predict(X_train.reshape(-1, 1)).reshape(1, -1), color = 'blue')
plt.plot(X_train, regressor.predict(X_train.reshape(-1, 1)), color = 'blue')
plt.title('Salaire vs Age')
plt.xlabel('Age')
plt.ylabel('Salaire')
plt.show()