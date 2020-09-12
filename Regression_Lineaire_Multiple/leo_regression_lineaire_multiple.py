# Regression LinÃ©aire Multiple

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Importer le dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Gérer les variables catégoriques
# Gestion variable Etat
states = np.unique(X[:,3])
onehotencoder = OneHotEncoder(categories=[states])
tmp = onehotencoder.fit_transform(X[:,3].reshape(-1, 1)).toarray()
X = np.append(X[:, :-1], tmp,  axis=1)
#supprimer une des dummy variables
X = X[:, 0:-1]

# Diviser le dataset entre le Training set et le Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Pas de Feature Scaling pour les regressions linéaires

# Feature Scaling
#sc = StandardScaler()
#sc.fit(X_train[:, :])
#X_train[:, :] = sc.transform(X_train[:, :])
#X_test[:, :] = sc.transform(X_test[:, :])

# Construction du modèle
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Faire de nouvelles prédictions
y_pred = regressor.predict(X_test)
regressor.predict(np.array([[150000, 100000, 50000, 1, 0]]))

# Visualiser les résultats
#plt.scatter(X_train[:, 0], y_train)
#plt.scatter(X_train[:, 1], y_train)
#plt.scatter(X_train[:, 2], y_train)
#plt.scatter(X_train[:, 3], y_train)
#plt.scatter(X_train[:, 4], y_train)

plt.scatter(X_test[:, 0], y_test, color = 'red')
plt.scatter(X_test[:, 0], y_pred, color = 'yellow')

#plt.plot(X_train[:, 0], regressor.predict(X_train))
#plt.plot(X_test[:, 0], y_pred)

#plt.plot(X_train[:, 0], regressor.predict(X_train))
#plt.plot(X_train[:, 1], regressor.predict(X_train))
#plt.title('Salaire vs Experience')
plt.xlabel('dépense R&D')
plt.ylabel('Profit')
plt.show()