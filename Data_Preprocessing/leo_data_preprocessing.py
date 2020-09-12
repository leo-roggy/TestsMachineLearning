# Data Preprocessing

# Importer les librairies
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Import du dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Gérer les données manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Gérer les variables catégoriques
# Gestion variable Pays
countries = np.unique(X[:,0])
onehotencoder = OneHotEncoder(categories=[countries])#indice des colonnes à transormer
tmp = onehotencoder.fit_transform(X[:,0].reshape(-1, 1)).toarray()
X = np.append(tmp, X[:, 1:], axis=1)
# Gestion variable achat
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Diviser le dataset entre le Training set et le Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train[:, 3:5])
X_train[:, 3:5] = sc.transform(X_train[:, 3:5])
X_test[:, 3:5] = sc.transform(X_test[:, 3:5])