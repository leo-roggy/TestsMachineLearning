# Regression LinÃ©aire Multiple

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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