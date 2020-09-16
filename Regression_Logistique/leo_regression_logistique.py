# Regression Logistique

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Import du dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
# gestion du genre
X_with_genre = dataset.iloc[:, 1:-1].values
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# Gérer les variables catégoriques
# Gestion variable Genre
le = LabelEncoder()
le.fit(X_with_genre[:, 0])
X_with_genre[:, 0] = le.transform(X_with_genre[:, 0])

# Diviser le dataset entre le Training set et le Test set
X_with_genre_train, X_with_genre_test, y_with_genre_train, y_with_genre_test = train_test_split(X_with_genre, y, test_size = 0.2, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train[:, :])
X_train[:, :] = sc.transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

sc.fit(X_with_genre_train[:, 1:])
X_with_genre_train[:, 1:] = sc.transform(X_with_genre_train[:, 1:])
X_with_genre_test[:, 1:] = sc.transform(X_with_genre_test[:, 1:])

# Construction du modèle
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Faire de nouvelles prédictions
y_pred = classifier.predict(X_test)

classifier_with_genre = LogisticRegression()
classifier_with_genre.fit(X_with_genre_train, y_with_genre_train)
y_with_genre_pred = classifier_with_genre.predict(X_with_genre_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
cm_with_genre = confusion_matrix(y_with_genre_test, y_with_genre_pred)

# Visualiser les résultats
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Résultats du Training set')
plt.xlabel('Age')
plt.ylabel('Salaire Estimé')
plt.legend()