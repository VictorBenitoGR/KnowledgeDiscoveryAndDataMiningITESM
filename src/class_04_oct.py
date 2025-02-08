
import pandas as pd
from requests import get

# URL de la base de datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = get(url)
print(response.text)

# Asignar nombres a las columnas
names = ['sepal_length', 'sepal_width',
        'petal_length', 'petal_width', 'class']

dataset = pd.read_csv(url, names=names)

# Mostrar
print(dataset.head())
