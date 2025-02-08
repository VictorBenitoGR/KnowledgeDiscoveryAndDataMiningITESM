# * Arbol de decisión

# *** Importaciones -----------------------------------------------------------

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import plotly.graph_objects as go
from sklearn.tree import export_graphviz
import graphviz

np.random.seed(17)

# *** Carga de datos ----------------------------------------------------------

df = pd.read_csv("./data/500hits.csv", encoding="latin-1")
df.head()

#          PLAYER  YRS     G     AB     R     H   2B   3B   HR   RBI    BB    SO   SB   CS     BA  HOF
# 0       Ty Cobb   24  3035  11434  2246  4189  724  295  117   726  1249   357  892  178  0.366    1
# 1   Stan Musial   22  3026  10972  1949  3630  725  177  475  1951  1599   696   78   31  0.331    1
# 2  Tris Speaker   22  2789  10195  1882  3514  792  222  117   724  1381   220  432  129  0.345    1
# 3   Derek Jeter   20  2747  11195  1923  3465  544   66  260  1311  1082  1840  358   97  0.310    1
# 4  Honus Wagner   21  2792  10430  1736  3430  640  252  101     0   963   327  722   15  0.329    1

# *** Arbol de decision -------------------------------------------------------

df = df.drop(columns=["PLAYER", "CS"])
# Asegúrate de que esto selecciona correctamente las columnas
X = df.iloc[:, :13]
y = df.iloc[:, 13]

scaler = StandardScaler()  # Inicializa el escalador
X_scaled = scaler.fit_transform(X)  # Escala los datos

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, random_state=17, test_size=0.2)

X_train.shape
y_test.shape

dtc = DecisionTreeClassifier(random_state=17)

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

confusion_matrix(y_test, y_pred)
# [[52  9]
#  [10 22]]

# 52 + 22 = 74
# 9 + 10 = 19

# Accuracy = (52 + 22) / (52 + 22 + 9 + 10) = 0.7956 o 79.56%

print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support

#            0       0.84      0.85      0.85        61
#            1       0.71      0.69      0.70        32

#     accuracy                           0.80        93
#    macro avg       0.77      0.77      0.77        93
# weighted avg       0.79      0.80      0.79        93

dtc.feature_importances_
# Cada uno significa que tan importante es la variable para el modelo

# array([0.02394512, 0.03355581, 0.03380506, 0.07328956, 0.38574252,
#        0.05755522, 0.05217487, 0.        , 0.08416125, 0.07281114,
#        0.03117801, 0.04489398, 0.10688743])

X.columns

# Index(['YRS', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB',
#        'BA'],

ft = pd.DataFrame(dtc.feature_importances_, index=X.columns, columns=["Importancia"])
ft = ft.sort_values(by="Importancia", ascending=False).head(15)
ft

#      Importancia
# H       0.385743
# BA      0.106887
# RBI     0.084161
# R       0.073290
# BB      0.072811
# 2B      0.057555
# 3B      0.052175
# SB      0.044894
# AB      0.033805
# G       0.033556
# SO      0.031178
# YRS     0.023945
# HR      0.000000

# *** Segundo arbol de decision -----------------------------------------------

dtc2 = DecisionTreeClassifier(ccp_alpha=0.004, criterion="entropy")

# La entropía es una medida de la impureza de un nodo, y se utiliza para evitar
# el sobreajuste.

dtc2.fit(X_train, y_train)

y_pred2 = dtc2.predict(X_test)

print(confusion_matrix(y_test, y_pred2))

# [[49 12]
#  [14 18]]

# Exportar el árbol de decisión a un archivo .dot
dot_data = export_graphviz(dtc2, out_file=None, 
                           feature_names=X.columns,  
                           class_names=[f'Clase {i}' for i in np.unique(y_train)],
                           filled=True, rounded=True,  
                           special_characters=True)  

# Crear un gráfico a partir del archivo .dot
graph = graphviz.Source(dot_data)
graph.render("arbol_decision")  # Esto generará un archivo PDF llamado 'arbol_decision.pdf'

# Mostrar el gráfico en un objeto gráfico de Plotly
fig = go.Figure(data=[go.Image(z=graph.pipe(format='png'))])
fig.update_layout(title="Árbol de Decisión", width=1000, height=500)
fig.show()

