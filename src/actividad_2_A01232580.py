# * Actividad 2. Análisis de conglomerados - A01232580

# *** Importaciones -----------------------------------------------------------

import numpy as np  # Cálculos
import pandas as pd  # Manipulación de datos
import plotly.graph_objects as go  # Gráficos interactivos
import requests  # Peticiones HTTP
from sklearn.cluster import KMeans  # Clustering
from sklearn.metrics import silhouette_score  # Evaluación
from sklearn.preprocessing import StandardScaler  # Escalamiento


# *** Obtención de datos ------------------------------------------------------

# * Definir año e indicadores
year = 2023
indicators = {
    'NY.GDP.PCAP.CD': 'GDP per capita',
    'SL.UEM.TOTL.ZS': 'Unemployment rate',
    'NV.IND.TOTL.ZS': 'Industry value added'
}

# * Obtener datos del Banco Mundial
base_url = "http://api.worldbank.org/v2/country/all/indicator/"
data_list = []

for indicator_code in indicators:
    url = f"{base_url}{indicator_code}"
    params = {
        'date': year,
        'format': 'json',
        'per_page': 300
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()[1]
        for entry in data:
            if entry['value'] is not None:
                data_list.append({
                    'country': entry['country']['value'],
                    'indicator': indicators[indicator_code],
                    'value': float(entry['value'])
                })


# * Crear DataFrame
df = pd.pivot_table(
    pd.DataFrame(data_list),
    values='value',
    index='country',
    columns='indicator'
).reset_index()

# *** Preparación de datos ----------------------------------------------------

# Eliminar filas con valores faltantes
df = df.dropna()

print(df.head())
print("\nNúmero de países:", len(df))

# * Seleccionar características para clustering
features = list(indicators.values())
X = df[features].values

# * Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# *** Número óptimo de clústeres usando el método del codo --------------------

# * Lista para almacenar la suma de errores cuadráticos
sec = []

# Comenzar desde 1 porque el codo se puede calcular desde 1 clúster
for k in range(1, 101):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sec.append(kmeans.inertia_)

# * Número óptimo de clústeres
optimal_k_elbow = np.diff(sec, 2).argmin() + 2  # +2 porque comenzamos en 1

# * Visualización
fig_elbow = go.Figure()

fig_elbow.add_trace(go.Scatter(
    x=list(range(1, 101)),
    y=sec,
    mode='lines+markers',
    name='SSE',
    marker=dict(size=10),
))

fig_elbow.add_shape(
    type='line',
    x0=optimal_k_elbow, y0=0,
    x1=optimal_k_elbow, y1=max(sec),
    line=dict(color='red', width=2, dash='dash'),
)

fig_elbow.update_layout(
    title='<b>Número óptimo de clústeres usando el método del codo</b>',
    xaxis_title='Número de clústeres',
    yaxis_title='Suma de errores cuadráticos',
    template='simple_white',
    showlegend=False
)

# * Visualizar
fig_elbow.show()

# * Interpretación
# Podemos ver que el número óptimo de clústeres es de 5. Para solucionar un
# problema con un socioformador había visto que existe también el puntaje de
# silueta, por lo que por curiosidad veremos qué tanto difiere.


# *** Número óptimo de clústeres usando el método de la silueta ---------------

# * Lista para almacenar las puntuaciones de silueta
silhouette_scores = []

# Comenzar desde 2 porque la silueta no se puede calcular para 1 clúster
for k in range(2, 101):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# * Número óptimo de clústeres
optimal_k_silhouette = silhouette_scores.index(
    max(silhouette_scores)) + 2  # +2 porque comenzamos en 2

# * Visualización
fig_silhouette = go.Figure()

fig_silhouette.add_trace(go.Scatter(
    x=list(range(2, 101)),
    y=silhouette_scores,
    mode='lines+markers',
    name='Puntuación de Silueta',
    marker=dict(size=10),
))

fig_silhouette.add_shape(
    type='line',
    x0=optimal_k_silhouette, y0=0,
    x1=optimal_k_silhouette, y1=max(silhouette_scores),
    line=dict(color='red', width=2, dash='dash'),
)

fig_silhouette.update_layout(
    title='<b>Número óptimo de clústeres usando el puntaje de silueta</b>',
    xaxis_title='Número de clústeres',
    yaxis_title='Puntuación de silueta',
    template='simple_white',
    showlegend=False
)

# * Visualizar
fig_silhouette.show()

# * Interpretación
# En este caso el número óptimo es de 3. Investigando al respecto, la principal
# diferencia es que el método del codo busca minimizar la suma de errores
# cuadráticos, mientras que el puntaje de silueta busca maximizar la separación
# entre clústeres. Usaré la SEC porque

# *** KMeans ------------------------------------------------------------------

# * Modelo
kmeans = KMeans(
    # Número de clústeres a encontrar
    n_clusters=optimal_k_elbow,
    # Número máximo de iteraciones
    max_iter=100,
    # Reproducibilidad
    random_state=42
)

# * Entrenamiento
kmeans.fit(X_scaled)

# * Predicción
y_pred = kmeans.fit_predict(X_scaled)
df['Cluster'] = y_pred + 1


# *** Gráfico de dispersión 2D ------------------------------------------------

# * Colores para los clústeres
colors = ['green', 'blue', 'purple', 'red', 'brown']

# * Formas
shapes = ['diamond', 'circle', 'x', 'square', 'cross']

# * Creación
fig_2d = go.Figure()

for cluster in np.unique(y_pred + 1):
    cluster_data = df[df['Cluster'] == cluster]

    # * Identificar a México de otros países
    mexico_data = cluster_data[cluster_data['country'] == 'Mexico']
    other_data = cluster_data[cluster_data['country'] != 'Mexico']

    # * Detalles generales
    if not other_data.empty:
        fig_2d.add_trace(go.Scatter(
            x=other_data['GDP per capita'],
            y=other_data['Unemployment rate'],
            mode='markers',
            name=f'Cluster {cluster}',
            text=other_data['country'],
            marker=dict(
                symbol=shapes[cluster - 1],
                size=10,
                color=colors[cluster - 1]
            ),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'GDP per capita: %{x:,.0f}<br>' +
                'Unemployment: %{y:.1f}%<br>'
            )
        ))

    # * Resaltar México
    if not mexico_data.empty:
        fig_2d.add_trace(go.Scatter(
            x=mexico_data['GDP per capita'],
            y=mexico_data['Unemployment rate'],
            mode='markers+text',
            name='México',
            textposition='top center',
            marker=dict(
                symbol=shapes[cluster - 1],
                size=12,
                color=colors[cluster - 1],
                line=dict(width=2, color='black')
            ),
            hovertemplate=(
                '<b>México</b><br>' +
                'GDP per capita: %{x:,.0f}<br>' +
                'Unemployment: %{y:.1f}%<br>'
            )
        ))


# * Detalles
fig_2d.update_layout(
    title='<b>Clústeres de países: PIB per cápita vs Tasa de desempleo</b>',
    xaxis_title='PIB per cápita (USD)',
    yaxis_title='Tasa de desempleo (%)',
    template='simple_white',
    showlegend=True
)

# * Visualizar
fig_2d.show()

# * Exportar
fig_2d.write_html('./assets/actividad_2_2D.html')

# * Interpretación
# Es curioso ver que México se encuentra al lado de Rusia quien en 2023 ya tenía
# un año de guerra por la invasión de Ucrania. Podemos apreciar algunos países
# en extremos como Luxemburgo de gran PIB per cápita y reducida tasa de
# desempleo, así como el caso de "Eswatini" (o "Suazilandia" en español) que
# para ser honesto desconocía, y que se encuentra en el extremo opuesto con un
# reducido PIB per cápita y más de un tercio de la población desempleada.
# Resulta extraño que el 4°to cluster tiene un solo país, Libia, que sin embargo
# no está en un extremo como el caso anterior.

# Se perfectamente que los gráficos 3D no son buenos en general, sin embargo al
# tratarse de clusterización de múltiples dimensiones lo considero oportuno, más
# por el caso de Libia.

# *** Gráfico 3D --------------------------------------------------------------

# * Print colnames
print(df.columns)

# * Creación
fig_3d = go.Figure()

for cluster in np.unique(y_pred + 1):
    cluster_data = df[df['Cluster'] == cluster]

    # * Identificar a México de otros países
    mexico_data = cluster_data[cluster_data['country'] == 'Mexico']
    other_data = cluster_data[cluster_data['country'] != 'Mexico']

    # * Detalles generales
    if not other_data.empty:
        fig_3d.add_trace(go.Scatter3d(
            x=other_data['GDP per capita'],
            y=other_data['Unemployment rate'],
            z=other_data['Industry value added'],
            mode='markers',
            name=f'Cluster {cluster}',
            text=other_data['country'],
            marker=dict(
                symbol=shapes[cluster - 1],
                size=5,
                color=colors[cluster - 1]
            ),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'GDP per capita: %{x:,.0f}<br>' +
                'Unemployment: %{y:.1f}%<br>' +
                'Industry value: %{z:.1f}%<br>'
            )
        ))

    # * Resaltar México
    if not mexico_data.empty:
        fig_3d.add_trace(go.Scatter3d(
            x=mexico_data['GDP per capita'],
            y=mexico_data['Unemployment rate'],
            z=mexico_data['Industry value added'],
            mode='markers+text',
            name='México',
            marker=dict(
                symbol=shapes[cluster - 1],
                size=7,
                color=colors[cluster - 1],
                line=dict(width=2, color='black')
            ),
            hovertemplate=(
                '<b>México</b><br>' +
                'GDP per capita: %{x:,.0f}<br>' +
                'Unemployment: %{y:.1f}%<br>' +
                'Industry value: %{z:.1f}%<br>'
            )
        ))

# * Detalles
fig_3d.update_layout(
    title='<b>Clústeres de países (3D)</b>',
    scene=dict(
        xaxis_title='PIB per cápita (USD)',
        yaxis_title='Tasa de desempleo (%)',
        zaxis_title='Valor agregado industrial (%)'
    ),
    template='plotly'
)

# * Visualizar
fig_3d.show()

# * Exportar
fig_3d.write_html('./assets/actividad_2_3D.html')

# * Interpretación
# Podemos ver que Libia sí que se encontraba en un extremos que sin embargo no
# podíamos apreciar viendo los datos desde una sola cara, pues la dimensión del
# valor agregado lo pone claramente como un outlier que amerita su propio
# clúster. Investigando al respecto, el 95% de sus exportaciones son petróleo y
# gas, sin embargo su PIB es equiparable al resto de países africanos y casi un
# quinto de la población está desempleada.
