# * Minería de Datos (ITESM) - Exploración de datos

# *** Bibliotecas --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# *** Importación de datos ----------------------------------------------------

# * Importar datos
resultados = pd.read_excel("data/Centros Comerciales EM19 Tr5.xlsx")

# *Guardar nombre de las columnas en el portapapeles
pd.DataFrame(resultados.columns).to_clipboard()

# ** Constantes ---------------------------------------------------------------

# Paleta de colores Okabe-Ito
okabe_ito = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7"
]

# *** Funciones ---------------------------------------------------------------

# * Definir clase de cada columna


def definir_clase(data):
    """
    Esta función convierte las columnas de un DataFrame a sus tipos de datos
    apropiados. La columna 'ID' se convierte a tipo carácter,
    'Duration__in_seconds_' se convierte a numérico, y todas las demás columnas
    se convierten a tipo categórico.

    Parameters: data (DataFrame): Un DataFrame que contiene los datos a
    procesar.

    Returns: DataFrame: Un DataFrame con las columnas convertidas a sus
    respectivas clases.
    """
    data['ID'] = data['ID'].astype(str)
    data['Duration__in_seconds_'] = pd.to_numeric(
        data['Duration__in_seconds_'], errors='coerce')
    for col in data.columns:
        if col not in ['ID', 'Duration__in_seconds_']:
            data[col] = data[col].astype('category')
    return data

# * Remover respuestas "0" o NA en "l1"


def remover_no_deseados(data, column):
    """
    Esta función elimina las filas de un DataFrame donde la columna especificada
    tiene un valor de "0" o es NA (Not Available).

    Parameters:
    data (DataFrame): Un DataFrame que contiene los datos a procesar.
    column (str): El nombre de la columna en el DataFrame que se desea procesar.

    Returns:
    DataFrame: Un DataFrame con las filas donde la columna especificada no tiene
    valores "0" o NA.
    """
    data = data[data[column] != 0]
    data = data[data[column].notna()]
    return data

# * Remover outliers de Duration__in_seconds_


def remover_outliers(data, column):
    """
    Esta función elimina los outliers de una columna específica en un DataFrame
    utilizando el método del rango intercuartílico (IQR). Los outliers se
    definen como aquellos valores que están por debajo de Q1 - 1.5 * IQR o por
    encima de Q3 + 1.5 * IQR, donde Q1 es el primer cuartil y Q3 es el tercer
    cuartil.

    Parameters: data (DataFrame): Un DataFrame que contiene los datos a
    procesar. column (str): El nombre de la columna en el DataFrame de la cual
    se desean remover los outliers.

    Returns: DataFrame: Un DataFrame con los outliers removidos de la columna
    especificada.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data


resultados = definir_clase(resultados)
resultados = remover_no_deseados(resultados, "l1")
resultados_sin_outliers = remover_outliers(resultados, "Duration__in_seconds_")

# Boxplot de Duration__in_seconds_
plt.figure(figsize=(4, 6))
sns.boxplot(
    data=resultados_sin_outliers,
    y="Duration__in_seconds_",
    color=okabe_ito[0]
)
plt.yscale('log')
plt.title("Duración en segundos")
plt.savefig("./assets/boxplot_segundos.png", dpi=600)
plt.show()
