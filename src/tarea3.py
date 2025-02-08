# ./src/tarea3.py

# *** Importaciones -----------------------------------------------------------

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import contractions

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# *** Constantes --------------------------------------------------------------

# * URLs de los guiones
URL_ALADDIN_1992 = "https://imsdb.com/scripts/Aladdin.html"
URL_ALADDIN_2019 = "https://www.scripts.com/script/aladdin_2019_26877"

# * Categorías de personajes
PERSONAJES_PRINCIPALES = {
    'ALADDIN', 'JASMINE', 'GENIE', 'JAFAR',
}

PERSONAJES_SECUNDARIOS = {
    'SULTAN', 'IAGO', 'ABU', 'GAZEEM', 'PRINCE', 'PEDDLER'
}

PERSONAJES_CORO = {
    'BOTH', 'CHORUS', "MEN'S CHORUS", 'CHORUS OF MEN',
    'CHORUS OF WOMEN', 'ALADDIN and JASMINE', 'DUP. GENIES',
    'MARCHERS', 'WOMEN', 'CROWD'
}

PERSONAJES_EXTRAS = {
    'GUARD', 'GUARD 1', 'GUARD 2', 'GUARDS',
    'SHOPKEEPER 1', 'SHOPKEEPER 2', 'SHOPKEEPER 3', 'SHOPKEEPER 4',
    'BYSTANDER 1', 'BYSTANDER 2', 'WOMAN', 'WOMAN 1',
    'OLD MAN', 'PROPRIETOR', 'LADY', 'SWORDSMEN',
    'HARRY', 'JUNE', 'MAJOR'
}

PERSONAJES_AMBIENTE = {
    'CAVE', 'CAVE VOICE', 'BEE', 'FLAMINGO'
}

personajes = list(PERSONAJES_PRINCIPALES | PERSONAJES_SECUNDARIOS |
                  PERSONAJES_CORO | PERSONAJES_EXTRAS | PERSONAJES_AMBIENTE)


# *** Funciones ---------------------------------------------------------------

# * Extracción de la versión de 1992
def extraer_texto_aladdin_1992():
    # Configuración simplificada de Chrome
    options = Options()
    options.add_argument("--headless")

    # Crear el driver usando webdriver_manager
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.get(URL_ALADDIN_1992)

    # Extraer el texto
    guion_1992 = driver.find_element(
        By.CSS_SELECTOR, "#mainbody > table:nth-child(3) > tbody > tr > td:nth-child(3) > table > tbody > tr > td > pre")
    texto = guion_1992.text if guion_1992 else "No se pudo encontrar el guion."

    driver.quit()
    return texto


# * Extracción de la versión de 2019
def extraer_texto_aladdin_2019():
    """Extrae el texto del guión de Aladdin 2019 de múltiples páginas."""
    options = Options()
    options.add_argument("--headless")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    texto_completo = []

    # Obtener texto de la primera página (formato diferente)
    driver.get(URL_ALADDIN_2019)
    try:
        texto = driver.find_element(
            By.CSS_SELECTOR, "#disp-quote-int > blockquote").text
        texto_completo.append(texto)
    except:
        print("Error al extraer texto de la página principal")

    # Obtener texto de las páginas 2 a 5
    for pagina in range(2, 6):  # 6 porque python no cuenta el último número
        url = f"{URL_ALADDIN_2019}/{pagina}"
        driver.get(url)
        try:
            texto = driver.find_element(
                By.CSS_SELECTOR, "#disp-quote-body").text
            texto_completo.append(texto)
        except:
            print(f"Error al extraer texto de la página {pagina}")

    driver.quit()
    return ' '.join(texto_completo)


# * Limpieza de texto
def limpiar_texto(texto):
    """Limpia y normaliza el texto del guión."""
    texto = re.sub(r'\s+', ' ', texto).strip()  # Normalizar espacios
    texto = re.sub(r'\(.*?\)', '', texto)  # Eliminar paréntesis y su contenido
    texto = re.sub(r'^[^:]*:[^:]*:', '', texto)  # Eliminar hasta segundo ':'
    texto = re.sub(r'\.\s+', ' ', texto)  # Reemplazar puntos
    texto = re.sub(r'-+', ' ', texto)  # Reemplazar guiones

    # Expandir contracciones
    texto = contractions.fix(texto)  # i'm -> i am, don't -> do not, etc.

    # Eliminar menciones de personajes
    for personaje in personajes:
        texto = re.sub(r'\b' + re.escape(personaje) +
                       r'\b[.,;!?\s\'"]*', '', texto, flags=re.IGNORECASE)

    # Limpieza final
    texto = re.sub(r'[^a-zA-Z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


# * Análisis de texto
def analizar_texto(texto, titulo):
    """Realiza análisis de frecuencia de palabras en el texto."""
    # Tokenización
    tokens = word_tokenize(texto.lower())

    # Remover stopwords y palabras cortas
    stop_words = set(stopwords.words('english'))
    tokens_filtrados = [palabra for palabra in tokens
                        if palabra not in stop_words
                        and len(palabra) > 1]

    # Análisis de frecuencia
    freq_dist = FreqDist(tokens_filtrados)
    palabras_comunes = freq_dist.most_common(20)

    # Crear DataFrame
    df = pd.DataFrame(palabras_comunes, columns=['Palabra', 'Frecuencia'])

    # Crear gráfica
    fig = px.bar(df,
                 x='Palabra',
                 y='Frecuencia',
                 title=f'20 palabras más frecuentes en {titulo}')

    return df, fig


# *** Ejecución ---------------------------------------------------------------

# * Obtener y procesar guión 1992
aladdin_1992 = extraer_texto_aladdin_1992()
aladdin_1992 = limpiar_texto(aladdin_1992)

# * Obtener y procesar guión 2019
aladdin_2019 = extraer_texto_aladdin_2019()
aladdin_2019 = limpiar_texto(aladdin_2019)

# * Análisis de texto
df_1992, fig_1992 = analizar_texto(aladdin_1992, 'Aladdin 1992')
df_2019, fig_2019 = analizar_texto(aladdin_2019, 'Aladdin 2019')

# * Comparación de frecuencias
df_comparacion = pd.merge(df_1992, df_2019,
                          on='Palabra',
                          how='outer',
                          suffixes=('_1992', '_2019')).fillna(0)

# *** Visualización -----------------------------------------------------------

# Inicializar figura
fig_comparacion = go.Figure()

# Agregar barras para Aladdin 1992
fig_comparacion.add_trace(
    go.Bar(
        x=df_comparacion['Palabra'],
        y=df_comparacion['Frecuencia_1992'],
        name='1992',
        marker_color='blue'
    )
)

# Agregar barras para Aladdin 2019
fig_comparacion.add_trace(
    go.Bar(
        x=df_comparacion['Palabra'],
        y=df_comparacion['Frecuencia_2019'],
        name='2019',
        marker_color='orange'
    )
)

# Configurar el diseño del gráfico
fig_comparacion.update_layout(
    title_text="Comparación de frecuencias de palabras",
    barmode='group',
    xaxis_tickangle=-45,
    height=600,
    showlegend=False
)

# * Comparar vocabulario único
vocab_1992 = set(word_tokenize(aladdin_1992.lower()))
vocab_2019 = set(word_tokenize(aladdin_2019.lower()))

palabras_comunes = vocab_1992.intersection(vocab_2019)
palabras_unicas_1992 = vocab_1992 - vocab_2019
palabras_unicas_2019 = vocab_2019 - vocab_1992

# Generar estadísticas básicas
stats = {
    'Total palabras': (len(word_tokenize(aladdin_1992)), len(word_tokenize(aladdin_2019))),
    'Vocabulario único': (len(vocab_1992), len(vocab_2019)),
    'Palabras compartidas': len(palabras_comunes),
    'Palabras exclusivas': (len(palabras_unicas_1992), len(palabras_unicas_2019))
}

# Obtener las 10 palabras más frecuentes en 1992 y sus frecuencias en 2019
top_10_1992 = df_1992.nlargest(10, 'Frecuencia')
top_10_1992 = top_10_1992.merge(
    df_2019, on='Palabra', how='left', suffixes=('_1992', '_2019')).fillna(0)

# Obtener las 10 palabras más frecuentes en 2019 y sus frecuencias en 1992
top_10_2019 = df_2019.nlargest(10, 'Frecuencia')
top_10_2019 = top_10_2019.merge(
    df_1992, on='Palabra', how='left', suffixes=('_2019', '_1992')).fillna(0)

# Crear una figura con dos subgráficas
fig_combinado = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Top 10 Aladdin 1992", "Top 10 Aladdin 2019")
)

# Agregar barras para 1992 en la primera subgráfica
fig_combinado.add_trace(
    go.Bar(
        x=top_10_1992['Palabra'],
        y=top_10_1992['Frecuencia_1992'],
        name='1992',
        marker_color='blue',
        width=0.6
    ),
    row=1, col=1
)

# Agregar barras comparativas para 2019 dentro de las barras de 1992
fig_combinado.add_trace(
    go.Bar(
        x=top_10_1992['Palabra'],
        y=top_10_1992['Frecuencia_2019'],
        name='2019',
        marker_color='orange',
        width=0.3
    ),
    row=1, col=1
)

# Agregar barras para 2019 en la segunda subgráfica
fig_combinado.add_trace(
    go.Bar(
        x=top_10_2019['Palabra'],
        y=top_10_2019['Frecuencia_2019'],
        name='2019',
        marker_color='orange',
        width=0.6
    ),
    row=1, col=2
)

# Agregar barras comparativas para 1992 dentro de las barras de 2019
fig_combinado.add_trace(
    go.Bar(
        x=top_10_2019['Palabra'],
        y=top_10_2019['Frecuencia_1992'],
        name='1992',
        marker_color='blue',
        width=0.3
    ),
    row=1, col=2
)

# Configurar el diseño del gráfico combinado
fig_combinado.update_layout(
    title_text="<b>Comparación de Frecuencias de Palabras en Aladdin 1992 y 2019</b>",
    barmode='overlay',
    xaxis_tickangle=-45,
    height=600,
    showlegend=False
)

# Cambiar la paleta de colores de las gráficas
fig_combinado.update_traces(
    selector=dict(name='1992'),
    marker_color='#A87676'
)
fig_combinado.update_traces(
    selector=dict(name='2019'),
    marker_color='#3E6D9C'
)
fig_combinado.update_layout(plot_bgcolor='white', paper_bgcolor='white')

# * Guardar resultados en HTML
with open('analisis_aladdin.html', 'w', encoding='utf-8') as f:
    f.write(f'''
    <html>
    <head>
        <title>Análisis comparativo de Aladdin</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}

            .container {{
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}

            h1 {{
                color: #000000;
                text-align: center;
                margin-bottom: 40px;
                font-size: 2.5em;
            }}

            h2 {{
                color: #000000;
                border-bottom: 2px solid #000000;
                padding-bottom: 10px;
                margin-top: 40px;
            }}

            h3 {{
                color: #000000;
                margin-top: 25px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
            }}

            th, td {{
                padding: 12px;
                text-align: center;
                border-bottom: 1px solid #ddd;
            }}

            th {{
                background-color: #000000;
                color: white;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <h1>Análisis Comparativo de Guiones de Aladdin (1992 vs 2019)</h1>

        <h2>Respuestas a las preguntas</h2>
        <h3>¿Se tomará y analizará todo lo que aparece en la página? ¿Que si y
            que no? ¿Por qué?</h3>
        <p>No, por buenas prácticas se ha excluido lo siguiente:</p>
        <ul>
            <li>Acotaciones escénicas (texto entre paréntesis)</li>
            <li>Nombres de personajes</li>
            <li>Signos de puntuación</li>
            <li>Números y caracteres especiales</li>
        </ul>
        <p>Esto se debe a que el enfoque de la actividad es comparar la
            modernización del diálogo, no en las instrucciones técnicas del guion,
            sin mencionar que esto brinda un terreno plano de comparativa pues la
            versión proporcionada de 2019 se limita precisamente a los diálogos.
            También cabe decir que el enfoque del scraping para ambas páginas se
            basó en extraer las partes relevantes acotadas por el JS path.</p>

        <div class="container">
            <h2>Estadísticas Generales</h2>
            <table>
                <thead>
                    <tr>
                        <th>Métrica</th>
                        <th>Aladdin 1992</th>
                        <th>Aladdin 2019</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total de palabras</td>
                        <td>{stats['Total palabras'][0]}</td>
                        <td>{stats['Total palabras'][1]}</td>
                    </tr>
                    <tr>
                        <td>Vocabulario único</td>
                        <td>{stats['Vocabulario único'][0]}</td>
                        <td>{stats['Vocabulario único'][1]}</td>
                    </tr>
                    <tr>
                        <td>Palabras exclusivas</td>
                        <td>{stats['Palabras exclusivas'][0]}</td>
                        <td>{stats['Palabras exclusivas'][1]}</td>
                    </tr>
                    <tr>
                        <td>Palabras compartidas</td>
                        <td colspan="2">{stats['Palabras compartidas']}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h2>Visualizaciones</h2>
        <div class="plot-container">
            <div class="plot-wrapper" id="grafCombinado"></div>
        </div>

        <script>
    ''')

    # Insertar los datos de los gráficos como JSON y crear los gráficos
    f.write(f'var grafCombinado = {fig_combinado.to_json()};')

    f.write('''
        Plotly.newPlot('grafCombinado', grafCombinado.data, grafCombinado.layout);
        </script>
    ''')

    f.write('''
        </div>
        
        <h2>Análisis</h2>
        <h3>Interpretación de los datos</h3>
        <p>Ambos guiones tienen un número parecido de palabras totales (8292 en
            1992 y 8317 en 2019), pero el vocabulario único es un poco más
            amplio en 1992 (1538 contra 1385 en 2019). Esto sugiere que el guion
            original usaba más variedad de palabras, mientras que el de 2019 parece
            ir por un lenguaje más sencillo y directo. Además, el guion de 1992
            tiene más palabras exclusivas (732 frente a 579), lo que indica que
            tenía un toque más único y enfocado en su época, mientras que el de
            2019 comparte muchas más palabras con el original (806 en total),
            manteniendo ese enlace con el pasado.</p>
        <p>Si combinamos esto con las frecuencias de palabras, en 1992 destacan
            términos como "princess" y "wish", que nos recuerdan a un cuento
            clásico de Disney. Por otro lado, en 2019 aparecen más palabras
            como "uh", "like" y "get", que suenan más casuales y modernas. Esto
            demuestra cómo el guion se adaptó para conectar con una audiencia
            actual, usando un lenguaje más natural y cotidiano, pero sin
            perder completamente el estilo que hizo memorable a la película
            original. Es como una actualización que intenta balancear lo nuevo
            con lo nostálgico.</p>
    </body>
    </html>
    ''')
