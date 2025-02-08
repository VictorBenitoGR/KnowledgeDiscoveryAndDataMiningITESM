# ./src/tarea3.py

# *** Instrucciones -----------------------------------------------------------

# La actividad corresponde en comparar dos guiones de película, uno es de
# Aladdin de 1992 y el otro es de Aladdin de 2019. El primero es una versión
# animada, por lo cual la tarea es analizar que tanto se ha modernizado el nuevo
# guión y si sigue siendo fiel a su antecesor.

# El primer guión lo encuentran aquí:
# https://imsdb.com/scripts/Aladdin.html

# El de la película de 2019 (hay que picarle "Next" para conseguir las otras
# partes, no está todo en una sola página):
# https://www.scripts.com/script/aladdin_2019_26877

# El documento que estás a punto de crear primero tendrá el análisis de texto y
# al final tendrá la descripción de lo que hiciste.

# 1. Gran parte del análisis de texto es buscarlo, limpiarlo y estructurarlo en
#    uno o varios archivos o corpus. Esta es el punto donde creas el corpus (33
#    puntos)

#    a) Además de limpiar contesta a esta pregunta (al final de tu archivo html)
#    ¿Se tomará y analizará Todo lo que aparece en la página? ¿Que si y que no?
#    ¿Por qué?

# 2. Ahora procederás a realizar un análisis como el visto en clase, considera
#    agregar stopwords, y realiza una comparación de frecuencia . Describe lo
#    realizado al final del html en el punto

# *** Importaciones -----------------------------------------------------------

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from urllib.parse import urljoin
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
from matplotlib_venn import venn2
import io
import base64

# Descargar recursos necesarios de NLTK
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
    # Normalización básica
    texto = re.sub(r'\s+', ' ', texto).strip()
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
    texto = re.sub(r'[^a-zA-Z\s]', ' ', texto)  # Solo letras y espacios
    texto = re.sub(r'\s+', ' ', texto).strip()  # Normalizar espacios

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

    return df, fig  # Devolver tanto el DataFrame como la figura


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

# Ahora sí podemos hacer el merge
df_comparacion = pd.merge(df_1992, df_2019,
                          on='Palabra',
                          how='outer',
                          suffixes=('_1992', '_2019')).fillna(0)

# Crear el gráfico de comparación
fig_comparacion = make_subplots(rows=1, cols=2,
                                subplot_titles=('Aladdin 1992', 'Aladdin 2019'))

fig_comparacion.add_trace(
    go.Bar(x=df_1992['Palabra'], y=df_1992['Frecuencia'], name='1992'),
    row=1, col=1
)

fig_comparacion.add_trace(
    go.Bar(x=df_2019['Palabra'], y=df_2019['Frecuencia'], name='2019'),
    row=1, col=2
)

fig_comparacion.update_layout(
    title_text="Comparación de frecuencias de palabras",
    height=600,
    showlegend=True,
    xaxis_tickangle=-45,
    xaxis2_tickangle=-45
)

# * Comparar vocabulario único
vocab_1992 = set(word_tokenize(aladdin_1992.lower()))
vocab_2019 = set(word_tokenize(aladdin_2019.lower()))

palabras_comunes = vocab_1992.intersection(vocab_2019)
palabras_unicas_1992 = vocab_1992 - vocab_2019
palabras_unicas_2019 = vocab_2019 - vocab_1992

# Crear el diagrama de Venn con matplotlib
plt.figure(figsize=(12, 8))
venn2([set(vocab_1992), set(vocab_2019)],
      set_labels=('Aladdin 1992', 'Aladdin 2019'),
      set_colors=('#3498db', '#2ecc71'))

# Agregar título
plt.title('Comparación de Vocabulario entre Versiones',
          pad=20, size=16, fontweight='bold')

# Agregar texto con estadísticas
stats_text = (
    f"Estadísticas Generales\n\n"
    f"Aladdin 1992:\n"
    f"• Total palabras: {len(word_tokenize(aladdin_1992)):,}\n"
    f"• Vocabulario único: {len(vocab_1992):,}\n\n"
    f"Aladdin 2019:\n"
    f"• Total palabras: {len(word_tokenize(aladdin_2019)):,}\n"
    f"• Vocabulario único: {len(vocab_2019):,}\n\n"
    f"Vocabulario Compartido:\n"
    f"• Palabras comunes: {len(palabras_comunes):,}\n"
    f"• Porcentaje del total: {
        (len(palabras_comunes) / max(len(vocab_1992), len(vocab_2019))) * 100:.1f}%"
)

plt.figtext(0.86, 0.5, stats_text,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=12,
            verticalalignment='center')

# Ajustar los márgenes para que quepa todo
plt.subplots_adjust(right=0.75)

# Convertir el gráfico a base64 para incluirlo en el HTML
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
buf.seek(0)
venn_img = base64.b64encode(buf.getvalue()).decode('utf-8')
plt.close()

# * Generar estadísticas básicas
stats = {
    'Total palabras 1992': len(word_tokenize(aladdin_1992)),
    'Total palabras 2019': len(word_tokenize(aladdin_2019)),
    'Vocabulario único 1992': len(vocab_1992),
    'Vocabulario único 2019': len(vocab_2019),
    'Palabras compartidas': len(palabras_comunes),
    'Palabras exclusivas 1992': len(palabras_unicas_1992),
    'Palabras exclusivas 2019': len(palabras_unicas_2019)
}

# * Guardar resultados en HTML
with open('analisis_aladdin.html', 'w', encoding='utf-8') as f:
    f.write('''
    <html>
    <head>
        <title>Análisis Comparativo de Guiones de Aladdin</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            
            .container {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 40px;
                font-size: 2.5em;
            }
            
            h2 {
                color: #3498db;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-top: 40px;
            }
            
            h3 {
                color: #2980b9;
                margin-top: 25px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .stat-card {
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .stat-card strong {
                color: #2c3e50;
                font-size: 1.1em;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
            }
            
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            
            th {
                background-color: #3498db;
                color: white;
            }
            
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            
            tr:hover {
                background-color: #f5f5f5;
            }
            
            .plot-container {
                margin: 30px 0;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .plot-wrapper {
                width: 100%;
                height: 500px;
            }
            
            .conclusions {
                background: #fff;
                padding: 20px;
                border-left: 4px solid #3498db;
                margin: 20px 0;
            }
            
            ul {
                list-style-type: none;
                padding-left: 0;
            }
            
            ul li {
                padding: 8px 0;
                padding-left: 20px;
                position: relative;
            }
            
            ul li:before {
                content: "•";
                color: #3498db;
                font-weight: bold;
                position: absolute;
                left: 0;
            }
        </style>
    </head>
    <body>
        <h1>Análisis Comparativo de Guiones de Aladdin (1992 vs 2019)</h1>
        
        <div class="container" style="display: flex; justify-content: center; align-items: center;">
            <img src="data:image/png;base64,''' + venn_img + '''" 
                 style="max-width: 100%; height: auto;" 
                 alt="Diagrama de Venn de vocabulario">
        </div>
        
        <h2>Visualizaciones</h2>
        <div class="plot-container">
            <h3>Frecuencias de Palabras - Aladdin 1992</h3>
            <div class="plot-wrapper" id="graf1992"></div>
        </div>
        
        <div class="plot-container">
            <h3>Frecuencias de Palabras - Aladdin 2019</h3>
            <div class="plot-wrapper" id="graf2019"></div>
        </div>
        
        <div class="plot-container">
            <h3>Comparación de Frecuencias</h3>
            <div class="plot-wrapper" id="grafComparacion"></div>
        </div>

        <script>
    ''')

    # Insertar los datos de los gráficos como JSON y crear los gráficos
    f.write(f'var graf1992 = {fig_1992.to_json()};')
    f.write(f'var graf2019 = {fig_2019.to_json()};')
    f.write(f'var grafComparacion = {fig_comparacion.to_json()};')

    f.write('''
        Plotly.newPlot('graf1992', graf1992.data, graf1992.layout);
        Plotly.newPlot('graf2019', graf2019.data, graf2019.layout);
        Plotly.newPlot('grafComparacion', grafComparacion.data, grafComparacion.layout);
        </script>
    ''')

    # Continuar con el resto del HTML
    f.write('''
        <div class="container">
            <h2>20 Palabras más frecuentes</h2>
            <h3>Aladdin 1992</h3>
    ''')
    f.write(df_1992.to_html(classes='dataframe'))
    f.write('''
            <h3>Aladdin 2019</h3>
    ''')
    f.write(df_2019.to_html(classes='dataframe'))
    f.write('''
        </div>
        
        <h2>Respuestas a las preguntas</h2>
        <h3>¿Se tomará y analizará todo lo que aparece en la página?</h3>
        <p>No, se han excluido los siguientes elementos:</p>
        <ul>
            <li>Acotaciones escénicas (texto entre paréntesis)</li>
            <li>Nombres de personajes</li>
            <li>Signos de puntuación</li>
            <li>Números y caracteres especiales</li>
        </ul>
        <p>Esto se debe a que queremos enfocarnos en el contenido del diálogo y la narrativa,
        no en las instrucciones técnicas del guion.</p>
        
        <h3>Análisis de modernización y fidelidad</h3>
        <p>[Aquí puedes agregar tus conclusiones sobre la modernización del guion y su fidelidad
        basándote en las estadísticas y palabras frecuentes]</p>
    </body>
    </html>
    ''')
