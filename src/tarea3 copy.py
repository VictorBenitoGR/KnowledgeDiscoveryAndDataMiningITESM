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
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from urllib.parse import urljoin

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# URLs de los guiones
URL_ALADDIN_1992 = "https://imsdb.com/scripts/Aladdin.html"
URL_ALADDIN_2019 = "https://www.scripts.com/script/aladdin_2019_26877"

def obtener_texto_pagina(url):
    """Obtiene el texto de una URL dada."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"Error al obtener la página {url}: {e}")
        return None

def limpiar_texto_1992(soup):
    """Limpia y extrae el texto del guión de 1992."""
    if not soup:
        return ""
    script_text = soup.find('pre')
    if script_text:
        texto = script_text.get_text()
        # Eliminar acotaciones entre paréntesis
        texto = re.sub(r'\([^)]*\)', '', texto)
        # Eliminar espacios múltiples y líneas en blanco extras
        texto = re.sub(r'\n\s*\n', '\n', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()
    return ""

def obtener_urls_2019(soup):
    """Obtiene las URLs de todas las páginas del guión 2019."""
    if not soup:
        return []
    
    urls = []
    base_url = "https://www.scripts.com"
    
    # Buscar enlaces de paginación
    pagination = soup.find_all('a', class_='page-link')
    for link in pagination:
        href = link.get('href')
        if href and 'aladdin_2019' in href:
            full_url = urljoin(base_url, href)
            if full_url not in urls:
                urls.append(full_url)
    
    return urls

def limpiar_texto_2019(soup):
    """Limpia y extrae el texto del guión de 2019."""
    if not soup:
        return ""
    
    # Buscar el contenido del guión
    script_content = soup.find('div', class_='script-content')
    if script_content:
        texto = script_content.get_text()
        # Eliminar acotaciones entre paréntesis
        texto = re.sub(r'\([^)]*\)', '', texto)
        # Eliminar espacios múltiples y líneas en blanco extras
        texto = re.sub(r'\n\s*\n', '\n', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()
    return ""

def procesar_texto(texto):
    """Procesa el texto: tokeniza, elimina stopwords y caracteres especiales."""
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Tokenización
    tokens = word_tokenize(texto)
    
    # Obtener stopwords en inglés
    stop_words = set(stopwords.words('english'))
    
    # Filtrar tokens
    tokens = [token for token in tokens 
             if token.isalnum() and  # Solo caracteres alfanuméricos
             token not in stop_words and  # Eliminar stopwords
             len(token) > 1]  # Eliminar tokens de un solo carácter
    
    return tokens

def analizar_frecuencias(tokens):
    """Analiza la frecuencia de palabras en los tokens."""
    fdist = FreqDist(tokens)
    return pd.DataFrame(fdist.most_common(30), columns=['Palabra', 'Frecuencia'])

def generar_analisis():
    """Función principal que coordina todo el análisis."""
    # Obtener y procesar guión 1992
    print("Obteniendo guión de 1992...")
    soup_1992 = obtener_texto_pagina(URL_ALADDIN_1992)
    texto_1992 = limpiar_texto_1992(soup_1992)
    tokens_1992 = procesar_texto(texto_1992)
    
    # Obtener y procesar guión 2019
    print("Obteniendo guión de 2019...")
    texto_2019 = ""
    soup_2019 = obtener_texto_pagina(URL_ALADDIN_2019)
    texto_2019 += limpiar_texto_2019(soup_2019)
    
    # Obtener páginas adicionales
    urls_2019 = obtener_urls_2019(soup_2019)
    for url in urls_2019:
        print(f"Procesando página adicional: {url}")
        soup = obtener_texto_pagina(url)
        texto_2019 += " " + limpiar_texto_2019(soup)
    
    tokens_2019 = procesar_texto(texto_2019)
    
    # Análisis de frecuencias
    freq_1992 = analizar_frecuencias(tokens_1992)
    freq_2019 = analizar_frecuencias(tokens_2019)
    
    # Guardar resultados
    freq_1992.to_csv('frecuencias_1992.csv', index=False)
    freq_2019.to_csv('frecuencias_2019.csv', index=False)
    
    # Generar reporte HTML
    generar_reporte_html(freq_1992, freq_2019, len(tokens_1992), len(tokens_2019))

def generar_reporte_html(freq_1992, freq_2019, total_tokens_1992, total_tokens_2019):
    """Genera un reporte HTML con los resultados del análisis."""
    html_content = f"""
    <html>
    <head>
        <title>Análisis Comparativo de Guiones de Aladdin</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Análisis Comparativo de Guiones de Aladdin</h1>
        
        <h2>Estadísticas Generales</h2>
        <p>Total de palabras (1992): {total_tokens_1992}</p>
        <p>Total de palabras (2019): {total_tokens_2019}</p>
        
        <h2>Palabras más frecuentes en Aladdin 1992</h2>
        {freq_1992.to_html()}
        
        <h2>Palabras más frecuentes en Aladdin 2019</h2>
        {freq_2019.to_html()}
        
        <h2>Notas sobre el análisis</h2>
        <p>Para este análisis:</p>
        <ul>
            <li>Se eliminaron las stopwords (palabras comunes como "the", "a", "an", etc.)</li>
            <li>Se eliminaron los caracteres especiales y números</li>
            <li>Se convirtió todo el texto a minúsculas</li>
            <li>Se eliminaron las acotaciones entre paréntesis</li>
        </ul>
        
        <h2>Respuesta a las preguntas</h2>
        <h3>¿Se tomará y analizará Todo lo que aparece en la página?</h3>
        <p>No, se han excluido específicamente:</p>
        <ul>
            <li>Acotaciones escénicas (texto entre paréntesis)</li>
            <li>Elementos de navegación y estructura de la página</li>
            <li>Stopwords y caracteres especiales</li>
            <li>Números y tokens de un solo carácter</li>
        </ul>
        <p>Esto se hace para concentrarnos en el contenido significativo del diálogo y la narrativa.</p>
    </body>
    </html>
    """
    
    with open('analisis_aladdin.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    generar_analisis()

