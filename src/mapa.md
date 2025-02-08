# Minería de Datos: Ecosistema Integrado
## Modelos Estadísticos
### Modelos Dicotómicos
#### Logit
- Basado en la función logística.
- Útil para distribuciones no normales.
- **Aplicación:** Probabilidad de que un cliente compre un producto.

#### Probit
- Basado en la función acumulativa normal.
- Ideal para precisar umbrales.
- **Aplicación:** Aprobación de un crédito.

### Modelos de Conglomerados
#### Clustering Jerárquico
- Agrupa datos en niveles anidados.
- **Ventajas:** 
  - Útil para estructuras jerárquicas.
  - Mejor para conjuntos pequeños.
- **Visualización:** Dendrogramas.

#### Clustering Particional
##### K-Means
- Divide datos en grupos homogéneos basados en centroides.
- **Aplicación:** Segmentación de clientes.
##### K-Medoids
- Similar a K-Means pero robusto ante outliers.
- **Aplicación:** Datos con métricas no euclídeas.

## Técnicas de Análisis
### Correlación
#### Pearson
- Para relaciones lineales con datos normales.
- **Ejemplo:** Relación entre ingresos y gastos.
#### Spearman
- Para datos no lineales o sin distribución normal.
- **Ejemplo:** Relación entre antigüedad y satisfacción laboral.

### ANOVA
#### ANOVA Un Factor
- Compara medias de un solo factor.
- **Ejemplo:** Impacto de campañas en ventas.
#### ANOVA Dos Factores
- Evalúa interacción entre variables.
- **Ejemplo:** Interacción entre región y segmento.

## Transformación de Datos
### Escalado
#### Min-Max
- Escala valores entre 0 y 1.
- **Aplicación:** Preservar distribución original.
#### Z-Score
- Ajusta datos con media 0 y desviación 1.
- **Aplicación:** Comparar variables de distintas unidades.

### Reducción de Dimensionalidad
#### PCA (Análisis de Componentes Principales)
- Reduce ruido al conservar la varianza.
- **Aplicación:** Simplificar conjuntos con muchas variables correlacionadas.

## Infraestructura de Datos
### Bases SQL
#### MySQL
- Optimizada para transacciones relacionales.
- **Ventajas:**
  - Consistencia y confiabilidad (ACID).
  - Ideal para datos estructurados.

### Bases NoSQL
#### MongoDB
- Maneja datos no estructurados como JSON.
- **Ventajas:**
  - Escalabilidad horizontal.
  - Flexibilidad en formatos de datos.

## Modelado Predictivo
### Árboles de Decisión
#### CART
- Divide datos con reglas interpretables.
- **Ventajas:**
  - Fácil de entender.
  - Útil para decisiones rápidas.
- **Ejemplo:** Clasificación de clientes en categorías de riesgo.

### Redes Neuronales
#### Redes Multicapa
- Capturan patrones no lineales complejos.
- **Aplicaciones:**
  - Predicción de series temporales.
  - Análisis de imágenes y texto.

## Procesamiento de Lenguaje Natural (NLP)
### Técnicas Clave
- **Tokenización:** Dividir texto en palabras o frases.
- **Embeddings:** Representar palabras en vectores.
### Aplicaciones
- Análisis de sentimientos.
- Clasificación automática de textos.

## Secuencia de Pasos para un Modelo Óptimo
1. Dividir datos en entrenamiento y prueba.
2. Realizar análisis exploratorio.
3. Normalizar y escalar variables.
4. Identificar correlaciones relevantes.
5. Probar supuestos estadísticos (ANOVA, estacionalidad).
6. Entrenar el modelo seleccionado.
7. Evaluar residuos y métricas de desempeño.
8. Ajustar hiperparámetros según resultados.
