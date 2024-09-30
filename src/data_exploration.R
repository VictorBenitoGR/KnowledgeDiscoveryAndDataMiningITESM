# * Minería de Datos (ITESM) - Exploración de datos

# *** Bibliotecas --------------------------------------------------------------

library(readxl)
library(dplyr)
library(knitr)
library(ggplot2)

# *** Importación de datos ----------------------------------------------------

resultados <- read_excel("data/Centros Comerciales EM19 Tr5.xlsx")

View(resultados)

# *** Exploración de datos ----------------------------------------------------

# Guardar nombre de las columnas en el portapapeles
writeClipboard(kable(colnames(resultados), format = "markdown"))

# |x                     |
# |:---------------------|
# |ID                    |
# |Duration__in_seconds_ |
# |Finished              |
# |I1                    |
# |F1                    |
# |P1_1                  |
# |P1_2                  |
# |P1_3                  |
# |P1_4                  |
# |P1_5                  |
# |P1_6                  |
# |P1_7                  |
# |P1_8                  |
# |P1_9                  |
# |P1_10                 |
# |P2_1                  |
# |P2_2                  |
# |P2_3                  |
# |P2_4                  |
# |P2_5                  |
# |P2_6                  |
# |P2_7                  |
# |P2_8                  |
# |P2_9                  |
# |P2_10                 |
# |P3                    |
# |P4_1                  |
# |P4_2                  |
# |P4_3                  |
# |P4_4                  |
# |P4_5                  |
# |P4_6                  |
# |P4_7                  |
# |P4_8                  |
# |P4_9                  |
# |P4_10                 |
# |P5                    |
# |P6_1_1_1              |
# |P6_1_2_1              |
# |P6_1_3_1              |
# |P6_1_4_1              |
# |P7                    |
# |P8                    |
# |P9                    |
# |P10                   |
# |P11_1                 |
# |P11_2                 |
# |P11_3                 |
# |P11_4                 |
# |P11_5                 |
# |P11_6                 |
# |P11_7                 |
# |P11_8                 |
# |P11_9                 |
# |P11_10                |
# |P12                   |
# |P13_1_1_1             |
# |P13_1_2_1             |
# |P13_1_3_1             |
# |P13_1_4_1             |
# |P14                   |
# |P15                   |
# |P16                   |
# |P17                   |
# |P18_1                 |
# |P18_2                 |
# |P18_3                 |
# |P18_4                 |
# |P18_5                 |
# |P18_6                 |
# |P18_7                 |
# |P18_8                 |
# |P18_9                 |
# |P18_10                |
# |P19                   |
# |P20_1_1_1             |
# |P20_1_2_1             |
# |P20_1_3_1             |
# |P20_1_4_1             |
# |P21                   |
# |P22                   |
# |P23                   |
# |D1                    |
# |D2                    |
# |D3                    |
# |P8_Visita             |
# |P7_Satisfacción       |
# |P15_Visita            |
# |P14_Satisfacción      |
# |P22_Visita            |
# |P21_Satisfacción      |

# * Definir la clase de las columnas
#' Todas las columnas excepto ID y Duración_en_segundos_ deberían ser un factor
#' ID debería ser carácter. Duración_en_segundos_ debería ser numérico.
#' 
#' @param data: Un data frame que contiene los datos a procesar.
#' @return Un data frame con las columnas convertidas a sus respectivas clases.
definir_clase <- function(data) {
  for (col in colnames(data)) {
    if (col == "ID") {
      data[[col]] <- as.character(data[[col]])
    } else if (col == "Duración_en_segundos_") {
      data[[col]] <- as.numeric(data[[col]])
    } else {
      data[[col]] <- as.factor(data[[col]])
    }
  }
  return(data)
}

# * Remover respuestas "0" o NA en "l1"
#' Remueve las respuestas "0" o NA en una columna específica de un data frame.
#'
#' @param data: Un data frame que contiene los datos a procesar.
#' @param column: El nombre de la columna en el data frame que se desea procesar
#' @return Un data frame con las respuestas "0" o NA removidas.
remover_no_deseados <- function(data, column) {
  data <- data[data[[column]] != 0, ]
  data <- data[!is.na(data[[column]]), ]
  return(data)
}

resultados <- definir_clase(resultados)
resultados <- remover_no_deseados(resultados, "l1")

View(resultados)

# *** Visualización preliminar -------------------------------------------------

# * Remover outliers de Duration__in_seconds_
#' Remueve los outliers de una columna específica de un data frame.
#'
#' @param data: Un data frame que contiene los datos a procesar.
#' @param column: El nombre de la columna en el data frame que se desea procesar
#' @return Un data frame con los outliers removidos.
remover_outliers <- function(data, column) {
  q1 <- quantile(data[[column]], 0.25)
  q3 <- quantile(data[[column]], 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  data <- data[data[[column]] >= lower_bound & data[[column]] <= upper_bound, ]
  return(data)
}

# Aplicar la función para remover outliers
resultados_sin_outliers <- remover_outliers(resultados, "Duration__in_seconds_")

# Paleta de colores Okabe-Ito
okabe_ito <- c(
  "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

# Boxplot de Duration__in_seconds_
boxplot_segundos <- ggplot(
  resultados_sin_outliers,
  aes(x = "", y = Duration__in_seconds_)
) +
  geom_boxplot(outlier.alpha = 0.5, fill = okabe_ito[1]) +
  scale_y_log10() +
  labs(
    title = "Duración en segundos",
    x = NULL,
    y = NULL
  ) +
  theme_bw() +
  theme(
    title = element_text(face = "bold", color = "black"),
    axis.text = element_text(color = "black")
  )

# Exportar gráfico
ggsave(
  "./assets/boxplot_segundos.png",
  plot = boxplot_segundos,
  width = 4,
  height = 6,
  dpi = 600
)
