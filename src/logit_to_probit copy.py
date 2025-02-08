
# *** Importaciones -----------------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix  # Ev. y matriz
from sklearn.model_selection import train_test_split  # División de datos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.preprocessing import StandardScaler  # Escalado opcional
import matplotlib.pyplot as plt  # Visualización de datos
import pandas as pd  # Manipulación y análisis de datos
import seaborn as sns  # Visualización de datos

# *** Carga de datos ----------------------------------------------------------

# Cargar los conjuntos de datos de training y test
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# Eliminar la columna "Loan_ID" ya que no aporta información útil para la predicción
train = train.drop("Loan_ID", axis=1)
test = test.drop("Loan_ID", axis=1)

# Convertir "Loan_Status" de categórica a numérica: "Y" se convierte en 1 y "N" en 0
train["Loan_Status"] = train["Loan_Status"].map({"Y": 1, "N": 0})
train["Loan_Status"].unique()


# Rellenar valores faltantes en columnas numéricas
def fill_missing_numeric(df, columns):
    """
    Rellena los valores faltantes en las columnas numéricas del DataFrame.

    Parámetros:
    df : DataFrame
        El DataFrame en el que se rellenarán los valores faltantes.
    columns : list
        Lista de nombres de columnas en las que se buscarán valores faltantes.
    """
    for col in columns:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)


# Rellenar valores faltantes en columnas categóricas
def fill_missing_categorical(df, columns):
    """
    Rellena los valores faltantes en las columnas categóricas del DataFrame.

    Parámetros:
    df : DataFrame
        El DataFrame en el que se rellenarán los valores faltantes.
    columns : list
        Lista de nombres de columnas en las que se buscarán valores faltantes.
    """
    for col in columns:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)


# Columnas numéricas y categóricas
numeric_cols = ["ApplicantIncome", "CoapplicantIncome",
                "LoanAmount", "Loan_Amount_Term", "Credit_History"]
categorical_cols = ["Gender", "Married", "Dependents",
                    "Education", "Self_Employed", "Property_Area"]

# Rellenar valores faltantes en el conjunto de entrenamiento
fill_missing_numeric(train, numeric_cols)
fill_missing_categorical(train, categorical_cols)

# Rellenar valores faltantes en el conjunto de prueba
fill_missing_numeric(test, numeric_cols)
fill_missing_categorical(test, categorical_cols)

# Combina los conjuntos de entrenamiento y prueba para asegurar consistencia en las columnas dummy
combined = pd.concat(
    [train[categorical_cols], test[categorical_cols]], axis=0, ignore_index=True)

# One-Hot Encoding
combined_encoded = pd.get_dummies(combined, drop_first=True)

# Separar de nuevo train y test
train_encoded = combined_encoded[:len(train)]
test_encoded = combined_encoded[len(train):]

# Incluir las características numéricas y las variables dummy categóricas
X = pd.concat([train[numeric_cols], train_encoded], axis=1)
y = train["Loan_Status"]

# Dividir el conjunto de datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=16
)

# Escalar las características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Verificar tipos de datos antes de ajustar el modelo
print(X_train.dtypes)
print(X_val.dtypes)

# Ajustar el modelo LDA con las características escaladas
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled, y_train)

# Realizar predicciones en el conjunto de validación
y_pred = lda_model.predict(X_val_scaled)
# Probabilidades para la clase positiva
y_pred_prob = lda_model.predict_proba(X_val_scaled)[:, 1]

# Evaluar el modelo utilizando la matriz de confusión
cnf_matrix = confusion_matrix(y_val, y_pred)
print(cnf_matrix)

# Visualizar la matriz de confusión
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
plt.title("Matriz de Confusión")
plt.ylabel("Etiqueta Real")
plt.xlabel("Etiqueta Predicha")
plt.show()

# Informe de clasificación
target_names = ["Rechazada", "Aprobada"]
print(classification_report(y_val, y_pred, target_names=target_names))
