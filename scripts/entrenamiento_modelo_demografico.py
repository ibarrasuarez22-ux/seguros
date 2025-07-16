import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 🗂️ Cargar datos enriquecidos
df = pd.read_pickle("data/dataset_enriquecido.pkl")

# 🔍 Validación mínima
df = df.dropna(subset=["Producto_Recomendado"])
df["Producto_Recomendado"] = df["Producto_Recomendado"].astype(str)

# 🔀 Variables
X = df.drop(columns=["ID_Cliente", "Producto_Recomendado"], errors="ignore")
y = df["Producto_Recomendado"]

# ⚙️ Identificación de columnas
numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categoricas = X.select_dtypes(include=["object"]).columns.tolist()

# 🔧 Pipeline
preprocesador = ColumnTransformer([
    ("num", StandardScaler(), numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
])

modelo = Pipeline([
    ("pre", preprocesador),
    ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42))
])

# 🔬 Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
modelo.fit(X_train, y_train)

# 💾 Guardar modelo
joblib.dump(modelo, "modelos/recomendador_demografico.pkl")