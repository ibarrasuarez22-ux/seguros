import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# 📥 Ruta y archivo
input_path = "data/dataset_enriquecido.pkl"
output_model = "modelos/recomendador.pkl"

# 🗂️ Cargar datos enriquecidos
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Archivo no encontrado: {input_path}")

df = pd.read_pickle(input_path)

# 🧼 Preprocesamiento
df = df.dropna(subset=["Producto_Recomendado"])
df["Producto_Recomendado"] = df["Producto_Recomendado"].astype(str)

# 🔀 Variables
X = df.drop(columns=["Producto_Recomendado", "ID_Cliente"], errors="ignore")
y = df["Producto_Recomendado"]

# 🧪 Separar numéricas y categóricas
numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 🔧 Pipeline de preprocesamiento
preprocesador = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
])

# 🌲 Modelo con red neuronal
modelo = Pipeline(steps=[
    ("pre", preprocesador),
    ("clf", MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42))
])

# 🧪 Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
modelo.fit(X_train, y_train)

# 📊 Validación rápida
y_pred = modelo.predict(X_test)
print("🧠 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# 💾 Guardar modelo entrenado
joblib.dump(modelo, output_model)
print(f"✅ Modelo guardado en: {output_model}")