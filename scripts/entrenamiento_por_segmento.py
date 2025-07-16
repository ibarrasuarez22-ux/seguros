import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 📥 Cargar datos
df = pd.read_pickle("data/dataset_enriquecido.pkl")

# 🎯 Definir segmentos etarios
def clasificar_edad(edad):
    if edad < 30:
        return "joven"
    elif 30 <= edad <= 59:
        return "adulto"
    else:
        return "adulto_mayor"

df = df.dropna(subset=["Edad", "Producto_Recomendado"])
df["Segmento_Edad"] = df["Edad"].apply(clasificar_edad)

# 🧠 Función para entrenar por segmento
def entrenar_por_segmento(nombre_segmento, datos_segmentados):
    X = datos_segmentados.drop(columns=["ID_Cliente", "Producto_Recomendado", "Segmento_Edad"], errors="ignore")
    y = datos_segmentados["Producto_Recomendado"].astype(str)

    numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categoricas = X.select_dtypes(include=["object"]).columns.tolist()

    preprocesador = ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
    ])

    modelo = Pipeline([
        ("pre", preprocesador),
        ("clf", MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=600, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)

    print(f"🧠 Modelo entrenado para segmento: {nombre_segmento}")
    print(classification_report(y_test, modelo.predict(X_test)))

    ruta = f"modelos/recomendador_{nombre_segmento}.pkl"
    joblib.dump(modelo, ruta)
    print(f"✅ Modelo guardado en: {ruta}")

# 🔄 Entrenar por cada grupo
for segmento in df["Segmento_Edad"].unique():
    datos = df[df["Segmento_Edad"] == segmento]
    if len(datos) >= 30:  # mínimo de datos para entrenamiento fiable
        entrenar_por_segmento(segmento, datos)
    else:
        print(f"⚠️ Segmento {segmento} omitido por falta de datos.")