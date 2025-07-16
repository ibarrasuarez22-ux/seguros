import pandas as pd
import numpy as np
from datetime import datetime

def generar_dataset_enriquecido():
    # Cargar archivos base
    clientes = pd.read_csv("data/clientes.csv")
    polizas = pd.read_csv("data/polizas.csv")
    reclamos = pd.read_csv("data/reclamaciones.csv")

    # Validar columnas clave
    columnas_necesarias = ["Razon_Social", "Delegacion_Municipio", "Giro_Empresa", "Ingresos_Anuales", "Email_Contacto", "Telefono"]
    for col in columnas_necesarias:
        if col not in clientes.columns:
            clientes[col] = "Información no disponible"

    resumen_polizas = polizas.groupby("ID_Cliente").agg({
        "Prima_Anual": "sum",
        "ID_Seguro": "count"
    }).rename(columns={"Prima_Anual": "Prima_Total", "ID_Seguro": "Num_Polizas"})

    resumen_reclamos = reclamos.groupby("ID_Cliente").agg({
        "ID_Reclamacion": "count",
        "Monto_Reclamacion": "sum"
    }).rename(columns={"ID_Reclamacion": "Num_Reclamaciones", "Monto_Reclamacion": "Monto_Reclamado"})

    df = clientes.set_index("ID_Cliente")[columnas_necesarias].join(resumen_polizas).join(resumen_reclamos)
    df = df.fillna({"Num_Polizas": 0, "Prima_Total": 0, "Num_Reclamaciones": 0, "Monto_Reclamado": 0})

    df["Probabilidad_Conversion"] = np.clip(70 + df["Prima_Total"] * 0.0005 - df["Num_Reclamaciones"] * 5, 10, 95).round(1)
    df["Indice_Riesgo"] = pd.cut(df["Monto_Reclamado"], bins=[0,10000,30000,60000,np.inf], labels=["Bajo","Medio","Alto","Crítico"])
    df["Segmento"] = pd.cut(df["Probabilidad_Conversion"], bins=[0,40,70,100], labels=["Bajo", "Medio", "Alto"])
    df["Producto_Recomendado"] = np.where(df["Segmento"] == "Alto", "Seguro Empresarial", "Seguro Mixto")
    df["Accion_Sugerida"] = np.where(df["Indice_Riesgo"] == "Crítico", "Visita inmediata", "Enviar propuesta")
    df["Fecha_Última_Actualización"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df.reset_index().to_pickle("data/dataset_enriquecido.pkl")