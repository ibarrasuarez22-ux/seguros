import streamlit as st
import pandas as pd
import os
import joblib
import pydeck as pdk

from modulos.panel_cliente import mostrar_panel_cliente
from modulos.estadisticas_demograficas import mostrar_estadisticas_demograficas

# 🧭 Configuración general
st.set_page_config(page_title="Agente de Seguros Inteligente", layout="wide")
st.title("📊 Inteligencia Comercial del Agente de Seguros")

# 📦 Funciones de carga de datos
@st.cache_data
def cargar_datos():
    polizas = pd.read_pickle('data/polizas.pkl')
    reclamos = pd.read_pickle('data/reclamaciones.pkl')
    clientes = pd.read_pickle('data/clientes.pkl')
    return polizas, reclamos, clientes

@st.cache_data
def cargar_enriquecido():
    path = 'data/dataset_enriquecido.pkl'
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        return pd.DataFrame()

# 🔃 Botón para actualizar dataset enriquecido
st.subheader("🔄 Actualizar dataset enriquecido")
if st.button("🔃 Regenerar información consolidada"):
    try:
        from scripts.fusionador import generar_dataset_enriquecido
        generar_dataset_enriquecido()
        st.success("✅ Dataset enriquecido actualizado con éxito.")
    except Exception as e:
        st.error(f"❌ Error al actualizar: {e}")

# 📥 Cargar datos
polizas, reclamos, clientes = cargar_datos()
enriched = cargar_enriquecido()

# 🧭 Organización por pestañas
tabs = st.tabs([
    "📊 KPIs Generales",
    "💡 Recomendaciones IA",
    "📋 Panel Estratégico del Cliente",
    "📊 Estadísticas Demográficas"
])

with tabs[0]:
    st.header("📊 KPIs Generales")
    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes únicos", polizas['ID_Cliente'].nunique())
    col2.metric("Total de Pólizas", len(polizas))
    col3.metric("Prima Total", f"${polizas['Prima_Anual'].sum():,.2f}")

    st.subheader("📘 Distribución por tipo de seguro")
    if 'Tipo_Seguro' in polizas.columns:
        tipo_data = polizas['Tipo_Seguro'].value_counts().reset_index()
        tipo_data.columns = ['Tipo de Seguro', 'Cantidad']
        st.bar_chart(tipo_data.set_index('Tipo de Seguro'))
    else:
        st.warning("La columna 'Tipo_Seguro' no está presente en polizas.pkl.")

    st.subheader("🗺️ Mapa de calor por delegación")
    if os.path.exists("data/mapa_calor.csv"):
        zonas = pd.read_csv("data/mapa_calor.csv")
        zonas = zonas.rename(columns={"Latitud": "lat", "Longitud": "lon"})

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=zonas,
            get_position='[lon, lat]',
            get_radius=5000,
            get_color='[255, 140 - Tasa_Siniestro * 100, 0]',
            pickable=True,
            auto_highlight=True
        )

        view_state = pdk.ViewState(latitude=19.4, longitude=-99.1, zoom=10)

        tooltip = {
            "html": "<b>{Delegacion_Municipio}</b><br/>Tasa de siniestro: <b>{Tasa_Siniestro}</b>",
            "style": {"backgroundColor": "black", "color": "white"}
        }

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        ))
    else:
        st.warning("El archivo 'mapa_calor.csv' no fue encontrado.")

with tabs[1]:
    st.subheader("💡 Recomendación personalizada")
    if os.path.exists('modelos/recomendador.pkl') and not enriched.empty:
        modelo = joblib.load('modelos/recomendador.pkl')
        muestra = enriched.drop(columns=["ID_Cliente", "Producto_Recomendado"], errors="ignore")
        entrada = muestra.sample(1, random_state=42)

        try:
            pred = modelo.predict(entrada)[0]
            proba = modelo.predict_proba(entrada)[0]
            cliente_id = enriched.iloc[entrada.index[0]]["ID_Cliente"]

            st.success(f"Cliente ID `{cliente_id}` → Recomendación: **{pred}**")

            st.markdown("### 🔍 Probabilidades por tipo de seguro")
            clases = modelo.classes_
            df_proba = pd.DataFrame({
                "Tipo de Seguro": clases,
                "Probabilidad (%)": (proba * 100).round(2)
            })
            st.bar_chart(df_proba.set_index("Tipo de Seguro"))
            for tipo, p in zip(clases, proba):
                st.write(f"- {tipo}: {round(p * 100, 2)}%")
        except Exception as e:
            st.error(f"⚠️ Error al predecir: {e}")
    else:
        st.info("🔎 El modelo entrenado no está disponible o falta el dataset enriquecido.")

with tabs[2]:
    st.subheader("🧠 Panel Estratégico del Cliente")
    if not enriched.empty:
        fecha_modificacion = os.path.getmtime('data/dataset_enriquecido.pkl')
        fecha_legible = pd.to_datetime(fecha_modificacion, unit='s')
        st.caption(f"📅 Última actualización del dataset enriquecido: {fecha_legible.strftime('%Y-%m-%d %H:%M:%S')}")

        columnas_requeridas = ['Razon_Social', 'Delegacion_Municipio', 'Giro_Empresa',
                               'Ingresos_Anuales', 'Email_Contacto', 'Telefono']
        faltantes = [col for col in columnas_requeridas if col not in enriched.columns]

        if faltantes:
            st.error(f"🚫 El dataset enriquecido no contiene las columnas requeridas: {', '.join(faltantes)}")
        else:
            mostrar_panel_cliente(enriched)
    else:
        st.warning("No se ha generado el dataset enriquecido aún. Usa el botón 🔃 para consolidar la información.")

with tabs[3]:
    st.subheader("📊 Estadísticas Demográficas y de Riesgo")
    if not enriched.empty:
        mostrar_estadisticas_demograficas(enriched)
    else:
        st.warning("⚠️ El dataset aún no está disponible para estadísticas demográficas.")