import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_estadisticas_demograficas(df):
    st.header("📊 Estadísticas Demográficas y de Riesgo")

    if "Edad" in df.columns:
        st.subheader("🎂 Distribución de edades")
        fig_edad = px.histogram(df, x="Edad", nbins=20, color="Sexo", barmode="overlay")
        st.plotly_chart(fig_edad, use_container_width=True)

    if "Sexo" in df.columns:
        st.subheader("👥 Proporción por sexo")
        sexo_data = df["Sexo"].value_counts().reset_index()
        sexo_data.columns = ["Sexo", "Cantidad"]
        fig_sexo = px.pie(sexo_data, names="Sexo", values="Cantidad", hole=0.4)
        st.plotly_chart(fig_sexo, use_container_width=True)

    if "Delegacion_Municipio" in df.columns:
        st.subheader("📍 Siniestros por delegación")
        mapa = df.groupby("Delegacion_Municipio").size().reset_index(name="Casos")
        fig_mapa = px.bar(mapa.sort_values("Casos", ascending=False),
                          x="Delegacion_Municipio", y="Casos")
        st.plotly_chart(fig_mapa, use_container_width=True)

    if "Historial_Médico" in df.columns:
        st.subheader("🧬 Enfermedades registradas")
        enfermedades = df["Historial_Médico"].value_counts().reset_index()
        enfermedades.columns = ["Condición", "Cantidad"]
        st.dataframe(enfermedades)