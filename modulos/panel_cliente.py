import streamlit as st
import pandas as pd

def mostrar_panel_cliente(enriched):
    tab1, tab2 = st.tabs(["📋 Cliente individual", "📈 Tendencias comerciales"])

    with tab1:
        st.markdown("### 🔎 Información por cliente")
        cliente_id = st.selectbox("Selecciona un cliente por ID:", enriched["ID_Cliente"].unique())
        registro = enriched[enriched["ID_Cliente"] == cliente_id]

        if not registro.empty:
            cliente = registro.iloc[0]
            st.markdown(f"**Razon Social:** {cliente['Razon_Social']}")
            st.markdown(f"**Delegación:** {cliente['Delegacion_Municipio']}")
            st.markdown(f"**Giro empresarial:** {cliente['Giro_Empresa']}")
            st.markdown(f"**Ingresos anuales:** ${cliente['Ingresos_Anuales']:,.0f}")
            st.markdown(f"**Contacto:** 📧 {cliente['Email_Contacto']} | ☎️ {cliente['Telefono']}")

            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("🔮 Probabilidad de conversión", f"{cliente['Probabilidad_Conversion']}%")
            col2.metric("🔥 Índice de riesgo", cliente["Indice_Riesgo"])
            st.metric("💼 Producto recomendado", cliente["Producto_Recomendado"])
            st.metric("📣 Acción sugerida", cliente["Accion_Sugerida"])
            st.metric("📦 Pólizas activas", int(cliente["Num_Polizas"]))
            st.metric("🧾 Reclamaciones registradas", int(cliente["Num_Reclamaciones"]))

    with tab2:
        st.markdown("### 📈 Tendencias comerciales agregadas")
        col1, col2, col3 = st.columns(3)
        col1.metric("Clientes segmento alto", (enriched["Segmento"] == "Alto").sum())
        col2.metric("Promedio prima cliente", f"${enriched['Prima_Total'].mean():,.0f}")
        col3.metric("Reclamaciones totales", enriched["Num_Reclamaciones"].sum())

        st.markdown("#### Distribución de productos recomendados")
        prod_data = enriched["Producto_Recomendado"].value_counts().reset_index()
        prod_data.columns = ["Producto", "Clientes"]
        st.bar_chart(prod_data.set_index("Producto"))

        st.markdown("#### Índice de riesgo por delegación")
        riesgo_data = enriched.groupby("Delegacion_Municipio")["Indice_Riesgo"].value_counts().unstack().fillna(0)
        st.dataframe(riesgo_data.style.highlight_max(axis=1), use_container_width=True)