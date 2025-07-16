import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker('es_MX')
np.random.seed(42)
random.seed(42)

# 📌 1. CLIENTES
def generar_clientes(n=100):
    delegaciones = ["Benito Juárez", "Cuauhtémoc", "Iztapalapa", "Coyoacán", "Miguel Hidalgo", "Tlalpan", "Azcapotzalco"]
    giros = ["Tecnología", "Educación", "Alimentos", "Seguros", "Construcción", "Transporte", "Salud", "Finanzas", "Logística", "Manufactura"]
    
    clientes = []
    for i in range(n):
        razon = fake.company() + " " + random.choice(["del Norte", "Integral", "Express", "y Asociados", "Latinoamérica", "MX"])
        clientes.append({
            "ID_Cliente": 1000 + i,
            "Razon_Social": razon,
            "Delegacion_Municipio": random.choice(delegaciones),
            "Giro_Empresa": random.choice(giros),
            "Num_Empleados": random.randint(10, 500),
            "Ingresos_Anuales": random.randint(500000, 25000000),
            "Fecha_Alta": fake.date_between(start_date='-15y', end_date='-1d').strftime("%Y-%m-%d"),
            "Email_Contacto": fake.company_email(),
            "Telefono": fake.phone_number()
        })
    df = pd.DataFrame(clientes)
    df.to_csv("data/clientes.csv", index=False)
    df.to_pickle("data/clientes.pkl")

# 📌 2. POLIZAS
def generar_polizas(n=250, clientes_ids=None):
    tipos = ["Seguro Empresarial", "Seguro Médico", "Seguro Vehicular", "Seguro Mixto", "Seguro de Daños", "Seguro Contra Robo"]
    polizas = []
    for i in range(n):
        cliente = random.choice(clientes_ids)
        inicio = fake.date_between(start_date='-2y', end_date='today')
        fin = inicio + timedelta(days=365)
        polizas.append({
            "ID_Seguro": 2000 + i,
            "ID_Cliente": cliente,
            "Tipo_Seguro": random.choice(tipos),
            "Prima_Anual": random.randint(10000, 75000),
            "Fecha_Inicio": inicio.strftime("%Y-%m-%d"),
            "Fecha_Vencimiento": fin.strftime("%Y-%m-%d"),
            "Cobertura_Total": random.randint(100000, 1000000)
        })
    df = pd.DataFrame(polizas)
    df.to_csv("data/polizas.csv", index=False)
    df.to_pickle("data/polizas.pkl")

# 📌 3. RECLAMACIONES
def generar_reclamaciones(n=150, polizas_df=None):
    descripciones = [
        "Colisión múltiple", "Incendio total", "Robo de vehículo", "Daño estructural", "Hurto domiciliario", 
        "Vandalismo urbano", "Accidente vial", "Explosión interna", "Falla mecánica grave", "Inundación severa",
        "Daño por granizo", "Accidente en obra", "Derrumbe estructural", "Daños por manifestación", "Daño por terremoto"
    ]
    estados = ["Aprobada", "Rechazada", "En Proceso"]
    reclamaciones = []
    for i in range(n):
        poliza = polizas_df.sample(1).iloc[0]
        reclamaciones.append({
            "ID_Reclamacion": 3000 + i,
            "ID_Seguro": poliza["ID_Seguro"],
            "ID_Cliente": poliza["ID_Cliente"],
            "Fecha_Reclamacion": fake.date_between(start_date='-3y', end_date='today').strftime("%Y-%m-%d"),
            "Descripcion_Reclamacion": random.choice(descripciones),
            "Monto_Reclamacion": round(random.uniform(8000, 60000), 2),
            "Estado_Reclamacion": random.choice(estados)
        })
    df = pd.DataFrame(reclamaciones)
    df.to_csv("data/reclamaciones.csv", index=False)
    df.to_pickle("data/reclamaciones.pkl")

# 🚀 Ejecutar generación
generar_clientes()
clientes_df = pd.read_csv("data/clientes.csv")
generar_polizas(clientes_ids=clientes_df["ID_Cliente"].tolist())
polizas_df = pd.read_csv("data/polizas.csv")
generar_reclamaciones(polizas_df=polizas_df)

print("✅ Bases generadas: clientes.csv, polizas.csv, reclamaciones.csv + .pkl")