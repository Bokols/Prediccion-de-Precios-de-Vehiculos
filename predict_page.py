# Página de Predicción de Precios de Vehículos Mejorada
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from PIL import Image

# ===== Sidebar con Perfil =====
with st.sidebar:
    st.title("🔧 Configuración")

    # Cargar imagen de perfil local
    try:
        profile_img = Image.open("assets/Bo-Kolstrup.png")
        st.image(profile_img, width=200, use_column_width='auto')
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen de perfil: {str(e)}")

    # Sección "Sobre mí"
    st.markdown("""
    ### Sobre mí
    Apasionado por aplicar ciencia de datos y machine learning para resolver desafíos reales de negocio.
    """)

# ===== Carga del Modelo con Validación =====
@st.cache_resource
def load_model():
    try:
        with open("model_compressed.pbz2", "rb") as file:
            model_data = pickle.load(file)

        required = ['model', 'le_model', 'le_condition', 'le_fuel', 'le_transmission', 'le_type']
        for component in required:
            if component not in model_data:
                st.error(f"Componente faltante en el modelo: {component}")
                st.stop()

        return model_data
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

model_data = load_model()
model = model_data["model"]
encoders = {k: v for k, v in model_data.items() if k.startswith('le_')}
expected_features = getattr(model, 'feature_names_in_', [
    'model_year', 'model', 'condition', 'fuel', 
    'odometer', 'transmission', 'type'
])

# ===== Página de Predicción =====
def show_predict_page():
    st.title("🚗 Calculadora de Precio de Vehículos")
    st.markdown("""
    Esta herramienta predice el precio estimado de un vehículo usado en EE.UU.  
    Complete los campos con la información real del vehículo y haga clic en **Calcular Precio**.
    """)

    with st.form("formulario_prediccion"):
        col1, col2 = st.columns(2)

        with col1:
            model_year = st.slider(
                "Año del modelo",
                min_value=1990,
                max_value=datetime.now().year,
                value=2015,
                help="Año en el que se fabricó el vehículo."
            )

            car_model = st.selectbox(
                "Marca del vehículo",
                sorted(encoders["le_model"].classes_),
                help="Seleccione el modelo o marca del vehículo (ej. Corolla, Civic, F-150)."
            )

            condition = st.selectbox(
                "Condición del vehículo",
                sorted(encoders["le_condition"].classes_),
                help="Estado general del vehículo según su uso y desgaste."
            )

        with col2:
            odometer = st.slider(
                "Kilometraje (en kilómetros)",
                min_value=0,
                max_value=300000,
                value=50000,
                step=1000,
                help="Distancia total recorrida por el vehículo."
            )

            fuel = st.selectbox(
                "Tipo de combustible",
                sorted(encoders["le_fuel"].classes_),
                help="Tipo de combustible utilizado por el vehículo (gasolina, diésel, eléctrico, etc.)."
            )

            transmission = st.selectbox(
                "Tipo de transmisión",
                sorted(encoders["le_transmission"].classes_),
                help="Transmisión automática o manual del vehículo."
            )

            car_type = st.selectbox(
                "Tipo de vehículo",
                sorted(encoders["le_type"].classes_),
                help="Tipo de carrocería o clase del vehículo (sedán, SUV, camioneta, etc.)."
            )

        submitted = st.form_submit_button("🔍 Calcular Precio")

    if submitted:
        try:
            input_data = {
                'model_year': model_year,
                'model': car_model,
                'condition': condition,
                'fuel': fuel,
                'odometer': odometer,
                'transmission': transmission,
                'type': car_type
            }

            # Codificar variables categóricas
            X = pd.DataFrame([input_data])
            for col in ['model', 'condition', 'fuel', 'transmission', 'type']:
                X[col] = encoders[f"le_{col}"].transform(X[col])

            X = X[expected_features]

            # Predicción
            price = model.predict(X)[0]
            st.success(f"## 💵 Precio estimado: ${price:,.2f}")

            with st.expander("📊 Ver detalles de la predicción"):
                st.json(input_data)
                st.markdown(f"**Orden esperado por el modelo:** `{list(expected_features)}`")

        except Exception as e:
            st.error(f"❌ Error durante la predicción: {str(e)}")
            st.info("Por favor verifique los valores ingresados.")

# ===== Punto de entrada =====
if __name__ == "__main__":
    show_predict_page()
