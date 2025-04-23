import streamlit as st
from PIL import Image

# Configuración inicial de la app
st.set_page_config(page_title="Predictor de Precios", layout="wide")

from explore_page import show_explore_page
from predict_page import show_predict_page

# Selector de página primero
with st.sidebar:
    st.title("🚘 Navegación")
    page = st.selectbox("Seleccione una opción:", ("Explorar", "Predecir"))

    st.markdown("---")  # Línea divisoria

    # Imagen de perfil y sección "Sobre mí"
    try:
        profile_img = Image.open("assets/Bo-Kolstrup.png")
        st.image(profile_img, width=200, use_container_width=True)  # Changed here
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen de perfil: {str(e)}")

    st.markdown("""
    ### Sobre mí
    Me apasiona aplicar ciencia de datos y machine learning para resolver desafíos reales de negocios.
    """)

# Mostrar la página correspondiente
if page == "Explorar":
    show_explore_page()
else:
    show_predict_page()