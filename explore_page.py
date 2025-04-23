import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

@st.cache_data
def load_data():
    # Updated to read from GitHub URL
    url = 'https://raw.githubusercontent.com/Bokols/Prediccion-de-Precios-de-Vehiculos/main/data/vehicles_us.csv'
    df = pd.read_csv(url)
    
    # Data cleaning and preprocessing
    df['is_4wd'] = df['is_4wd'].fillna(0)
    df['paint_color'] = df['paint_color'].fillna('unknown')

    numeric_cols = ['model_year', 'odometer', 'cylinders']
    for col in numeric_cols:
        df[col] = df.groupby('model')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    brand_mapping = {
        'ford': 'Ford', 'chevrolet': 'Chevrolet', 'toyota': 'Toyota',
        'honda': 'Honda', 'nissan': 'Nissan', 'jeep': 'Jeep',
        'bmw': 'BMW', 'mercedes': 'Mercedes', 'hyundai': 'Hyundai'
    }
    df['model'] = df['model'].str.lower().replace(brand_mapping, regex=True)

    df['date_posted'] = pd.to_datetime(df['date_posted'])
    df['year_posted'] = df['date_posted'].dt.year
    df['month_posted'] = df['date_posted'].dt.month
    df['vehicle_age'] = df['year_posted'] - df['model_year']
    df['mileage_per_year'] = df['odometer'] / np.maximum(1, df['vehicle_age'])

    return df

def show_explore_page():
    df = load_data()

    st.title("🚗 Explorador de Datos de Vehículos")
    st.markdown("Explora datos de vehículos usados en EE.UU. usando filtros interactivos y gráficos intuitivos.")

    st.subheader("🔍 Filtros")
    st.markdown("Use los siguientes filtros para personalizar los datos visualizados:")

    available_models = sorted(df['model'].unique())
    possible_defaults = ['Ford', 'Toyota', 'Chevrolet', 'Honda', 'Nissan']
    default_models = [m for m in possible_defaults if m in available_models][:3]

    col1, col2 = st.columns(2)

    selected_models = col1.multiselect(
        "Seleccione Modelos",
        options=available_models,
        default=default_models if default_models else available_models[:3],
        help="Seleccione uno o más modelos de vehículos para filtrar los datos."
    )

    min_price, max_price = int(df['price'].min()), int(df['price'].max())
    default_min = max(min_price, int(df['price'].quantile(0.25)))
    default_max = min(max_price, int(df['price'].quantile(0.75)))

    price_range = col2.slider(
        "Rango de Precios ($)",
        min_value=min_price,
        max_value=max_price,
        value=(default_min, default_max),
        help="Ajuste este control para mostrar vehículos dentro de un rango de precios específico."
    )

    available_types = df['type'].unique()
    selected_types = st.multiselect(
        "Tipos de Vehículo",
        options=available_types,
        default=available_types,
        help="Filtra los datos por tipos como SUV, sedán, camioneta, etc."
    )

    if not selected_models:
        st.warning("Por favor, seleccione al menos un modelo.")
        filtered_df = df.iloc[:0]
    else:
        filtered_df = df[
            (df['model'].isin(selected_models)) &
            (df['price'].between(*price_range)) &
            (df['type'].isin(selected_types))
        ]

    if not filtered_df.empty:
        st.markdown(f"📊 **{len(filtered_df)} vehículos encontrados** que coinciden con sus criterios.")

        cols = st.columns(3)
        cols[0].metric("Precio Promedio", f"${filtered_df['price'].mean():,.0f}")
        cols[1].metric("Kilometraje Promedio", f"{filtered_df['odometer'].mean():,.0f} km")
        cols[2].metric("Edad Promedio", f"{filtered_df['vehicle_age'].mean():.1f} años")

        tab1, tab2, tab3 = st.tabs(["Distribución de Precios", "Edad vs Kilometraje", "Condición del Vehículo"])

        with tab1:
            st.subheader("Distribución de Precios")
            st.markdown("Compara el precio promedio entre diferentes modelos y tipos de vehículos.")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            model_prices = filtered_df.groupby('model')['price'].mean().sort_values()
            ax1.barh(model_prices.index, model_prices.values)
            ax1.set_title("Por Modelo")
            type_prices = filtered_df.groupby('type')['price'].mean().sort_values()
            ax2.barh(type_prices.index, type_prices.values)
            ax2.set_title("Por Tipo de Vehículo")
            st.pyplot(fig)

        with tab2:
            st.subheader("Relación Edad vs Kilometraje")
            st.markdown("Este gráfico muestra cómo varía el precio según la edad del vehículo y su kilometraje.")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                filtered_df['vehicle_age'], filtered_df['odometer'],
                c=filtered_df['price'], cmap='viridis', alpha=0.6)
            ax.set_xlabel("Edad del Vehículo (años)")
            ax.set_ylabel("Kilometraje")
            plt.colorbar(scatter, label="Precio ($)")
            st.pyplot(fig)

        with tab3:
            st.subheader("Condición del Vehículo")
            st.markdown("Muestra la distribución de los vehículos por condición declarada.")
            condition_order = ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']
            condition_data = filtered_df['condition'].value_counts().reindex(condition_order).fillna(0)
            st.bar_chart(condition_data)
    else:
        st.warning("No hay vehículos que coincidan con los filtros seleccionados.")

if __name__ == '__main__':
    show_explore_page()