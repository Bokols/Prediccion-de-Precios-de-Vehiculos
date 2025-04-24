# **Predicción de Precios de Vehículos Usados Utilizando Aprendizaje Automático**

**Prueba el modelo [aquí](https://prediccion-de-precios-de-vehiculos-ijtuu6g5tvusyxuzts3fod.streamlit.app/):**

## **Introducción**

El objetivo de este proyecto es predecir el precio de vehículos usados basándose en varias características como el año del modelo, la marca, el estado, el tipo de combustible, el kilometraje, el tipo de transmisión y la categoría del vehículo. Se aplican técnicas de **Random Forest**, **Gradient Boosting** y **XGBoost** para desarrollar modelos predictivos. Utilizamos **GridSearchCV** para optimizar los hiperparámetros de estos modelos y evaluamos su rendimiento utilizando el **Error Cuadrático Medio Raíz (RMSE)**.

Este proyecto incluye varias etapas, desde el **Análisis Exploratorio de Datos (EDA)** hasta el **entrenamiento y optimización de modelos**, con enfoque en lograr el mejor modelo para predecir los precios de los vehículos.

## **Análisis Exploratorio de Datos (EDA)**

El Análisis Exploratorio de Datos (EDA) es un paso crucial para comprender la estructura y las relaciones dentro del conjunto de datos. A través de EDA, obtenemos información sobre la distribución de las características, su relación con la variable objetivo (`precio`) y los posibles problemas de datos que deben ser abordados.

### **1. Visión General de los Datos**

El conjunto de datos consta de varias características relacionadas con los atributos de los vehículos, que incluyen:

- **model_year**: Año del modelo del vehículo.
- **model**: Marca o modelo del vehículo.
- **condition**: Variable categórica que indica el estado del vehículo (por ejemplo, excelente, bueno, regular).
- **fuel**: Tipo de combustible utilizado por el vehículo (por ejemplo, gasolina, diésel).
- **odometer**: Kilometraje o distancia recorrida por el vehículo.
- **transmission**: Tipo de transmisión (manual o automática).
- **type**: Tipo de vehículo (por ejemplo, SUV, sedán, camión).

La variable objetivo es **price**, que es el precio que deseamos predecir a partir de estas características.

### **2. Limpieza y Preparación de los Datos**

Antes de comenzar cualquier análisis, primero aseguramos que el conjunto de datos esté limpio y listo para el análisis:

- **Manejo de Valores Faltantes**: Verificamos si hay valores faltantes en el conjunto de datos y aplicamos métodos adecuados para llenarlos o eliminarlos.
- **Eliminación de Duplicados**: Inspeccionamos y eliminamos registros duplicados que puedan afectar el análisis.
- **Detección de Valores Atípicos**: Los valores atípicos se visualizan utilizando diagramas de caja, particularmente para características como `odometer` y `model_year`.
- **Verificación de Tipos de Datos**: Confirmamos que las variables categóricas estén correctamente codificadas y que las variables numéricas sean tratadas como numéricas.

### **3. Estadísticas Resumidas**

Calculamos estadísticas descriptivas para cada característica para entender su distribución:

- **Tendencia Central**: Se calculan la media, mediana y moda para características numéricas como `odometer` y `price`.
- **Dispersión**: Se evalúan la desviación estándar y el rango para ver cuán dispersos están los valores.
- **Correlación**: Se calcula una matriz de correlación para explorar las relaciones entre las características y la variable objetivo (`price`).

### **4. Análisis Univariante**

Realizamos un análisis univariante para comprender la distribución de las características individuales:

- **Características Numéricas**: Examinamos la distribución de `price`, `odometer` y `model_year` mediante histogramas y gráficos de densidad.
- **Características Categóricas**: Para características como `condition`, `fuel` y `transmission`, utilizamos diagramas de barras para visualizar la frecuencia de cada categoría.

### **5. Análisis Bivariante**

Analizamos las relaciones entre las características individuales y la variable objetivo:

- **Características Numéricas**: Usamos diagramas de dispersión para explorar cómo características como `odometer` y `model_year` se relacionan con `price`.
- **Características Categóricas**: Se utilizan diagramas de caja para visualizar cómo las características categóricas como `condition`, `fuel` y `transmission` afectan la distribución del precio.

### **6. Importancia de las Características**

Para comprender qué características son más influyentes en la predicción del precio, utilizamos **Random Forest** para calcular la importancia de las características. Se espera que características como `odometer`, `condition` y `model_year` sean los predictores más significativos del precio.

### **7. Visualización de Relaciones**

Las visualizaciones son clave para descubrir patrones en los datos:

- **Distribución de Características Numéricas**: Se utilizan histogramas y diagramas de caja para entender la distribución de características como `price`, `odometer` y `model_year`.
- **Mapa de Calor de Correlación**: Un mapa de calor de la matriz de correlación ayuda a identificar relaciones entre características numéricas y la variable objetivo.
- **Frecuencia de Categorías**: Los diagramas de barras visualizan la frecuencia de características categóricas como `condition`, `fuel` y `transmission`.
- **Precio vs. Características**: Diagramas de dispersión y diagramas de caja se utilizan para mostrar la relación entre `price` y otras características como `odometer`, `model_year` y `condition`.

### **8. Ideas Derivadas del EDA**

A partir del EDA, obtenemos varias ideas que guían los siguientes pasos en el proceso de modelado:

- **Distribución del Precio**: La característica `price` tiene una distribución sesgada a la derecha, lo que indica que los vehículos con precios bajos son más comunes.
- **Estado y Precio**: Los vehículos en excelente estado tienen precios significativamente más altos que los de estado bueno o regular.
- **Año del Modelo y Precio**: Los vehículos de modelos más nuevos tienden a tener precios más altos, aunque hay excepciones para ciertas marcas o modelos.
- **Kilometraje y Precio**: Un mayor kilometraje generalmente se correlaciona con precios más bajos, pero algunos vehículos retienen su valor a pesar del alto kilometraje.

## **Entrenamiento y Optimización del Modelo**

Una vez completado el EDA y preparada la data, procedemos con la selección y optimización del modelo. Utilizamos varios modelos de aprendizaje automático para predecir los precios de los vehículos, incluyendo **Random Forest**, **Gradient Boosting** y **XGBoost**. El objetivo es identificar el modelo con mejor rendimiento a través de la optimización de hiperparámetros y validación cruzada.

### **1. Random Forest Regressor**

Entrenamos el modelo **Random Forest Regressor** para predecir el precio de los vehículos. Utilizamos **GridSearchCV** para ajustar los hiperparámetros, como `max_depth`, `n_estimators` y `min_samples_split`. El modelo se entrena y calculamos el **Error Cuadrático Medio Raíz (RMSE)** para evaluar su rendimiento.

### **2. Gradient Boosting Regressor**

De manera similar, entrenamos el **Gradient Boosting Regressor** utilizando los mejores hiperparámetros obtenidos mediante **GridSearchCV**. Evaluamos el rendimiento utilizando RMSE y comparamos los resultados con el modelo de Random Forest.

### **3. XGBoost Regressor**

El **XGBoost Regressor** también se entrena y sus hiperparámetros se optimizan mediante **GridSearchCV**. Se calcula el RMSE y comparamos el rendimiento con los otros modelos.

### **4. Evaluación del Modelo**

La selección final del modelo se basa en el rendimiento de RMSE. Entre los tres modelos probados, **Random Forest Regressor** alcanzó el mejor rendimiento con un RMSE de **$1,591.18**. Los otros modelos, **Gradient Boosting** y **XGBoost**, tuvieron un rendimiento ligeramente inferior, con RMSE de **$1,969.30** y **$3,580.22**, respectivamente.

## **Despliegue del Modelo: Predicción y Guardado del Modelo**

Una vez identificado el mejor modelo, lo desplegamos para realizar predicciones con nuevos datos. Preparamos un conjunto de datos de prueba con características de vehículos (por ejemplo, año del modelo, estado, tipo de combustible), transformamos las variables categóricas utilizando los **LabelEncoders** entrenados y predecimos el precio utilizando el **Random Forest Regressor**.

Además, el modelo entrenado y los codificadores se guardan utilizando **Pickle** para su uso posterior, asegurando que el modelo pueda ser cargado y utilizado para futuras predicciones sin necesidad de reentrenarlo.


## **Tecnologías Utilizadas**

- **Python**: Lenguaje de programación utilizado para el análisis de datos y la construcción de modelos.
- **Pandas**: Biblioteca para la manipulación y análisis de datos.
- **NumPy**: Utilizado para operaciones matemáticas y manipulación de matrices.
- **Scikit-learn**: Herramienta principal para la creación y evaluación de modelos de machine learning (incluyendo RandomForestRegressor, GradientBoostingRegressor, y XGBRegressor).
- **Matplotlib** y **Seaborn**: Bibliotecas para visualización de datos.
- **Pickle**: Utilizado para guardar y cargar los modelos entrenados y los codificadores.

## **Conclusión**

Este proyecto ha demostrado cómo aplicar técnicas de aprendizaje automático para predecir el precio de vehículos usados. A través del uso de **Random Forest**, **Gradient Boosting**, y **XGBoost**, hemos identificado que **Random Forest** es el modelo más adecuado para esta tarea, logrando el mejor rendimiento con un RMSE de **$1,591.18**.

**Prueba el modelo [aquí](https://prediccion-de-precios-de-vehiculos-ijtuu6g5tvusyxuzts3fod.streamlit.app/):**
