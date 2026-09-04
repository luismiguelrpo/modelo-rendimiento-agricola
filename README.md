# Modelo de rendimiento agrícola

Este proyecto tiene como objetivo desarrollar un modelo de machine learning para predecir el rendimiento agrícola utilizando información de las Evaluaciones Agropecuarias Municipales EVA.

El desarrollo sigue la metodología CRISP-DM, incluyendo las etapas de comprensión del problema, comprensión de los datos, preparación, modelado, evaluación y despliegue.

## Fuente de datos

La información utilizada proviene del portal Datos Abiertos Colombia y corresponde a la base agrícola de las Evaluaciones Agropecuarias Municipales EVA.

## Variables seleccionadas

Las variables utilizadas para el modelamiento son

- Departamento
- Grupo cultivo
- Periodo
- Rendimiento

La variable Rendimiento corresponde a la variable objetivo.

## Modelos a evaluar

Se evaluarán diferentes algoritmos de regresión

- Regresión Lineal
- Árbol de Regresión
- KNN Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

También se evaluarán modelos de ensamble.

## Métricas

La métrica principal será RMSE.

De forma complementaria se utilizarán

- MAE
- R²

## Herramientas utilizadas

- Python
- Pandas
- Scikit-learn
- Google Colab
- ydata-profiling
- Streamlit

## Estructura del proyecto

- `A_base_datos_original_EVA.xlsx`
- `B_preparacion_datos_EVA.ipynb`
- `C_reporte_ydata_EVA.html`
- `D_modelamiento_EVA.ipynb`
- `E_modelo_final_EVA.pkl`
- `F_app.py`
- `G_requirements.txt`

## Objetivo final

Construir un modelo capaz de estimar el rendimiento agrícola a partir del departamento, el grupo de cultivo y el periodo agrícola.
