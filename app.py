from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DIRECTORIO = Path(__file__).resolve().parent
VARIABLES_ENTRADA = ["Departamento", "Grupo cultivo", "Periodo"]


st.set_page_config(
    page_title="Predicción del rendimiento agrícola",
    page_icon="🌱",
    layout="centered",
)


@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load(DIRECTORIO / "modelo_mlp.joblib")
    encoder = joblib.load(DIRECTORIO / "onehot_encoder.joblib")
    escalador = joblib.load(DIRECTORIO / "minmax_scaler.joblib")
    return modelo, encoder, escalador


@st.cache_data
def cargar_datos_referencia():
    return joblib.load(DIRECTORIO / "datos_referencia.joblib")


def ordenar_periodos(valores):
    return sorted(valores, key=lambda valor: (int(str(valor)[:4]), str(valor)))


modelo, encoder, escalador = cargar_artefactos()
datos_referencia = cargar_datos_referencia()

st.title("Predicción del rendimiento agrícola")
st.write(
    "Esta aplicación estima el rendimiento esperado a partir del "
    "departamento, el grupo de cultivo y el tipo de periodo seleccionado."
)

departamentos = sorted(datos_referencia["Departamento"].astype(str).unique())

departamento = st.selectbox("Departamento", departamentos)

referencia_departamento = datos_referencia[
    datos_referencia["Departamento"].astype(str) == departamento
]

grupos_cultivo = sorted(
    referencia_departamento["Grupo cultivo"].astype(str).unique()
)
grupo_cultivo = st.selectbox("Grupo de cultivo", grupos_cultivo)

referencia_filtrada = referencia_departamento[
    referencia_departamento["Grupo cultivo"].astype(str) == grupo_cultivo
]

periodos_disponibles = referencia_filtrada["Periodo"].astype(str).unique()
tipos_disponibles = []

if any(periodo.endswith("A") for periodo in periodos_disponibles):
    tipos_disponibles.append("A")
if any(periodo.endswith("B") for periodo in periodos_disponibles):
    tipos_disponibles.append("B")
if any(periodo.isdigit() for periodo in periodos_disponibles):
    tipos_disponibles.append("Completo")

tipo_periodo = st.selectbox(
    "Tipo de periodo",
    tipos_disponibles,
    help=(
        "A y B corresponden a periodos semestrales. Completo corresponde "
        "a cultivos permanentes reportados para el año completo."
    ),
)

predecir = st.button(
    "Predecir rendimiento",
    type="primary",
    use_container_width=True,
)

if predecir:
    if tipo_periodo == "A":
        periodos_modelo = [
            periodo
            for periodo in periodos_disponibles
            if periodo.endswith("A")
        ]
    elif tipo_periodo == "B":
        periodos_modelo = [
            periodo
            for periodo in periodos_disponibles
            if periodo.endswith("B")
        ]
    else:
        periodos_modelo = [
            periodo
            for periodo in periodos_disponibles
            if periodo.isdigit()
        ]

    periodos_modelo = ordenar_periodos(periodos_modelo)

    if not periodos_modelo:
        st.error(
            "No existen registros históricos para la combinación "
            "y el tipo de periodo seleccionados."
        )
    else:
        nuevos_registros = pd.DataFrame(
            {
                "Departamento": departamento,
                "Grupo cultivo": grupo_cultivo,
                "Periodo": periodos_modelo,
            },
            columns=VARIABLES_ENTRADA,
        )

        registros_codificados = encoder.transform(nuevos_registros)
        if hasattr(registros_codificados, "toarray"):
            registros_codificados = registros_codificados.toarray()

        registros_escalados = escalador.transform(registros_codificados)
        predicciones = modelo.predict(registros_escalados)

        resultado = nuevos_registros.copy()
        resultado["Rendimiento predicho"] = predicciones
        prediccion_promedio = float(predicciones.mean())

        st.success("Predicción realizada correctamente")
        st.metric(
            "Rendimiento promedio estimado",
            f"{prediccion_promedio:.2f} t/ha",
        )

        st.caption(
            f"Promedio calculado a partir de {len(periodos_modelo)} "
            f"periodos históricos del tipo {tipo_periodo}."
        )

        st.write("Predicciones utilizadas para calcular el promedio")
        st.dataframe(
            resultado,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "El resultado representa un promedio histórico basado en las "
            "categorías disponibles en la base EVA."
        )
