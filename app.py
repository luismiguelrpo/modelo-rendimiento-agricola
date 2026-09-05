from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DIRECTORIO = Path(__file__).resolve().parent
VARIABLES_ENTRADA = ["Departamento", "Grupo cultivo", "Tipo periodo"]


st.set_page_config(
    page_title="Predicción del rendimiento agrícola",
    page_icon="🌱",
    layout="centered",
)


@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load(DIRECTORIO / "modelo_final.joblib")
    encoder = joblib.load(DIRECTORIO / "onehot_encoder.joblib")
    escalador = joblib.load(DIRECTORIO / "minmax_scaler.joblib")
    return modelo, encoder, escalador


@st.cache_data
def cargar_datos_referencia():
    return joblib.load(DIRECTORIO / "datos_referencia.joblib")


try:
    modelo, encoder, escalador = cargar_artefactos()
    datos_referencia = cargar_datos_referencia()
except FileNotFoundError as error:
    st.error(
        "No fue posible cargar todos los archivos del modelo. Verifica que "
        "modelo_final.joblib, onehot_encoder.joblib, minmax_scaler.joblib y "
        "datos_referencia.joblib estén en la misma carpeta que app.py."
    )
    st.exception(error)
    st.stop()


columnas_faltantes = [
    columna
    for columna in VARIABLES_ENTRADA
    if columna not in datos_referencia.columns
]

if columnas_faltantes:
    st.error(
        "Los archivos .joblib todavía corresponden a la versión anterior "
        "del modelo. Vuelve a generarlos en Colab usando la variable "
        "'Tipo periodo' y reemplázalos en GitHub."
    )
    st.stop()


st.title("Predicción del rendimiento agrícola")
st.write(
    "Esta aplicación estima el rendimiento esperado a partir del "
    "departamento, el grupo de cultivo y el tipo de periodo seleccionado."
)

departamentos = sorted(
    datos_referencia["Departamento"].dropna().astype(str).unique()
)
departamento = st.selectbox("Departamento", departamentos)

referencia_departamento = datos_referencia[
    datos_referencia["Departamento"].astype(str) == departamento
]

grupos_cultivo = sorted(
    referencia_departamento["Grupo cultivo"].dropna().astype(str).unique()
)
grupo_cultivo = st.selectbox("Grupo de cultivo", grupos_cultivo)

referencia_filtrada = referencia_departamento[
    referencia_departamento["Grupo cultivo"].astype(str) == grupo_cultivo
]

orden_tipos = ["A", "B", "Completo"]
tipos_en_datos = set(
    referencia_filtrada["Tipo periodo"].dropna().astype(str).unique()
)
tipos_disponibles = [
    tipo for tipo in orden_tipos if tipo in tipos_en_datos
]

tipo_periodo = st.selectbox(
    "Tipo de periodo",
    tipos_disponibles,
    help=(
        "A y B corresponden a periodos semestrales. Completo corresponde "
        "a cultivos reportados para el año completo."
    ),
)

predecir = st.button(
    "Predecir rendimiento",
    type="primary",
    use_container_width=True,
)

if predecir:
    nuevo_registro = pd.DataFrame(
        [
            {
                "Departamento": departamento,
                "Grupo cultivo": grupo_cultivo,
                "Tipo periodo": tipo_periodo,
            }
        ],
        columns=VARIABLES_ENTRADA,
    )

    registro_codificado = encoder.transform(nuevo_registro)
    if hasattr(registro_codificado, "toarray"):
        registro_codificado = registro_codificado.toarray()

    registro_escalado = escalador.transform(registro_codificado)
    rendimiento_predicho = float(modelo.predict(registro_escalado)[0])

    resultado = nuevo_registro.copy()
    resultado["Rendimiento predicho (t/ha)"] = rendimiento_predicho

    st.success("Predicción realizada correctamente")
    st.metric(
        "Rendimiento estimado",
        f"{rendimiento_predicho:.2f} t/ha",
    )

    st.write("Datos utilizados para la predicción")
    st.dataframe(
        resultado.round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "El resultado corresponde a una predicción directa del modelo para "
        "la combinación seleccionada y no a un promedio de periodos históricos."
    )
