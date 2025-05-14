import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

# Configuración de página
st.set_page_config(page_title="Análisis de Sentimiento", page_icon="💬", layout="centered")

# Estilo CSS personalizado
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .box {
        border: 2px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Título y subtítulo
st.markdown('<div class="title">💬 Análisis de Sentimiento Multilinguaje para calificar un producto</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;">Escribe una frase y el modelo intentará detectar si el sentimiento es positivo, negativo o neutral.</div>', unsafe_allow_html=True)

# Entrada de texto
with st.container():
    text_input = st.text_area("✏️ Ingresa tu texto aquí:", height=150)

# Cargar modelo con todas las probabilidades
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", return_all_scores=True)

classifier = load_model()

# Interpretación de sentimiento
def interpretar_sentimiento(label):
    interpretaciones = {
        "1 star": ("Muy negativo 😡", "#ff4b4b"),
        "2 stars": ("Negativo 🙁", "#ff944d"),
        "3 stars": ("Neutral 😐", "#f0ad4e"),
        "4 stars": ("Positivo 🙂", "#5bc0de"),
        "5 stars": ("Muy positivo 😄", "#5cb85c")
    }
    return interpretaciones.get(label, ("Desconocido", "#cccccc"))

# Análisis al presionar botón
if st.button("🔍 Analizar Sentimiento"):
    if text_input.strip():
        results = classifier(text_input)[0]  # lista de resultados por clase
        df = pd.DataFrame(results)
        df['interpretacion'], df['color'] = zip(*df['label'].map(interpretar_sentimiento))

        mejor = df.loc[df['score'].idxmax()]
        interpretacion, color = interpretar_sentimiento(mejor['label'])

        # Mostrar resultado principal
        st.markdown(f"""
            <div class="box">
                <div class="result" style="color:{color};">
                    📊 Interpretación: {interpretacion}
                </div>
                <p>🔎 Confianza del modelo: <b>{mejor['score']:.2%}</b></p>
                <p>⭐ Etiqueta original del modelo: <code>{mejor['label']}</code></p>
            </div>
        """, unsafe_allow_html=True)

        # Mostrar gráfica de barras
        st.markdown("### 📈 Distribución completa del modelo:")
        chart = alt.Chart(df).mark_bar(size=40).encode(
            x=alt.X('interpretacion', title='Sentimiento'),
            y=alt.Y('score', title='Confianza'),
            color=alt.Color('color', scale=None, legend=None),
            tooltip=['label', 'score']
        ).properties(width=600, height=300)

        st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("⚠️ Por favor, escribe un texto para analizar.")
