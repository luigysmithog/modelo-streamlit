import streamlit as st
from transformers import pipeline

# Configuración de página
st.set_page_config(page_title="Análisis de Sentimiento", page_icon="💬", layout="centered")

# Estilo con CSS para mejorar apariencia
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

# Título personalizado
st.markdown('<div class="title">💬 Análisis de Sentimiento Multilinguaje para calificar un producto</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;">Escribe una frase y el modelo intentará detectar si el sentimiento es positivo, negativo o neutral.</div>', unsafe_allow_html=True)

# Entrada de texto
with st.container():
    text_input = st.text_area("✏️ Ingresa tu texto aquí:", height=150)

# Cargar modelo con caché
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = load_model()

# Interpretación del label
def interpretar_sentimiento(label):
    interpretaciones = {
        "1 star": ("Muy negativo 😡", "#ff4b4b"),
        "2 stars": ("Negativo 🙁", "#ff944d"),
        "3 stars": ("Neutral 😐", "#f0ad4e"),
        "4 stars": ("Positivo 🙂", "#5bc0de"),
        "5 stars": ("Muy positivo 😄", "#5cb85c")
    }
    return interpretaciones.get(label, ("Desconocido", "#cccccc"))

# Botón de análisis
if st.button("🔍 Analizar Sentimiento"):
    if text_input.strip():
        result = classifier(text_input)
        label = result[0]['label']
        score = result[0]['score']
        interpretacion, color = interpretar_sentimiento(label)

        # Mostrar resultado
        st.markdown(f"""
            <div class="box">
                <div class="result" style="color:{color};">
                    📊 Interpretación: {interpretacion}
                </div>
                <p>🔎 Confianza del modelo: <b>{score:.2%}</b></p>
                <p>⭐ Etiqueta original del modelo: <code>{label}</code></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Por favor, escribe un texto para analizar.")
